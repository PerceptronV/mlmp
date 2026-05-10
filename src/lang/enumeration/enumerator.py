"""Core bottom-up enumerator with observational equivalence pruning."""

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

from ..grammar import Grammar, DefaultGrammar, T1, T2
from ..ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode, IntHoleNode,
)
from ..compiler import JITCompiler
from ..type_utils import (
    CallableOrig, get_args, get_origin,
    SubstitutionTable, substitute_type_vars, matchable, TypeType,
)
from ..utils import program_size, resolve_type, compute_valid_instantiations, PROBE_VALUES, RANDINT_PROBE_SEQUENCE
from .fingerprint import Fingerprint, FingerprintTable, make_hashable, FAIL, compute_fingerprint
from .filters import passes_quality_filter
from .test_suite import DEFAULT_TEST_SUITE


@dataclass
class TypedProgram:
    """A program with its type and fingerprint."""
    ast: ASTNode
    type: TypeType
    fingerprint: Fingerprint | None
    size: int
    substitution: list[int] | None = None


class ProgramBank:
    """
    Stores semantically distinct programs indexed by type and size.

    Access pattern: bank[resolved_type][size] -> list[TypedProgram]
    """

    def __init__(self):
        self._bank: dict[TypeType, dict[int, list[TypedProgram]]] = defaultdict(lambda: defaultdict(list))
        self._fingerprint_table: dict[TypeType, FingerprintTable] = defaultdict(FingerprintTable)

    def add(self, prog: TypedProgram) -> bool:
        """Add program if its fingerprint is novel for its type. Returns True if added."""
        if prog.fingerprint is None:
            # No fingerprint — just add (e.g., lambda param placeholders)
            self._bank[prog.type][prog.size].append(prog)
            return True
        fp_table = self._fingerprint_table[prog.type]
        if fp_table.insert(prog.fingerprint, prog.ast):
            self._bank[prog.type][prog.size].append(prog)
            return True
        return False

    def get(self, type_: TypeType, size: int) -> list[TypedProgram]:
        """Get all programs of a given type and exact size."""
        return self._bank.get(type_, {}).get(size, [])

    def get_up_to(self, type_: TypeType, max_size: int) -> list[TypedProgram]:
        """Get all programs of a given type up to a given size."""
        result = []
        for s in range(1, max_size + 1):
            result.extend(self.get(type_, s))
        return result

    def contains_fingerprint(self, type_: TypeType, fp: Fingerprint) -> bool:
        """Check if a fingerprint exists for a given type."""
        return self._fingerprint_table[type_].contains(fp)

    def count(self) -> int:
        """Total number of stored programs."""
        return sum(
            len(progs)
            for by_size in self._bank.values()
            for progs in by_size.values()
        )


class ContextualBank:
    """
    Program bank parameterised by a variable context.

    Child banks inherit from parents, mirroring lexical scoping.
    get() returns local + parent results. add_local() deduplicates
    against the local fingerprint table only (parent fingerprints
    have different lengths due to different probe combinations).
    """

    def __init__(self, parent: 'ProgramBank | ContextualBank | None' = None):
        self.parent = parent
        self._local: dict[TypeType, dict[int, list[TypedProgram]]] = defaultdict(lambda: defaultdict(list))
        self._local_fingerprints: dict[TypeType, FingerprintTable] = defaultdict(FingerprintTable)

    def get(self, type_: TypeType, size: int) -> list[TypedProgram]:
        """Get programs from this context AND all ancestor contexts."""
        results = list(self._local.get(type_, {}).get(size, []))
        if self.parent is not None:
            results.extend(self.parent.get(type_, size))
        return results

    def add_local(self, prog: TypedProgram) -> bool:
        """Add a program new to this context level. Checks local fingerprints only."""
        if prog.fingerprint is None:
            self._local[prog.type][prog.size].append(prog)
            return True
        fp_table = self._local_fingerprints[prog.type]
        if fp_table.insert(prog.fingerprint, prog.ast):
            self._local[prog.type][prog.size].append(prog)
            return True
        return False

    def contains_fingerprint(self, type_: TypeType, fp: Fingerprint) -> bool:
        """Check if a fingerprint exists at this level or any ancestor."""
        if self._local_fingerprints[type_].contains(fp):
            return True
        if self.parent is not None:
            return self.parent.contains_fingerprint(type_, fp)
        return False

    def local_count(self) -> int:
        """Count programs added at this context level only."""
        return sum(
            len(progs)
            for by_size in self._local.values()
            for progs in by_size.values()
        )

    def total_count(self) -> int:
        """Count all accessible programs (this level + ancestors)."""
        count = self.local_count()
        if self.parent is not None:
            if isinstance(self.parent, ContextualBank):
                count += self.parent.total_count()
            else:
                count += self.parent.count()
        return count


def integer_partitions(n: int, k: int) -> Iterator[tuple[int, ...]]:
    """
    Generate all ordered partitions of n into k parts, each >= 1.

    E.g., integer_partitions(4, 2) yields (1,3), (2,2), (3,1).
    """
    if k == 1:
        if n >= 1:
            yield (n,)
        return
    for first in range(1, n - k + 2):
        for rest in integer_partitions(n - first, k - 1):
            yield (first,) + rest


def _fresh_param_names(n_params: int, context: dict[str, TypeType]) -> list[str]:
    """Pick _p0, _p1, ... skipping names already in context."""
    names = []
    i = 0
    while len(names) < n_params:
        candidate = f"_p{i}"
        if candidate not in context:
            names.append(candidate)
        i += 1
    return names


class BottomUpEnumerator:
    """Bottom-up enumerator with observational equivalence pruning."""

    def __init__(
        self,
        grammar: Grammar = DefaultGrammar,
        test_suite: list[list[int]] | None = None,
        seed_constants: list[int] | None = None,
        max_size: int = 5,
        min_variability: float = 0.3,
        input_var_name: str = "x",
        input_type: TypeType = list[int],
        max_nesting: int = 2,
    ):
        self.grammar = grammar
        self.test_suite = test_suite if test_suite is not None else DEFAULT_TEST_SUITE
        self.seed_constants = seed_constants if seed_constants is not None else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.max_size = max_size
        self.min_variability = min_variability
        self.input_var_name = input_var_name
        self.input_type = input_type
        self.max_nesting = max_nesting

        self.bank = ProgramBank()
        self.jit = JITCompiler(grammar)
        self._child_bank_cache: dict = {}
        self._valid_instantiations = compute_valid_instantiations(grammar)

    def enumerate(self) -> ProgramBank:
        """Run bottom-up enumeration and return the populated program bank."""
        self._child_bank_cache = {}
        self._enumerate_base_case()
        print(f"Size 1: {self.bank.count()} total programs")
        for size in range(2, self.max_size + 1):
            self._enumerate_at_size(size)
            print(f"Size {size}: {self.bank.count()} total programs")
        return self.bank

    def _enumerate_base_case(self):
        """Populate the bank with all size-1 atoms."""
        # Integer hole (replaces concrete integer constant seeds)
        hole_node = IntHoleNode()
        self._try_add(hole_node, int, size=1)

        # Boolean constants
        for b in [True, False]:
            node = BooleanNode(b)
            self._try_add(node, bool, size=1)

        # Empty lists — one per list type in the universe
        for list_type in [list[int], list[bool], list[list[int]]]:
            node = ListNode([])
            self._try_add(node, list_type, size=1)

        # Input variable x : list[int]
        var_node = VariableNode(self.input_var_name)
        fp = self._compute_var_fingerprint()
        prog = TypedProgram(ast=var_node, type=self.input_type, fingerprint=fp, size=1)
        self.bank.add(prog)

    def _compute_var_fingerprint(self) -> Fingerprint:
        """Compute fingerprint for the input variable (identity on test suite)."""
        values = tuple(make_hashable(inp) for inp in self.test_suite)
        return Fingerprint(values)

    def _fingerprint(self, node: ASTNode, node_type: TypeType = None) -> Fingerprint | None:
        """
        Compute the fingerprint of a (possibly open) term.

        Wraps the term in (λ x <term>) and evaluates on the test suite.
        Returns None if compilation fails.
        """
        closed = LambdaNode([self.input_var_name], node)
        return compute_fingerprint(closed, self.test_suite, self.jit)

    def _fingerprint_in_context(
        self, node: ASTNode, context: dict[str, TypeType]
    ) -> Fingerprint | None:
        """
        Fingerprint a term that may reference any variables in context.

        For top-level context (only input var), delegates to _fingerprint().
        Otherwise wraps in nested lambdas and evaluates across
        test_suite × probe_value combinations.
        """
        inner_params = [
            (name, typ) for name, typ in context.items()
            if name != self.input_var_name
        ]

        if not inner_params:
            return self._fingerprint(node)

        # Build closed term: (λ x (λ _p0 (λ _p1 ... node)))
        closed = node
        for pname, _ in reversed(inner_params):
            closed = LambdaNode([pname], closed)
        closed = LambdaNode([self.input_var_name], closed)

        try:
            compiled, _ = self.jit.compile(closed)
        except Exception:
            return None

        # Compute probe value combinations for inner parameters
        inner_probes = [PROBE_VALUES.get(typ, [0]) for _, typ in inner_params]
        probe_combos = list(itertools.product(*inner_probes))

        values = []
        for inp in self.test_suite:
            for probes in probe_combos:
                try:
                    result = compiled(inp)
                    for p in probes:
                        result = result(p)
                    values.append(make_hashable(result))
                except Exception:
                    values.append(FAIL)

        return Fingerprint(tuple(values))

    def _try_add(self, node: ASTNode, resolved_type: TypeType, size: int) -> bool:
        """Compute fingerprint and add to bank if novel."""
        fp = self._fingerprint(node, resolved_type)
        if fp is None:
            return False
        prog = TypedProgram(ast=node, type=resolved_type, fingerprint=fp, size=size)
        return self.bank.add(prog)

    def _enumerate_at_size(self, size: int):
        """Enumerate all programs of exactly the given size."""
        for func_name in self.grammar.names:
            func_info = self.grammar[func_name]
            raw_arg_types = func_info['arg_types']
            raw_ret_type = func_info['ret_type']
            arity = len(raw_arg_types)

            for inst in self._valid_instantiations[func_name]:
                resolved_arg_types = self._resolve_types(raw_arg_types, inst)
                resolved_ret_type = resolve_type(raw_ret_type, instantiation=inst)

                if resolved_ret_type is None:
                    continue
                if any(t is None for t in resolved_arg_types):
                    continue

                # The function node costs 1, so arguments must sum to (size - 1)
                remaining = size - 1
                if remaining < arity:
                    continue  # Not enough size budget for all arguments

                # Detect higher-order arguments
                ho_indices = [
                    i for i, t in enumerate(resolved_arg_types)
                    if get_origin(t) == CallableOrig
                ]

                if ho_indices:
                    self._enumerate_higher_order(
                        func_name, resolved_arg_types, resolved_ret_type,
                        ho_indices, remaining, size,
                    )
                else:
                    self._enumerate_first_order(
                        func_name, resolved_arg_types, resolved_ret_type,
                        remaining, size,
                    )

    def _enumerate_first_order(
        self, func_name, arg_types, ret_type, remaining, total_size,
    ):
        """Enumerate applications of a first-order function."""
        arity = len(arg_types)

        for partition in integer_partitions(remaining, arity):
            arg_candidates = []
            for i, (arg_type, arg_size) in enumerate(zip(arg_types, partition)):
                candidates = self.bank.get(arg_type, arg_size)
                if not candidates:
                    break
                arg_candidates.append(candidates)
            else:
                for combo in itertools.product(*arg_candidates):
                    node = ApplicationNode(
                        VariableNode(func_name),
                        [c.ast for c in combo],
                    )
                    self._try_add(node, ret_type, total_size)

    def _enumerate_higher_order(
        self, func_name, arg_types, ret_type, ho_indices, remaining, total_size,
    ):
        """Enumerate applications of a higher-order function."""
        arity = len(arg_types)
        root_context = {self.input_var_name: self.input_type}

        for partition in integer_partitions(remaining, arity):
            arg_candidates = []
            skip = False

            for i, (arg_type, arg_size) in enumerate(zip(arg_types, partition)):
                if i in ho_indices:
                    lambdas = self._enumerate_lambda_arg(
                        self.bank, root_context, arg_type, arg_size,
                        nesting_depth=0,
                    )
                    if not lambdas:
                        skip = True
                        break
                    arg_candidates.append(lambdas)
                else:
                    candidates = self.bank.get(arg_type, arg_size)
                    if not candidates:
                        skip = True
                        break
                    arg_candidates.append(candidates)

            if skip:
                continue

            for combo in itertools.product(*arg_candidates):
                args = []
                for c in combo:
                    if isinstance(c, TypedProgram):
                        args.append(c.ast)
                    else:
                        # LambdaNode from _enumerate_lambda_arg
                        args.append(c)
                node = ApplicationNode(VariableNode(func_name), args)
                self._try_add(node, ret_type, total_size)

    def _enumerate_lambda_arg(
        self,
        parent_bank: 'ProgramBank | ContextualBank',
        parent_context: dict[str, TypeType],
        callable_type: TypeType,
        available_size: int,
        nesting_depth: int,
    ) -> list[ASTNode]:
        """
        Enumerate lambda expressions by recursively enumerating bodies
        in an extended context with a child bank.

        Returns list of LambdaNode ASTs.
        """
        args = get_args(callable_type)
        param_types = args[0]  # list of parameter types
        body_type = args[1]    # return type

        body_budget = available_size - 1
        if body_budget < 1:
            return []

        param_names = _fresh_param_names(len(param_types), parent_context)

        # Extend context with lambda parameters
        child_context = parent_context.copy()
        for pname, ptype in zip(param_names, param_types):
            child_context[pname] = ptype

        # Get or build the child bank
        child_bank = self._get_or_build_child_bank(
            parent_bank, child_context, body_budget, nesting_depth + 1,
        )

        # Collect body terms of the target type, wrap as lambdas
        results = []
        for body_size in range(1, body_budget + 1):
            for prog in child_bank.get(body_type, body_size):
                results.append(LambdaNode(param_names, prog.ast))

        return results

    def _get_or_build_child_bank(
        self,
        parent_bank: 'ProgramBank | ContextualBank',
        child_context: dict[str, TypeType],
        body_budget: int,
        nesting_depth: int,
    ) -> ContextualBank:
        """Return a cached child bank or build one via recursive enumeration."""
        key = (
            id(parent_bank),
            frozenset(child_context.items()),
            body_budget,
            nesting_depth,
        )

        if key not in self._child_bank_cache:
            child_bank = ContextualBank(parent=parent_bank)
            self._enumerate_in_child_context(
                child_bank, child_context, body_budget, nesting_depth,
            )
            self._child_bank_cache[key] = child_bank

        return self._child_bank_cache[key]

    def _enumerate_in_child_context(
        self,
        child_bank: ContextualBank,
        context: dict[str, TypeType],
        max_size: int,
        nesting_depth: int,
    ):
        """
        Recursive workhorse: enumerate programs in an extended context.

        Adds lambda params as size-1 atoms, then for sizes 2..max_size
        iterates grammar functions. Higher-order functions recurse if
        nesting_depth < max_nesting; otherwise they are skipped.
        """
        # Add lambda params as size-1 atoms (fingerprinted in context)
        for var_name, var_type in context.items():
            if var_name == self.input_var_name:
                continue  # already in parent bank
            node = VariableNode(var_name)
            fp = self._fingerprint_in_context(node, context)
            if fp is not None:
                prog = TypedProgram(ast=node, type=var_type, fingerprint=fp, size=1)
                child_bank.add_local(prog)

        # Enumerate applications at each size
        for size in range(2, max_size + 1):
            for func_name in self.grammar.names:
                func_info = self.grammar[func_name]
                raw_arg_types = func_info['arg_types']
                raw_ret_type = func_info['ret_type']
                arity = len(raw_arg_types)

                for inst in self._valid_instantiations[func_name]:
                    resolved_arg_types = self._resolve_types(raw_arg_types, inst)
                    resolved_ret_type = resolve_type(raw_ret_type, instantiation=inst)

                    if resolved_ret_type is None:
                        continue
                    if any(t is None for t in resolved_arg_types):
                        continue

                    remaining = size - 1
                    if remaining < arity:
                        continue

                    # Detect higher-order arguments
                    ho_indices = [
                        i for i, t in enumerate(resolved_arg_types)
                        if get_origin(t) == CallableOrig
                    ]

                    # Skip higher-order functions if at max nesting
                    # nesting_depth tracks current lambda depth (1 = first lambda body)
                    # max_nesting=2 means allow HOFs at depth 1 and 2 but not depth 3
                    if ho_indices and nesting_depth > self.max_nesting:
                        continue

                    for partition in integer_partitions(remaining, arity):
                        arg_candidates = []
                        skip = False

                        for i, (arg_type, arg_size) in enumerate(zip(resolved_arg_types, partition)):
                            if i in ho_indices:
                                lambdas = self._enumerate_lambda_arg(
                                    child_bank, context, arg_type, arg_size,
                                    nesting_depth,
                                )
                                if not lambdas:
                                    skip = True
                                    break
                                # Wrap lambdas as TypedProgram for uniform handling
                                arg_candidates.append(lambdas)
                            else:
                                candidates = child_bank.get(arg_type, arg_size)
                                if not candidates:
                                    skip = True
                                    break
                                arg_candidates.append(candidates)

                        if skip:
                            continue

                        for combo in itertools.product(*arg_candidates):
                            args_list = []
                            for c in combo:
                                if isinstance(c, TypedProgram):
                                    args_list.append(c.ast)
                                else:
                                    # LambdaNode from _enumerate_lambda_arg
                                    args_list.append(c)
                            body = ApplicationNode(
                                VariableNode(func_name),
                                args_list,
                            )
                            fp = self._fingerprint_in_context(body, context)
                            if fp is not None:
                                prog = TypedProgram(
                                    ast=body, type=resolved_ret_type,
                                    fingerprint=fp, size=size,
                                )
                                child_bank.add_local(prog)

    def _resolve_types(self, types: tuple, instantiation: dict | None = None) -> list[TypeType | None]:
        if instantiation is not None:
            return [resolve_type(t, instantiation=instantiation) for t in types]
        return [resolve_type(t) for t in types]

    def _types_match(self, type1: TypeType, type2: TypeType) -> bool:
        subs = SubstitutionTable()
        return matchable(type1, type2, subs, update=False)

    def extract_corpus(
        self,
        min_variability: float = 0.3,
        min_successes: int = 3,
    ) -> list[TypedProgram]:
        """Extract the final corpus of quality-filtered programs."""
        corpus = []
        for type_key, by_size in self.bank._bank.items():
            for size, progs in by_size.items():
                for prog in progs:
                    if prog.fingerprint is not None and passes_quality_filter(
                        prog.fingerprint,
                        min_successes=min_successes,
                        min_variability=min_variability,
                    ):
                        corpus.append(prog)
        return corpus
