"""Core bottom-up enumerator with observational equivalence pruning."""

import itertools
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

from ..lang.grammar import Grammar, DefaultGrammar, T1, T2
from ..lang.ast_nodes import (
    ASTNode, NumberNode, BooleanNode, VariableNode,
    LambdaNode, ApplicationNode, ListNode,
)
from ..lang.compiler import JITCompiler
from ..lang.type_utils import (
    CallableOrig, get_args, get_origin,
    SubstitutionTable, substitute_type_vars, matchable, TypeType,
)
from ..utils import program_size, resolve_type
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

    def count(self) -> int:
        """Total number of stored programs."""
        return sum(
            len(progs)
            for by_size in self._bank.values()
            for progs in by_size.values()
        )


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
    ):
        self.grammar = grammar
        self.test_suite = test_suite if test_suite is not None else DEFAULT_TEST_SUITE
        self.seed_constants = seed_constants if seed_constants is not None else [0, 1, 2, 3]
        self.max_size = max_size
        self.min_variability = min_variability
        self.input_var_name = input_var_name
        self.input_type = input_type

        self.bank = ProgramBank()
        self.jit = JITCompiler(grammar)

    def enumerate(self) -> ProgramBank:
        """Run bottom-up enumeration and return the populated program bank."""
        self._enumerate_base_case()
        print(f"Size 1: {self.bank.count()} total programs")
        for size in range(2, self.max_size + 1):
            self._enumerate_at_size(size)
            print(f"Size {size}: {self.bank.count()} total programs")
        return self.bank

    def _enumerate_base_case(self):
        """Populate the bank with all size-1 atoms."""
        # Integer constants
        for c in self.seed_constants:
            node = NumberNode(c)
            self._try_add(node, int, size=1)

        # Boolean constants
        for b in [True, False]:
            node = BooleanNode(b)
            self._try_add(node, bool, size=1)

        # Empty list
        node = ListNode([])
        self._try_add(node, list[int], size=1)

        # Input variable x : list[int]
        var_node = VariableNode(self.input_var_name)
        fp = self._compute_var_fingerprint()
        prog = TypedProgram(ast=var_node, type=self.input_type, fingerprint=fp, size=1)
        self.bank.add(prog)

    def _compute_var_fingerprint(self) -> Fingerprint:
        """Compute fingerprint for the input variable (identity on test suite)."""
        values = tuple(make_hashable(inp) for inp in self.test_suite)
        return Fingerprint(values)

    def _fingerprint(self, node: ASTNode, node_type: TypeType) -> Fingerprint | None:
        """
        Compute the fingerprint of a (possibly open) term.

        Wraps the term in (λ x <term>) and evaluates on the test suite.
        Returns None if compilation fails.
        """
        closed = LambdaNode([self.input_var_name], node)
        return compute_fingerprint(closed, self.test_suite, self.jit)

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

            resolved_arg_types = self._resolve_types(raw_arg_types)
            resolved_ret_type = resolve_type(raw_ret_type)

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

        for partition in integer_partitions(remaining, arity):
            arg_candidates = []
            skip = False

            for i, (arg_type, arg_size) in enumerate(zip(arg_types, partition)):
                if i in ho_indices:
                    lambdas = self._enumerate_lambdas(arg_type, arg_size)
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
                        # LambdaNode from _enumerate_lambdas
                        args.append(c)
                node = ApplicationNode(VariableNode(func_name), args)
                self._try_add(node, ret_type, total_size)

    def _enumerate_lambdas(self, callable_type, available_size) -> list[ASTNode]:
        """
        Enumerate all lambda expressions of a given callable type and size.

        A lambda (λ params body) costs 1 + size(body), so the body budget is
        available_size - 1.

        Returns list of LambdaNode ASTs.
        """
        args = get_args(callable_type)
        param_types = args[0]  # list of parameter types
        body_type = args[1]    # return type

        body_budget = available_size - 1
        if body_budget < 1:
            return []

        param_names = [f"_p{i}" for i in range(len(param_types))]
        results = []

        # Size 1 bodies: lambda parameters or outer-bank size-1 terms
        if body_budget >= 1:
            for pname, ptype in zip(param_names, param_types):
                if self._types_match(ptype, body_type):
                    body = VariableNode(pname)
                    results.append(LambdaNode(param_names, body))

            for prog in self.bank.get(body_type, 1):
                results.append(LambdaNode(param_names, prog.ast))

        # Size 2+ bodies: apply first-order functions to params and outer terms
        for body_size in range(2, body_budget + 1):
            for func_name in self.grammar.names:
                func_info = self.grammar[func_name]
                f_arg_types = self._resolve_types(func_info['arg_types'])
                f_ret_type = resolve_type(func_info['ret_type'])

                if f_ret_type is None or f_ret_type != body_type:
                    continue
                if any(t is None for t in f_arg_types):
                    continue

                f_arity = len(f_arg_types)
                f_remaining = body_size - 1

                if f_remaining < f_arity:
                    continue

                # Skip higher-order functions inside lambda bodies for v1
                # TODO: Allow nested map/filter/fold for richer lambda expressions
                ho = any(get_origin(t) == CallableOrig for t in f_arg_types)
                if ho:
                    continue

                for partition in integer_partitions(f_remaining, f_arity):
                    arg_candidates = []
                    skip = False
                    for j, (at, s) in enumerate(zip(f_arg_types, partition)):
                        cands = list(self.bank.get(at, s))
                        # Add lambda params if they match and size is 1
                        if s == 1:
                            for pname, ptype in zip(param_names, param_types):
                                if self._types_match(ptype, at):
                                    cands.append(TypedProgram(
                                        ast=VariableNode(pname),
                                        type=ptype,
                                        fingerprint=None,
                                        size=1,
                                    ))
                        if not cands:
                            skip = True
                            break
                        arg_candidates.append(cands)

                    if skip:
                        continue

                    for combo in itertools.product(*arg_candidates):
                        body = ApplicationNode(
                            VariableNode(func_name),
                            [c.ast for c in combo],
                        )
                        results.append(LambdaNode(param_names, body))

        return results

    def _resolve_types(self, types: tuple) -> list[TypeType | None]:
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
