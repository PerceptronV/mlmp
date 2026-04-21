"""
Test robustness of corpus-a programs under perturbed grammars.

For Model B training we want to reuse corpus-a programs as queries. This
script checks what fraction of corpus-a programs still satisfy the corpus
quality filter (non-crashing, non-constant, variability >= 0.3) when the
grammar's primitive semantics are resampled from variation_templates.py.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.lang.ast_nodes import LambdaNode
from src.lang.compiler import JITCompiler
from src.lang.enumeration.filters import (
    is_non_constant,
    is_non_crashing,
    passes_quality_filter,
    variability,
)
from src.lang.enumeration.fingerprint import compute_fingerprint
from src.lang.enumeration.test_suite import DEFAULT_TEST_SUITE
from src.lang.grammar import DefaultGrammar
from src.lang.parser import parse
from src.lang.variation_templates import TemplateSemanticGrammar


def load_corpus(path: Path, n_programs: int | None, seed: int) -> list[str]:
    with open(path) as f:
        data = json.load(f)
    programs = [entry["program"] for entry in data]
    if n_programs is not None and n_programs < len(programs):
        rng = random.Random(seed)
        programs = rng.sample(programs, n_programs)
    return programs


def parse_all(program_strs: list[str]) -> list:
    asts, parse_fails = [], 0
    for s in program_strs:
        try:
            asts.append(parse(s))
        except Exception:
            asts.append(None)
            parse_fails += 1
    return asts, parse_fails


def _fast_variance(values) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / (n - 1)


def _generate_rule_candidates(rng, num_candidates, min_len, max_len, min_val, max_val):
    return [
        [rng.randint(min_val, max_val)
         for _ in range(rng.randint(min_len, max_len))]
        for _ in range(num_candidates)
    ]


INT_MOD = 100  # project convention: integer values wrap mod 100


def _compute_io_pairs(compiled_fn, candidates, max_len):
    pairs = []
    for inp in candidates:
        try:
            out = compiled_fn(inp)
        except Exception:
            continue
        if type(out) is not list:
            continue
        if not all(type(x) is int for x in out):
            continue
        if len(out) > max_len:
            continue
        out = [x % INT_MOD for x in out]
        pairs.append((inp, out))
    return pairs


def _select_best_io_pairs(all_pairs, n, rng, max_len):
    """Stratified selection by input length (mirrors RuleSampler)."""
    if len(all_pairs) <= n:
        return list(all_pairs)

    bucket_size = (max_len + 1) // 4
    num_buckets = 4
    buckets = [[] for _ in range(num_buckets)]
    for p in all_pairs:
        b = min(len(p[0]) // bucket_size, num_buckets - 1)
        buckets[b].append(p)

    seen_outputs = set()
    selected = []
    base, rem = divmod(n, num_buckets)

    def priority(pair):
        inp, out = pair
        return (
            int(tuple(out) not in seen_outputs),
            int(inp != out),
            rng.random(),
        )

    for bi in range(num_buckets):
        target = base + (1 if bi < rem else 0)
        for p in sorted(buckets[bi], key=priority, reverse=True):
            if len(selected) >= n or target <= 0:
                break
            selected.append(p)
            seen_outputs.add(tuple(p[1]))
            target -= 1

    if len(selected) < n:
        selected_ids = {id(p) for p in selected}
        rest = [p for p in all_pairs if id(p) not in selected_ids]
        for p in sorted(rest, key=priority, reverse=True):
            if len(selected) >= n:
                break
            selected.append(p)
            seen_outputs.add(tuple(p[1]))
    return selected


def _score_io_set(pairs, max_len_var, max_elem_var):
    """Rule quality score (length var, element var, unique outputs, non-identity)."""
    if len(pairs) < 2:
        return float("-inf")
    in_lens = [len(inp) for inp, _ in pairs]
    out_lens = [len(out) for _, out in pairs]
    all_in = [x for inp, _ in pairs for x in inp]
    all_out = [x for _, out in pairs for x in out]
    length_var = _fast_variance(in_lens) + _fast_variance(out_lens)
    element_var = _fast_variance(all_in) + _fast_variance(all_out)
    norm_elem = element_var / (max_elem_var + 1)
    unique = len({tuple(out) for _, out in pairs}) / len(pairs)
    non_id = 1.0 - sum(1 for inp, out in pairs if inp == out) / len(pairs)
    return (
        0.25 * (length_var / (max_len_var + 1))
        + 0.25 * norm_elem
        + 0.25 * unique
        + 0.25 * non_id
    )


def evaluate_under_grammar_rule_style(
    asts,
    grammar,
    rng,
    num_io: int,
    num_candidates: int,
    min_len: int,
    max_len: int,
    min_val: int,
    max_val: int,
    min_quality: float,
):
    jit = JITCompiler(grammar)
    max_len_var = (max_len - min_len) ** 2 / 4
    max_elem_var = (max_val - min_val) ** 2 / 4

    n = len(asts)
    pass_flags = [False] * n
    insufficient = 0
    low_score = 0
    compile_fails = 0
    scores = []

    for i, ast in enumerate(asts):
        if ast is None:
            continue
        compile_node = ast if isinstance(ast, LambdaNode) else LambdaNode(["x"], ast)
        try:
            compiled_fn, _ = jit.compile(compile_node)
        except Exception:
            compile_fails += 1
            continue

        candidates = _generate_rule_candidates(
            rng, num_candidates, min_len, max_len, min_val, max_val
        )
        pairs = _compute_io_pairs(compiled_fn, candidates, max_len)
        if len(pairs) < num_io:
            insufficient += 1
            continue
        selected = _select_best_io_pairs(pairs, num_io, rng, max_len)
        score = _score_io_set(selected, max_len_var, max_elem_var)
        scores.append(score)
        if score >= min_quality:
            pass_flags[i] = True
        else:
            low_score += 1

    return {
        "pass_flags": pass_flags,
        "n_pass": sum(pass_flags),
        "insufficient_pairs": insufficient,
        "low_score": low_score,
        "compile_fails": compile_fails,
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
    }


def evaluate_under_grammar(asts, grammar, min_variability: float, min_successes: int):
    jit = JITCompiler(grammar)
    n = len(asts)
    pass_flags = [False] * n
    non_crashing = 0
    non_constant = 0
    var_ok = 0
    compile_fails = 0
    var_values = []

    for i, ast in enumerate(asts):
        if ast is None:
            continue
        try:
            fp = compute_fingerprint(ast, DEFAULT_TEST_SUITE, jit)
        except Exception:
            fp = None
        if fp is None:
            compile_fails += 1
            continue

        nc = is_non_crashing(fp, min_successes)
        ncc = is_non_constant(fp)
        v = variability(fp)
        if nc:
            non_crashing += 1
        if ncc:
            non_constant += 1
        if v >= min_variability:
            var_ok += 1
        var_values.append(v)
        if nc and ncc and v >= min_variability:
            pass_flags[i] = True

    return {
        "pass_flags": pass_flags,
        "n_pass": sum(pass_flags),
        "non_crashing": non_crashing,
        "non_constant": non_constant,
        "var_ok": var_ok,
        "compile_fails": compile_fails,
        "mean_variability": sum(var_values) / len(var_values) if var_values else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus",
        default="output/corpus-a/rl_corpus.json",
        help="Path to corpus JSON (list of {program, type, size}).",
    )
    ap.add_argument("--n-programs", type=int, default=2000)
    ap.add_argument("--n-grammars", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--canonical-prob", type=float, default=0.0,
                    help="Probability of sampling canonical variant per function.")
    ap.add_argument("--min-variability", type=float, default=0.3)
    ap.add_argument("--min-successes", type=int, default=3)
    ap.add_argument("--rule-style", action="store_true",
                    help="Use Rule-style per-program I/O selection instead of DEFAULT_TEST_SUITE.")
    ap.add_argument("--num-io-pairs", type=int, default=11)
    ap.add_argument("--num-candidate-inputs", type=int, default=100)
    ap.add_argument("--min-list-length", type=int, default=0)
    ap.add_argument("--max-list-length", type=int, default=15)
    ap.add_argument("--min-element-value", type=int, default=0)
    ap.add_argument("--max-element-value", type=int, default=100)
    ap.add_argument("--min-quality-score", type=float, default=0.7)
    ap.add_argument("--out", default=None, help="Optional JSON path for detailed results.")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    print(f"Loading corpus: {corpus_path}")
    t0 = time.time()
    programs = load_corpus(corpus_path, args.n_programs, args.seed)
    print(f"  Loaded {len(programs)} programs in {time.time() - t0:.1f}s")

    print("Parsing programs...")
    t0 = time.time()
    asts, parse_fails = parse_all(programs)
    print(f"  Parsed {len(asts) - parse_fails}/{len(asts)} "
          f"(parse fails: {parse_fails}) in {time.time() - t0:.1f}s")

    n = len(asts)

    def _eval(grammar, eval_rng):
        if args.rule_style:
            return evaluate_under_grammar_rule_style(
                asts, grammar, eval_rng,
                num_io=args.num_io_pairs,
                num_candidates=args.num_candidate_inputs,
                min_len=args.min_list_length,
                max_len=args.max_list_length,
                min_val=args.min_element_value,
                max_val=args.max_element_value,
                min_quality=args.min_quality_score,
            )
        return evaluate_under_grammar(
            asts, grammar, args.min_variability, args.min_successes
        )

    mode = "rule-style" if args.rule_style else "fixed test suite"
    print(f"\nBaseline (canonical grammar, {mode}):")
    t0 = time.time()
    baseline_rng = random.Random(args.seed + 10_000)
    baseline = _eval(DefaultGrammar, baseline_rng)
    if args.rule_style:
        print(f"  pass {baseline['n_pass']}/{n} "
              f"({100 * baseline['n_pass'] / n:.1f}%), "
              f"insufficient={baseline['insufficient_pairs']}, "
              f"low_score={baseline['low_score']}, "
              f"compile_fails={baseline['compile_fails']}, "
              f"mean_score={baseline['mean_score']:.3f}  "
              f"[{time.time() - t0:.1f}s]")
    else:
        print(f"  pass {baseline['n_pass']}/{n} "
              f"({100 * baseline['n_pass'] / n:.1f}%), "
              f"compile_fails={baseline['compile_fails']}, "
              f"mean_var={baseline['mean_variability']:.3f}  "
              f"[{time.time() - t0:.1f}s]")

    # Perturbed grammars
    meta_rng = random.Random(args.seed)
    grammar_results = []
    survive_counts = [0] * n

    print(f"\nEvaluating under {args.n_grammars} perturbed grammars:")
    for g_idx in range(args.n_grammars):
        t0 = time.time()
        grammar_seed = meta_rng.randint(0, 2**31 - 1)
        grng = random.Random(grammar_seed)
        grammar = TemplateSemanticGrammar.sample(
            DefaultGrammar, grng, canonical_prob=args.canonical_prob
        )
        eval_rng = random.Random(grammar_seed ^ 0xA5A5)
        result = _eval(grammar, eval_rng)
        for i, p in enumerate(result["pass_flags"]):
            if p:
                survive_counts[i] += 1
        elapsed = time.time() - t0
        pct = 100 * result["n_pass"] / n
        if args.rule_style:
            print(f"  [{g_idx+1:03d}/{args.n_grammars}] seed={grammar_seed} "
                  f"pass={result['n_pass']}/{n} ({pct:.1f}%) "
                  f"insuf={result['insufficient_pairs']} "
                  f"low={result['low_score']} "
                  f"comp_fail={result['compile_fails']} "
                  f"mean_score={result['mean_score']:.3f}  [{elapsed:.1f}s]")
            grammar_results.append({
                "grammar_seed": grammar_seed,
                "n_pass": result["n_pass"],
                "insufficient_pairs": result["insufficient_pairs"],
                "low_score": result["low_score"],
                "compile_fails": result["compile_fails"],
                "mean_score": result["mean_score"],
            })
        else:
            print(f"  [{g_idx+1:03d}/{args.n_grammars}] seed={grammar_seed} "
                  f"pass={result['n_pass']}/{n} ({pct:.1f}%) "
                  f"crash_ok={result['non_crashing']} nonconst={result['non_constant']} "
                  f"var_ok={result['var_ok']} "
                  f"comp_fail={result['compile_fails']} "
                  f"mean_var={result['mean_variability']:.3f}  [{elapsed:.1f}s]")
            grammar_results.append({
                "grammar_seed": grammar_seed,
                "n_pass": result["n_pass"],
                "non_crashing": result["non_crashing"],
                "non_constant": result["non_constant"],
                "var_ok": result["var_ok"],
                "compile_fails": result["compile_fails"],
                "mean_variability": result["mean_variability"],
            })

    # Aggregate summary
    pass_rates = [r["n_pass"] / n for r in grammar_results]
    mean_pass = sum(pass_rates) / len(pass_rates) if pass_rates else 0.0
    min_pass = min(pass_rates) if pass_rates else 0.0
    max_pass = max(pass_rates) if pass_rates else 0.0

    survive_fracs = [c / args.n_grammars for c in survive_counts]
    survive_all = sum(1 for c in survive_counts if c == args.n_grammars)
    survive_any = sum(1 for c in survive_counts if c > 0)
    mean_survive = sum(survive_fracs) / n if n else 0.0

    print("\n=== Summary ===")
    print(f"Per-grammar pass rate: mean={mean_pass:.3f} "
          f"min={min_pass:.3f} max={max_pass:.3f}")
    print(f"Per-program survival (frac grammars passed):")
    print(f"  mean={mean_survive:.3f}")
    print(f"  programs passing ALL {args.n_grammars} grammars: "
          f"{survive_all}/{n} ({100 * survive_all / n:.1f}%)")
    print(f"  programs passing ANY grammar: "
          f"{survive_any}/{n} ({100 * survive_any / n:.1f}%)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "args": vars(args),
                "n_programs": n,
                "parse_fails": parse_fails,
                "baseline": {
                    k: v for k, v in baseline.items() if k != "pass_flags"
                },
                "baseline_pass_flags": baseline["pass_flags"],
                "grammar_results": grammar_results,
                "survive_counts": survive_counts,
                "summary": {
                    "mean_pass_rate": mean_pass,
                    "min_pass_rate": min_pass,
                    "max_pass_rate": max_pass,
                    "mean_survive_frac": mean_survive,
                    "survive_all": survive_all,
                    "survive_any": survive_any,
                },
            }, f, indent=2)
        print(f"Wrote detailed results to {out_path}")


if __name__ == "__main__":
    main()
