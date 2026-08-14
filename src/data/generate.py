"""
Synthesise a program corpus for training Model A.

Phases:
  1. Bottom-up enumeration up to size 8  ->  ~644K quality sketches
  2. Sketch expansion (seed_constants substitution)  ->  ~1.79M concrete programs
  3. Warm-start policy (100K subsample, 5 epochs, MPS/CUDA if available)
     + RL collection (100K novel sketches, n_samples=4)
     + Post-RL expansion  ->  ~3.9M concrete programs

Total output: ~5.7M programs across enum_corpus.json and rl_corpus.json.

Checkpoints written to output_dir so interrupted runs resume from the last
completed phase:
  enum_corpus.pkl         -- skips Phase 1+2 on restart
  policy_warmstart.pt     -- skips warm-start on restart

Usage:
  python -m src.data.generate [--output-dir OUTPUT_DIR]
  generate-dataset [--output-dir OUTPUT_DIR]      (after ``pip install -e .``)
"""

import argparse
import os
import sys


def main() -> None:
    # The project's modules import each other as ``from src.foo...`` everywhere
    # (see e.g. src/lang/synthesis/pipeline.py). That works when invoked via
    # ``python -m src.data.generate`` from the project root (cwd → sys.path),
    # but it does NOT work when invoked as the ``generate-dataset`` console
    # script post ``pip install -e .``: the editable install puts ``src/`` on
    # sys.path, not the project root, so ``import src`` fails. Insert the
    # project root before doing the heavy imports so both entry paths work.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from src.lang.synthesis.pipeline import synthesise_corpus
    from src.lang.grammar import GRAMMARS, get_grammar

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        default="output/corpus-a",
        help="Directory for output files and checkpoints",
    )
    parser.add_argument(
        "--grammar",
        default="default",
        choices=sorted(GRAMMARS),
        help="Which grammar to synthesise the corpus from (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for Python random and torch RNG (default: unseeded)",
    )
    parser.add_argument(
        "--seed-constants",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="Integer constants substituted into sketch holes "
        "(default: 0 1 2 ... 10). Accepts negatives, e.g. -1 0 1 2.",
    )
    parser.add_argument(
        "--enum-max-size",
        type=int,
        default=8,
        help="Bottom-up enumeration program-size bound s_max (default: %(default)s)",
    )
    parser.add_argument(
        "--rl-expand-target",
        type=int,
        default=4_000_000,
        help="Target number of concrete programs from post-RL sketch expansion "
        "(default: %(default)s). Lower this on memory-constrained machines.",
    )
    args = parser.parse_args()

    synthesise_corpus(
        grammar=get_grammar(args.grammar),
        enum_max_size=args.enum_max_size,
        rl_expand_target=args.rl_expand_target,
        output_dir=args.output_dir,
        seed=args.seed,
        seed_constants=args.seed_constants,
    )


if __name__ == "__main__":
    main()
