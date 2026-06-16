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
    args = parser.parse_args()

    synthesise_corpus(
        grammar=get_grammar(args.grammar),
        enum_max_size=8,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
