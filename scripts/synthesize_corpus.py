"""
Synthesize a program corpus for training Model A.

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
  python scripts/synthesize_corpus.py [--output-dir OUTPUT_DIR]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lang.pipeline import synthesize_corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', default='output/corpus-a', help='Directory for output files and checkpoints')
    args = parser.parse_args()

    synthesize_corpus(
        enum_max_size=8,
        output_dir=args.output_dir,
    )
