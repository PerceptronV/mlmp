"""End-to-end smoke test for the training script with a tiny Transformer.

Runs one very small epoch on the query_first_template_seed42 train/validation
splits and computes functional validation accuracy.
"""

import os
import sys
from pathlib import Path

from src.train import train as train_main


def run():
    # Ensure we don't accidentally talk to a real W&B project
    os.environ.setdefault("WANDB_MODE", "disabled")

    repo_root = Path(__file__).parent

    # Preserve original argv so this can be imported safely
    orig_argv = sys.argv[:]
    try:
        sys.argv = [
            "train",
            "--train-dir",
            str(repo_root / "datasets/query_first_template_seed42/train"),
            "--val-dir",
            str(repo_root / "datasets/query_first_template_seed42/validation"),
            "--d-embed",
            "16",
            "--d-model",
            "16",
            "--n-heads",
            "2",
            "--n-layers",
            "1",
            "--dropout",
            "0.0",
            "--batch-size",
            "4",
            "--epochs",
            "1",
            "--steps-per-epoch",
            "2",
            "--val-examples",
            "10",
            "--no-wandb",
            "--device",
            "cpu",
            "--checkpoint-dir",
            str(repo_root / "checkpoints_test"),
            "--run-name",
            "train_small_smoketest",
        ]
        train_main()
    finally:
        sys.argv = orig_argv


if __name__ == "__main__":
    run()

