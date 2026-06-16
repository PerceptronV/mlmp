#!/usr/bin/env python
"""Print the best hyperparameters from one or more wandb sweeps.

The sweeps optimise `best_val_accuracy` (maximize), declared in sweeps/*.yaml,
so wandb's own `sweep.best_run()` returns the winning trial directly.

Usage:
    python scripts/sweep_best.py <sweep_id> [<sweep_id> ...]

A <sweep_id> may be a bare id (e.g. li7is16c) or a full path
(yiding-song-vincent/mlmp/li7is16c). Bare ids use the entity/project defaults
below.

Examples:
    python scripts/sweep_best.py abc123 def456
    python scripts/sweep_best.py yiding-song-vincent/mlmp/abc123
"""
import sys

import wandb

DEFAULT_ENTITY = "yiding-song-vincent"
DEFAULT_PROJECT = "mlmp"

# Only the knobs this sweep actually searched (argparse dests, underscored).
# lr-schedule is consumed by the wrapper and surfaces as the constant_lr flag.
SWEPT_KEYS = ["lr", "weight_decay", "grad_clip", "batch_size", "constant_lr"]


def resolve(sweep_id: str) -> str:
    """Accept a bare id or a full entity/project/id path."""
    if sweep_id.count("/") == 2:
        return sweep_id
    return f"{DEFAULT_ENTITY}/{DEFAULT_PROJECT}/{sweep_id}"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1

    api = wandb.Api()
    for raw in argv:
        path = resolve(raw)
        sweep = api.sweep(path)
        best = sweep.best_run()  # ranked by the sweep's declared metric

        mode = best.config.get("mode", "?")
        acc = best.summary.get("best_val_accuracy")
        acc_str = f"{acc:.4f}" if isinstance(acc, (int, float)) else str(acc)

        print(f"\n=== sweep {path}  (mode={mode}) ===")
        print(f"best run     : {best.name}  ({best.url})")
        print(f"best_val_accuracy: {acc_str}")
        print("hyperparameters:")
        for key in SWEPT_KEYS:
            if key in best.config:
                print(f"    {key:<14} = {best.config[key]}")
        # lr-schedule readout derived from the constant_lr flag.
        if "constant_lr" in best.config:
            sched = "constant" if best.config["constant_lr"] else "cosine"
            print(f"    {'lr-schedule':<14} = {sched}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
