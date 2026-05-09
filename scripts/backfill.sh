#!/bin/bash
# ============================================================================
# Backfill val_accuracy across all per-epoch checkpoints for a single run.
# Mirrors scripts/train.sh's env-var conventions so the checkpoint dir is
# computed identically.
#
# Usage:
#   RUN_NAME=<run> ./scripts/backfill_val_accuracy.sh
#   RUN_NAME=<run> MODE=symbol-shuffling ./scripts/backfill_val_accuracy.sh
#   RUN_NAME=<run> DATA_ROOT=/path/to/data CKPT_DIR=/path/to/ckpts ./scripts/backfill_val_accuracy.sh
#
# Forwards extra flags through to backfill_val_accuracy.py:
#   RUN_NAME=<run> ./scripts/backfill_val_accuracy.sh --dry-run
#   RUN_NAME=<run> ./scripts/backfill_val_accuracy.sh --val-examples 512
# ============================================================================

set -e

cd "$(dirname "$0")/.."

eval "$(micromamba shell hook --shell bash)"
micromamba activate ml13

if [ -z "${RUN_NAME:-}" ]; then
    echo "Error: RUN_NAME is required." >&2
    echo "Usage: RUN_NAME=<run> [MODE=...] $0 [extra flags forwarded to python]" >&2
    exit 1
fi

DATA_ROOT="${DATA_ROOT:-$HOME/yiding-in-georgia/datasets}"
CKPT_ROOT="${CKPT_DIR:-$HOME/yiding-in-georgia/mlmp_checkpoints}"
MODE="${MODE:-in-weight}"

# Identical to train.sh: <CKPT_ROOT>/<MODE>/<RUN_NAME>/
RUN_CKPT_DIR="${CKPT_ROOT}/${MODE}/${RUN_NAME}"

if [ ! -d "${RUN_CKPT_DIR}" ]; then
    echo "Error: checkpoint dir does not exist: ${RUN_CKPT_DIR}" >&2
    exit 1
fi

echo "Backfilling val_accuracy for ${RUN_CKPT_DIR}"
python -m scripts.backfill_val_accuracy \
    --checkpoint-dir "${RUN_CKPT_DIR}" \
    "$@"
