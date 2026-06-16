#!/bin/bash
# ============================================================================
# wandb sweep trial wrapper
# ============================================================================
# Invoked once per trial by `wandb agent`. The sweep injects the searched
# hyperparameters as `--key=value` args (see sweeps/*.yaml); this wrapper adds
# the fixed run config from the environment and launches `python -m src.train`.
#
# Two pieces of translation happen here:
#   * --lr-schedule={cosine,constant}  ->  presence/absence of the --constant-lr
#     store_true flag, which argparse can't receive as --constant-lr=value.
#   * --mode=<m> is read to nest checkpoints under ${CKPT_ROOT}/<m>/ (and still
#     passed through to src.train).
#
# Required env (exported by scripts/sweep_agent.slurm.sh):
#   ENUM_CORPUS, RL_CORPUS, VAL_CORPUS, CKPT_ROOT
# Optional env: NUM_WORKERS (8), WANDB_PROJECT (mlmp), GRAMMAR (default)
#   GRAMMAR must match the grammar the corpora were synthesised from; it drives
#   the tokeniser vocab and (for *-symbol-shuffling modes) the fn-name permutation.
#
# src/train.py sets args.run_name = wandb.run.name when --run-name is unset, and
# nests checkpoints under checkpoint_dir/run_name, so concurrent trials never
# collide despite sharing CKPT_ROOT/<mode>.
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

: "${ENUM_CORPUS:?export ENUM_CORPUS}"
: "${RL_CORPUS:?export RL_CORPUS}"
: "${VAL_CORPUS:?export VAL_CORPUS}"
: "${CKPT_ROOT:?export CKPT_ROOT}"

MODE="in-weight"
SCHED_FLAG=()
EXTRA=()
for arg in "$@"; do
  case "$arg" in
    --lr-schedule=constant) SCHED_FLAG=(--constant-lr) ;;
    --lr-schedule=cosine)   : ;;  # cosine is the default schedule; emit no flag
    --mode=*) MODE="${arg#--mode=}"; EXTRA+=("$arg") ;;
    *) EXTRA+=("$arg") ;;
  esac
done

python -m src.train \
    --train-corpus "${ENUM_CORPUS},${RL_CORPUS}" \
    --val-corpus "${VAL_CORPUS}" \
    --checkpoint-dir "${CKPT_ROOT}/${MODE}" \
    --grammar "${GRAMMAR:-default}" \
    --val-examples 256 \
    --num-workers "${NUM_WORKERS:-8}" \
    --wandb-project "${WANDB_PROJECT:-mlmp}" \
    "${SCHED_FLAG[@]}" \
    "${EXTRA[@]}"
