#!/bin/bash
# ============================================================================
# Local Training Script (without Slurm)
# ============================================================================
# Mirrors scripts/train.slurm.sh but for a non-Slurm host (rented GPU box,
# workstation, etc.). Activates conda, prints GPU info, then launches
# src.train. Output is teed into logs/train_<timestamp>.log.
#
# Usage:
#   ./scripts/train.sh                                  # default mode=in-weight
#   MODE=symbol-shuffling ./scripts/train.sh
#   DATA_ROOT=/path/to/data CKPT_DIR=/path/to/ckpts SEED=7 MODE=symbol-shuffling ./scripts/train.sh
# ============================================================================

cd "$(dirname "$0")/.."

eval "$(micromamba shell hook --shell bash)"
micromamba activate ml13

DATA_ROOT="${DATA_ROOT:-$HOME/yiding-in-georgia/datasets}"
ENUM_CORPUS="${DATA_ROOT}/corpus-a/enum_corpus_no_rule.json"
RL_CORPUS="${DATA_ROOT}/corpus-a/rl_corpus_no_rule.simplified.json"
VAL_CORPUS="${DATA_ROOT}/rule_val.json"
CKPT_ROOT="${CKPT_DIR:-$HOME/mlmp_checkpoints}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-500}"
SEED="${SEED:-42}"
MODE="${MODE:-in-weight}"

# Keep checkpoints from different training modes in separate subdirectories so
# in-weight and symbol-shuffling runs don't overwrite each other's best/latest.
CKPT_DIR="${CKPT_ROOT}/${MODE}"

# If RUN_NAME is provided, pin checkpoints to ${CKPT_DIR}/${RUN_NAME}/ and
# auto-resume from checkpoint_latest.pt there if one exists. If RUN_NAME is
# unset, fall back to the previous behaviour: wandb assigns a fresh name and
# the run starts from scratch.
RUN_NAME_FLAG=()
RESUME_FLAG=()
if [ -n "${RUN_NAME:-}" ]; then
    RUN_NAME_FLAG=(--run-name "${RUN_NAME}")
    LATEST_CKPT="${CKPT_DIR}/${RUN_NAME}/checkpoint_latest.pt"
    if [ -f "${LATEST_CKPT}" ]; then
        echo "Auto-resuming from ${LATEST_CKPT}"
        RESUME_FLAG=(--resume "${LATEST_CKPT}")
    fi
fi

python -m src.train \
    --train-corpus "${ENUM_CORPUS},${RL_CORPUS}" \
    --val-corpus "${VAL_CORPUS}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --val-examples 256 \
    --epochs $EPOCHS \
    --mode "${MODE}" \
    --seed $SEED \
    --num-workers $NUM_WORKERS \
    "${RUN_NAME_FLAG[@]}" \
    "${RESUME_FLAG[@]}"
