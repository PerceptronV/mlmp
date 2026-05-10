#!/bin/bash
# ============================================================================
# Inverse-MLC Training Script (without Slurm)
# ============================================================================
# Variant of scripts/train.sh that trains the seq2seq transformer on the
# inverse-MLC algebraic dataset bundled at src/data/inverse_mlc/data_algebraic
# (no external corpus, no --mode flag — episode type is selected via
# EPISODE_TYPE; defaults to "algebraic+biases" matching the full BIML setup).
#
# Usage:
#   ./scripts/inverse-mlc-train.sh                            # default episode_type
#   EPISODE_TYPE=algebraic ./scripts/inverse-mlc-train.sh     # algebraic only
#   CKPT_DIR=/path/to/ckpts SEED=7 ./scripts/inverse-mlc-train.sh
#   RUN_NAME=mlc-alg ./scripts/inverse-mlc-train.sh            # auto-resume if checkpoint_latest.pt exists
# ============================================================================

cd "$(dirname "$0")/.."

CKPT_ROOT="${CKPT_DIR:-$HOME/mlmp_checkpoints}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"
EPISODE_TYPE="${EPISODE_TYPE:-algebraic+biases}"

# Keep checkpoints from different inverse-MLC episode types in separate
# subdirectories so e.g. algebraic and algebraic+biases runs don't overwrite
# each other's best/latest.
CKPT_DIR="${CKPT_ROOT}/inverse-mlc/${EPISODE_TYPE}"

# If RUN_NAME is provided, pin checkpoints to ${CKPT_DIR}/${RUN_NAME}/ and
# auto-resume from checkpoint_latest.pt there if one exists. If RUN_NAME is
# unset, fall back to wandb-assigned name and start from scratch (mirrors
# the behaviour in train.sh).
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
    --dataset inverse-mlc \
    --inverse-mlc-episode-type "${EPISODE_TYPE}" \
    --max-seq-len 2048 \
    --batch-size 512 \
    --epochs 20 \
    --steps-per-epoch 1000 \
    --checkpoint-dir "${CKPT_DIR}" \
    --seed $SEED \
    --num-workers $NUM_WORKERS \
    "${RUN_NAME_FLAG[@]}" \
    "${RESUME_FLAG[@]}"
