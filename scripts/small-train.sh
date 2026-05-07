#!/bin/bash
# ============================================================================
# Local Training Script — enum-only corpus (no RL), grokking-style setup
# ============================================================================
# Mirrors scripts/train.sh but for the small-data grokking experiment:
#   - Trains on the enum corpus only (drops the RL corpus)
#   - Caps the corpus to MAX_TRAIN_PROGRAMS randomly-sampled programs
#   - Cranks weight decay up from 0.01 → 0.1
#   - Disables the cosine LR schedule (constant LR)
# Everything else (mode, batch size, steps-per-epoch, model size) is unchanged.
#
# Usage:
#   ./scripts/no-rl-train.sh                                  # default mode=in-weight, 100k programs
#   MODE=symbol-shuffling ./scripts/no-rl-train.sh
#   MAX_TRAIN_PROGRAMS=50000 ./scripts/no-rl-train.sh
#   WEIGHT_DECAY=0.3 MAX_TRAIN_PROGRAMS=20000 ./scripts/no-rl-train.sh
#   DATA_ROOT=/path/to/data CKPT_DIR=/path/to/ckpts SEED=7 MODE=symbol-shuffling ./scripts/no-rl-train.sh
# ============================================================================

cd "$(dirname "$0")/.."

eval "$(micromamba shell hook --shell bash)"
micromamba activate ml13

DATA_ROOT="${DATA_ROOT:-$HOME/yiding-in-georgia/datasets}"
ENUM_CORPUS="${DATA_ROOT}/corpus-a/enum_corpus_no_rule.json"
VAL_CORPUS="${DATA_ROOT}/rule_val.json"
CKPT_ROOT="${CKPT_DIR:-$HOME/yiding-in-georgia/mlmp_checkpoints}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS="${EPOCHS:-500}"
SEED="${SEED:-42}"
MODE="${MODE:-in-weight}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
MAX_TRAIN_PROGRAMS="${MAX_TRAIN_PROGRAMS:-100000}"

# Keep checkpoints from no-rl runs separate from the main train.sh runs so
# they don't collide on shared run names.
CKPT_DIR="${CKPT_ROOT}/no-rl-${MODE}"

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
    --train-corpus "${ENUM_CORPUS}" \
    --val-corpus "${VAL_CORPUS}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --val-examples 256 \
    --epochs $EPOCHS \
    --mode "${MODE}" \
    --weight-decay $WEIGHT_DECAY \
    --max-train-programs $MAX_TRAIN_PROGRAMS \
    --constant-lr \
    --seed $SEED \
    --num-workers $NUM_WORKERS \
    "${RUN_NAME_FLAG[@]}" \
    "${RESUME_FLAG[@]}"
