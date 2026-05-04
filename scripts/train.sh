#!/bin/bash
# ============================================================================
# Local Training Script (without Slurm)
# ============================================================================
# Mirrors scripts/train.slurm.sh but for a non-Slurm host (rented GPU box,
# workstation, etc.). Activates conda, prints GPU info, then launches
# src.train. Output is teed into logs/train_<timestamp>.log.
#
# Usage:
#   ./scripts/train.sh
#   DATA_ROOT=/path/to/data CKPT_DIR=/path/to/ckpts SEED=7 ./scripts/train.sh
# ============================================================================

cd "$(dirname "$0")/.."

eval "$(micromamba shell hook --shell bash)"
micromamba activate ml13

DATA_ROOT="${DATA_ROOT:-$HOME/yiding-in-georgia/datasets}"
ENUM_CORPUS="${DATA_ROOT}/corpus-a/enum_corpus_no_rule.json"
RL_CORPUS="${DATA_ROOT}/corpus-a/rl_corpus_no_rule.simplified.json"
VAL_CORPUS="${DATA_ROOT}/rule_val.json"
CKPT_DIR="${CKPT_DIR:-$HOME/yiding-in-georgia/mlmp_checkpoints}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-42}"

python -m src.train \
    --train-corpus "${ENUM_CORPUS},${RL_CORPUS}" \
    --val-corpus "${VAL_CORPUS}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --compile-layers \
    --val-examples 256 \
    --seed $SEED \
    --num-workers $NUM_WORKERS
