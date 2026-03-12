#!/bin/bash
# ============================================================================
# Local Dataset Generation Script (without Slurm)
# ============================================================================
# Usage:
#   ./scripts/generate_dataset.sh
#   ./scripts/generate_dataset.sh --n-train 10000 --num-workers 4
# ============================================================================

set -e

cd "$(dirname "$0")/.."

echo "=============================================="
echo "Dataset Generation (Local)"
echo "=============================================="

python -m src.data.generate_with_inverse \
    --n-train ${N_TRAIN:-100000} \
    --n-validation 210 \
    --n-support 30 \
    --n-io 11 \
    --max-program-depth 7 \
    --seed ${SEED:-42} \
    --noise 0.3 \
    --coverage-strength 3.0 \
    --num-workers ${NUM_WORKERS:-4} \
    --semantic-variations \
    --canonical-prob 0.01 \
    "$@"

echo ""
echo "Done!"
