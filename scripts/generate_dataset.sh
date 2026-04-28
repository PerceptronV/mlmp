#!/bin/bash
# ============================================================================
# Local Dataset Generation Script (without Slurm)
# ============================================================================
# Runs the corpus-a synthesis pipeline followed by the Rule et al. split:
#   Phase 1: Bottom-up enumeration (s_max=8, ν_min=0.3, ℓ_max=2)
#   Phase 2: Sketch expansion (target ~6M concrete -> ~1.79M after dedup)
#   Phase 3: Warm-start + RL collection (100K novel sketches, δ_max=12)
#            + post-RL expansion (target ~4M concrete)
#   Phase 4: Build Rule et al. validation set + filter Rule fingerprints
#            from the train corpora (-> *_no_rule.json + rule_val.json).
# Settings match docs/enumeration-rl.tex; baked into src/data/generate.py
# and src/lang/synthesis/pipeline.py.
#
# Usage:
#   ./scripts/generate_dataset.sh
#   OUTPUT_DIR=/path/to/out SEED=42 ./scripts/generate_dataset.sh
# ============================================================================

set -e

cd "$(dirname "$0")/.."

OUTPUT_DIR="${OUTPUT_DIR:-output/corpus-a}"
SEED="${SEED:-42}"
VAL_OUT="${VAL_OUT:-$(dirname "${OUTPUT_DIR}")/rule_val.json}"

echo "=============================================="
echo "Phase 1-3: Corpus-A Generation"
echo "=============================================="

python -m src.data.generate \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    "$@"

echo ""
echo "=============================================="
echo "Phase 4: Rule et al. split + fingerprint filter"
echo "=============================================="

python -m scripts.build_rule_split \
    --val-out "${VAL_OUT}" \
    --train-corpus "${OUTPUT_DIR}/enum_corpus.json" \
    --train-corpus "${OUTPUT_DIR}/rl_corpus.json"

echo ""
echo "Done!"
echo "  Train: ${OUTPUT_DIR}/{enum,rl}_corpus_no_rule.json"
echo "  Val:   ${VAL_OUT}"
