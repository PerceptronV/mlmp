#!/bin/bash
#SBATCH --job-name=mlmp_corpus_a
#SBATCH -c 1                                # placeholder; see --cpus-per-task below
#SBATCH -t 1-00:00                          # Runtime in D-HH:MM (RL phase can run several hours)
#SBATCH -p sapphire                         # CPU partition; switch to a GPU partition if torch.cuda is desired
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8                   # PyTorch intra-op threads + parallel fingerprint workers
#SBATCH --mem=64G                           # Enumeration bank + ~6M expanded programs are memory-hungry
#SBATCH -o logs/generate_dataset_%j.out
#SBATCH -e logs/generate_dataset_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yidingsong@college.harvard.edu

set -e

cd "$(dirname "$0")/.."
mkdir -p logs

source ~/.bashrc

conda init bash
conda deactivate
conda activate ml13

OUTPUT_DIR="/n/netscratch/gershman_lab/Lab/yiding/mlmp_datasets/corpus-a"
VAL_OUT="/n/netscratch/gershman_lab/Lab/yiding/mlmp_datasets/rule_val.json"
SEED=42

# ── Phase 1-3: synthesise corpus-a (~5.7M programs) ──────────────────────────
#   Phase 1: bottom-up enumeration (s_max=8, ν_min=0.3, ℓ_max=2)
#   Phase 2: sketch expansion (~1.79M concrete after dedup)
#   Phase 3: warm-start + RL (100K novel sketches, δ_max=12)
#            + post-RL expansion (~3.9M concrete after dedup)
# Settings baked into src/data/generate.py + src/lang/synthesis/pipeline.py;
# match docs/enumeration-rl.tex. Resumable via enum_corpus.pkl /
# policy_warmstart.pt checkpoints in OUTPUT_DIR.
python -m src.data.generate \
    --output-dir "${OUTPUT_DIR}" \
    --seed ${SEED}

# ── Phase 4: Rule et al. split + fingerprint filter ──────────────────────────
# Writes:
#   ${OUTPUT_DIR}/{enum,rl}_corpus_no_rule.json   (train; Rule fingerprints removed)
#   ${VAL_OUT}                                    (validation; ~217 Rule programs)
python -m scripts.build_rule_split \
    --val-out "${VAL_OUT}" \
    --train-corpus "${OUTPUT_DIR}/enum_corpus.json" \
    --train-corpus "${OUTPUT_DIR}/rl_corpus.json"

# ── Phase 5: equality-saturation simplification of the RL corpus ─────────────
# Writes ${OUTPUT_DIR}/rl_corpus_no_rule.simplified.json — train.slurm.sh reads this.
python -m scripts.simplify_corpus \
    --input "${OUTPUT_DIR}/rl_corpus_no_rule.json"
