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

# Under sbatch, $0 points to a Slurm spool copy; use SLURM_SUBMIT_DIR instead.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

source ~/.bashrc

conda init bash
conda deactivate
conda activate ml13

# Parametrised so non-default grammars (see src/lang/grammar.py:GRAMMARS) can be
# synthesised into their own corpus dir without touching the default path. With
# no overrides this reproduces the original corpus-a / rule_val.json run exactly.
#   GRAMMAR=small \
#   OUTPUT_DIR=/n/netscratch/.../mlmp_datasets/corpus-small \
#   VAL_OUT=/n/netscratch/.../mlmp_datasets/corpus-small/val.json \
#   sbatch scripts/generate_dataset.slurm.sh
GRAMMAR="${GRAMMAR:-default}"
OUTPUT_DIR="${OUTPUT_DIR:-/n/netscratch/gershman_lab/Lab/yiding/mlmp_datasets/corpus-a}"
VAL_OUT="${VAL_OUT:-/n/netscratch/gershman_lab/Lab/yiding/mlmp_datasets/rule_val.json}"
SEED="${SEED:-42}"

# ── Phase 1-3: synthesise the corpus (~5.7M programs for the default grammar) ─
#   Phase 1: bottom-up enumeration (s_max=8, ν_min=0.3, ℓ_max=2)
#   Phase 2: sketch expansion (~1.79M concrete after dedup)
#   Phase 3: warm-start + RL (100K novel sketches, δ_max=12)
#            + post-RL expansion (~3.9M concrete after dedup)
# Settings baked into src/data/generate.py + src/lang/synthesis/pipeline.py;
# match docs/enumeration-rl.tex. Resumable via enum_corpus.pkl /
# policy_warmstart.pt checkpoints in OUTPUT_DIR.
python -m src.data.generate \
    --grammar "${GRAMMAR}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed ${SEED}

# ── Phase 4: Rule et al. split + fingerprint filter ──────────────────────────
# Fingerprints under ${GRAMMAR}, so a subset grammar's val set is automatically
# restricted to the Rule behaviours it can express (e.g. SmallGrammar keeps ~84
# of the 220 Rule programs). Writes:
#   ${OUTPUT_DIR}/{enum,rl}_corpus_no_rule.json   (train; Rule fingerprints removed)
#   ${VAL_OUT}                                    (validation; Rule programs)
python -m scripts.build_rule_split \
    --grammar "${GRAMMAR}" \
    --val-out "${VAL_OUT}" \
    --train-corpus "${OUTPUT_DIR}/enum_corpus.json" \
    --train-corpus "${OUTPUT_DIR}/rl_corpus.json"

# ── Phase 5: equality-saturation simplification of the RL corpus ─────────────
# Rules that mention out-of-grammar functions are inert (their LHS can't match a
# subset-grammar program) and no rule introduces one on its RHS, so the default
# rule set is sound for every grammar. Writes
# ${OUTPUT_DIR}/rl_corpus_no_rule.simplified.json — train.slurm.sh reads this.
python -m scripts.simplify_corpus \
    --input "${OUTPUT_DIR}/rl_corpus_no_rule.json"
