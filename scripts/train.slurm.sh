#!/bin/bash
#SBATCH --job-name=mlmp_training_inweight
#SBATCH -c 1                                # Number of cores (-c)
#SBATCH -t 1-00:00                          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner                          # Partition to submit to
#SBATCH --account=kempner_undergrads
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G                          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=logs/train_%j.out          # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error=logs/train_%j.err           # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yidingsong@college.harvard.edu

set -e

# Under sbatch, $0 points to a Slurm spool copy; use SLURM_SUBMIT_DIR instead.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p logs

source ~/.bashrc

module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01

conda init bash
conda deactivate
conda activate ml13

nvidia-smi

ENUM_CORPUS="/n/netscratch/gershman_lab/Lab/yiding/mlmp_datasets/corpus-a/enum_corpus_no_rule.json"
RL_CORPUS="/n/netscratch/gershman_lab/Lab/yiding/mlmp_datasets/corpus-a/rl_corpus_no_rule.simplified.json"
VAL_CORPUS="/n/netscratch/gershman_lab/Lab/yiding/mlmp_datasets/rule_val.json"
CKPT_ROOT="/n/netscratch/gershman_lab/Lab/yiding/mlmp_checkpoints"
NUM_WORKERS=8
SEED=42
MODE="${MODE:-in-weight}"

# Keep checkpoints from different training modes in separate subdirectories so
# in-weight and symbol-shuffling runs don't overwrite each other's best/latest.
CKPT_DIR="${CKPT_ROOT}/${MODE}"

# If RUN_NAME is provided, pin checkpoints to ${CKPT_DIR}/${RUN_NAME}/ and
# auto-resume from checkpoint_latest.pt there if one exists. If RUN_NAME is
# unset, fall back to the previous behaviour: wandb assigns a fresh name and
# the run starts from scratch.
#   sbatch --export=ALL,MODE=symbol-shuffling,RUN_NAME=ss-run scripts/train.slurm.sh
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
    --mode "${MODE}" \
    --seed $SEED \
    --num-workers $NUM_WORKERS \
    "${RUN_NAME_FLAG[@]}" \
    "${RESUME_FLAG[@]}"
