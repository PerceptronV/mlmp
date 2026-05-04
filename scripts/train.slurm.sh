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

python -m src.train \
    --train-corpus "${ENUM_CORPUS},${RL_CORPUS}" \
    --val-corpus "${VAL_CORPUS}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --val-examples 256 \
    --mode "${MODE}" \
    --seed $SEED \
    --num-workers $NUM_WORKERS
