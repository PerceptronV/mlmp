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

source ~/.bashrc

module load cuda/12.4.1-fasrc01
module load cudnn/9.5.1.17_cuda12-fasrc01

conda init bash
conda deactivate
conda activate ml13

nvidia-smi

ENUM_CORPUS="/n/netscratch/gershman_lab/Lab/yiding/mlmp_datasets/corpus-a/enum_corpus_no_rule.json"
RL_CORPUS="/n/netscratch/gershman_lab/Lab/yiding/mlmp_datasets/corpus-a/rl_corpus_no_rule.json"
VAL_CORPUS="/n/netscratch/gershman_lab/Lab/yiding/mlmp_datasets/rule_val.json"
CKPT_DIR="/n/netscratch/gershman_lab/Lab/yiding/mlmp_checkpoints"
NUM_WORKERS=8
SEED=42

python -m src.train \
    --train-corpus "${ENUM_CORPUS},${RL_CORPUS}" \
    --val-corpus "${VAL_CORPUS}" \
    --checkpoint-dir "${CKPT_DIR}" \
    --seed $SEED \
    --num-workers $NUM_WORKERS
