#!/bin/bash
#SBATCH -c 24                               # Number of cores (-c)
#SBATCH -t 0-10:00                          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p sapphire                         # Partition to submit to
#SBATCH --mem=64G                           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o logs/generate_dataset_%j.out     # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logs/generate_dataset_%j.err     # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yidingsong@college.harvard.edu

conda init bash
conda activate ml13

OUTPUT_DIR="/n/netscratch/gershman_lab/Lab/yiding/mlmp_datasets"
NUM_WORKERS=24
SEED=42

python -m src.data.generate_with_inverse \
    --output-dir $OUTPUT_DIR \
    --seed $SEED \
    --num-workers $NUM_WORKERS \
    --semantic-variations \
    --noise-annealing
