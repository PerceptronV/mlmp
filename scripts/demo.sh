#!/bin/bash

mkdir -p media/composer_results/

for composer in template random empirical mcts; do
    python scripts/experiment_composers.py --composer $composer --num-samples 30 --visualize --depth 4 \
    > media/composer_results/$composer.txt
    echo $composer "done"
done

python scripts/experiment_composers.py \
    --compare random,empirical,template,mcts \
    --num-samples 30 \
    --depth 6 \
    --seed 42 \
    --output media/comparison_results.json

python scripts/visualize_results.py media/comparison_results.json --output media/comparison_plots/
