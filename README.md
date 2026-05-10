# MLMP - Meta-Learning Metaprogram Search

Meta-learning metaprogram search for efficient, human-like program induction.

## Quick Start

Clone the repository, initialise submodules, and download OSF data to compare against from [Rule et al. (2024)](https://www.nature.com/articles/s41467-024-50966-x):

```bash
git clone https://github.com/PerceptronV/mlmp
cd mlmp
# Initialise submodules for baseline testing against Lake & Baroni (2023)
git submodule update --init --recursive
# Ensure you have curl and unzip installed
./scripts/setup.sh
```

Install dependencies in your environment of choice:

```bash
# micromamba activate ... / conda activate ... / ...
pip install -e .
```

## Dataset Generation

To generate the dataset (defaults to `~/mlmp_datasets/`), run the following script in your environment of choice:

```bash
./scripts/generate_dataset.sh
```

This takes a few hours to complete, and walks through the following phases:
1. Bottom-up enumeration: enumerate programs sketches up to size 8.
2. Post-enumeration expansion: substitute random constants into integer holes to expand the enumeration sketches into concrete programs.
3. Warm-start + RL collection: train a program synthesis policy via warm-start + train policy with RL and a priority queue buffer; collect novel sketches up to depth 12.
4. Post-RL expansion: expand the RL sketches into concrete programs.
5. Rule et al. split: build the Rule et al. validation set and filter Rule fingerprints from the train corpora.
6. Equality-saturation simplification of the RL corpus: simplify the RL corpus to make it approach the minimum description length.

## Training

The training scripts are `./scripts/{[train.sh](./scripts/train.sh),[small-train.sh](./scripts/small-train.sh)}`. The small-train script is a variant of the train script that trains on the enum corpus only (no RL), and gives higher validation accuracy more quickly.

In your environment of choice, run:

```bash
MODE=<mode> ./scripts/{train,small-train}.sh
```

Available modes are:

- `in-weight`: in-weight training
- `easy-symbol-shuffling`: symbol shuffling with curriculum learning
- `symbol-shuffling`: full symbol shuffling training from the onset

By default, the checkpoints are saved in `~/mlmp_checkpoints`. You can change this by setting the `CKPT_DIR` environment variable.

## Analysis

In your environment of choice, run:

```bash
python -m src.analysis <path/to/config.yaml>
```

The analysis scripts are used to analyse the training results, with default configs in `[src/analysis/configs/](./src/analysis/configs)`. Available analysis configs are:


| Config          | Purpose                                                                          | Command                                                     |
| --------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `main.yaml`     | Main analysis config                                                             | `python -m src.analysis src/analysis/configs/main.yaml`     |
| `probing.yaml`  | Probing config                                                                   | `python -m src.analysis src/analysis/configs/probing.yaml`  |
| `csv_only.yaml` | CSV-only smoke config that reproduces the analysis results in Rule et al's paper | `python -m src.analysis src/analysis/configs/csv_only.yaml` |


*Note*: change the Transformer `run_name`s in the config files to the actual run names you used.
