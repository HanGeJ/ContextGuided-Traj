# ContextGuided-Traj

Code for *Context-Guided Trend–Residual Learning for Efficient Multi-Hypothesis Pedestrian Trajectory Forecasting*.

[中文说明](README_ZH.md)

The model combines a shared motion trend with hypothesis-specific residuals to
predict several possible futures. Neighboring trajectories and scene features
provide social and visual context. The repository includes training and
evaluation scripts for ETH–UCY and the Stanford Drone Dataset (SDD).

## Setup

Use Python 3.10 or 3.11. Install PyTorch and torchvision for your CUDA version
following the [PyTorch installation guide](https://pytorch.org/get-started/locally/),
then install the remaining dependencies and check the model forward pass:

```bash
python -m pip install -r requirements.txt
python tools/smoke_model.py
```

Run the shell commands below in Bash (Linux, WSL, or Git Bash).

## Data and checkpoints

Prepare the datasets using [docs/DATASETS.md](docs/DATASETS.md). It describes the
required files, split definitions, directory layout, and SDD image crops.

Trained checkpoints are prepared separately. Their paths are listed in the
[checkpoint inventory](docs/CHECKPOINT_INVENTORY_20261014.json).

## Training and evaluation

### ETH–UCY

Prepare the five folds, then train the ETH fold with seed 42:

```bash
python prepare_ethucy_5fold_social.py \
  --raw_root data/eth_ucy_raw \
  --out_dir data/ethucy_5fold_social \
  --obs_len 8 --pred_len 12 --sample_stride 1 --neighbor_topk 4

FOLDS=eth SEEDS=42 bash scripts/reproduce_ethucy.sh
```

Run `bash scripts/reproduce_ethucy.sh` for all five folds and seeds 42, 43, and 44.
Checkpoint selection uses training-group validation
(`val_source=train_group`, `val_ratio=0.1`); the held-out scene is used for evaluation.

### SDD

Prepare the trajectories, image crops, and saved checkpoints. The check command
prints each required asset and its expected path:

```bash
bash scripts/reproduce_sdd.sh --check
```

Once the assets are ready, run the saved-checkpoint evaluation:

```bash
ACKNOWLEDGE_SDD_HELDOUT=1 bash scripts/reproduce_sdd.sh --evaluate
```

SDD benchmark results are reported in pixels. The visual ablation records use meters.

## Other experiments

- [TMM_TableIV_ControlledRepro_v1/](TMM_TableIV_ControlledRepro_v1/): controlled augmentation experiments.
- [experiments/efficiency/](experiments/efficiency/): model size and inference-time benchmarks.
- [results/](results/): recorded experiment results.

## Tests

Run the tool tests with:

```bash
python -m unittest discover -s tests -v
```

## Citation and license

Citation metadata is provided in [CITATION.cff](CITATION.cff).

The source code is released under the [MIT License](LICENSE). External datasets
and pretrained weights retain their own license terms.
