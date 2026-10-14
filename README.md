# ContextGuided-Traj

Code for *Context-Guided Trend–Residual Learning for Efficient Multi-Hypothesis Pedestrian Trajectory Forecasting*.

The implementation includes trend–residual trajectory generation, social context,
candidate-aligned visual correction, and training augmentation. Training and
evaluation scripts cover ETH–UCY and SDD.

[中文说明](README_ZH.md)

## Installation

Use Python 3.10 or 3.11. Install a compatible PyTorch/torchvision pair for your
CUDA environment using the [PyTorch installation guide](https://pytorch.org/get-started/locally/), then:

```bash
python -m pip install -r requirements.txt
python tools/smoke_model.py
```

The shell scripts below run with Bash on Linux, WSL, or Git Bash.

## Data and checkpoints

See [docs/DATASETS.md](docs/DATASETS.md) for dataset sources, directory layouts,
SDD recording splits, scale metadata, and image crop preparation.
Datasets and trained checkpoints are obtained separately. The checkpoint
inventory is in [docs/CHECKPOINT_INVENTORY_20261014.json](docs/CHECKPOINT_INVENTORY_20261014.json).

## ETH–UCY

Prepare the five folds, then run one fold and seed:

```bash
python prepare_ethucy_5fold_social.py --raw_root data/eth_ucy_raw --out_dir data/ethucy_5fold_social --obs_len 8 --pred_len 12 --sample_stride 1 --neighbor_topk 4
FOLDS=eth SEEDS=42 bash scripts/reproduce_ethucy.sh
```

Run `bash scripts/reproduce_ethucy.sh` for all five folds and seeds 42, 43, and 44.
The script selects checkpoints using training-group validation
(`val_source=train_group`, `val_ratio=0.1`) and evaluates on the held-out scene.

## SDD

Prepare the data and image crops using the dataset documentation. Place the
frozen checkpoints in the paths checked by `scripts/reproduce_sdd.sh`, then:

```bash
bash scripts/reproduce_sdd.sh --check
ACKNOWLEDGE_SDD_HELDOUT=1 bash scripts/reproduce_sdd.sh --evaluate
```

SDD benchmark results use pixels; the controlled ablation records use meters.

## Tests and tools

```bash
python -m unittest discover -s tests -v
python tools/check_release.py
```

- `train.py` and `evaluate.py`: core model training and evaluation.
- `TMM_TableIV_ControlledRepro_v1/`: controlled augmentation experiments.
- `experiments/efficiency/`: inference benchmarking and its timing protocol.
- `tools/export_autodl_checkpoints.py`: export the recorded checkpoints locally.
- `tools/verify_staged_checkpoints.py`: verify an export against its inventory.
- `tools/check_git_index.py`: verify staged source files against the staged hash manifest.

Configuration and result records are available in `results/` and `docs/`.
Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

Source code is distributed under the [MIT License](LICENSE). External datasets,
pretrained weights, and external repositories retain their respective terms.
