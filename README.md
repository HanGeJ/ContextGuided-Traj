# ContextGuided-Traj: A paper-aligned reproduction package

**Paper:** *Context-Guided Trend–Residual Learning for Efficient Multi-Hypothesis Pedestrian Trajectory Forecasting* (IEEE TIP submission candidate).

This is a **private-staging release candidate (RC2)** reconstructed from the archived TMM-v1 Stage 3E and Stage SDDS project code and the controlled Table IV audit. It deliberately excludes datasets, third-party code, downloaded pretrained weights, model checkpoints, and raw visualization frames.

> **Release state: candidate / not yet independently end-to-end verified.** Code provenance, CPU model-forward smoke and aggregate audit data are included. Reproducing published ETH–UCY and SDD scores requires the official datasets, the exact frozen checkpoints, and a compatible CUDA environment. Do not describe this archive as a one-click reproduction of the paper until the checklist in `docs/RELEASE_CHECKLIST.md` passes on a clean machine.

## Implemented method

- Shared time-linear trend + K=20 residual hypotheses, confidence logits, coordinate-wise variances
- Lightweight social encoding of observed relative neighbor histories
- Optional candidate-conditioned visual adapter using a frozen pretrained MobileNetV3-Small
- Training-only rotation, observation masking and Gaussian observation noise
- Five-fold ETH–UCY model training and evaluation, SDD locked evaluation, controlled augmentation ablation and audited efficiency tooling

## Minimal setup

Python 3.10–3.11 and a mutually compatible PyTorch/torchvision pair are recommended. Install CUDA-enabled packages using the [official PyTorch instructions](https://pytorch.org/get-started/locally/) before `pip install -r requirements.txt`. Images are downloaded by torchvision only when the optional ImageNet-pretrained visual encoder is instantiated. Dataset downloads are **not** performed implicitly.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tools/smoke_model.py
python tools/check_release.py
```

## Reproduce ETH–UCY (5-fold LOSO)

Obtain ETH/UCY from the rights holders. Arrange the canonical files under `data/eth_ucy_raw` as documented in `docs/DATASETS.md`. Then:

```bash
python prepare_ethucy_5fold_social.py \
  --raw_root data/eth_ucy_raw \
  --out_dir data/ethucy_5fold_social \
  --obs_len 8 --pred_len 12 --sample_stride 1 --neighbor_topk 4
# First run ONE fold for verification:
FOLDS=eth SEEDS=42 bash scripts/reproduce_ethucy.sh
# Full paper-controlled generator (5 folds x 3 seeds):
bash scripts/reproduce_ethucy.sh
```

- **Model selection**: `val_source=train_group`, validation ratio 0.1, held-out scene used only for final evaluation.
- **Training**: `multihyp_social`, 100 epochs, Adam, lr=0.01, batch=512, K=20, augmentation enabled, classification lambda=0.2, variance lambda=0.1.
- The training command saves a `best_model.pt` under `runs/tmm_stage3e_ethucy_final/...`; the held-out result and configuration are in `summary.json`.
- `baseline` and `multihyp` for Table III: `VARIANTS='baseline multihyp multihyp_social' bash scripts/reproduce_ethucy.sh`.

## Reproduce SDD

SDD is **not** available in this repository. It requires raw annotations and reference images, the specific TUTR recording split, and recording-scale YAML. See `docs/DATASETS.md`. The published SDD results use distinct pixel-space benchmark scores and meter-space controlled ablations; never mix these units. `scripts/reproduce_sdd.sh` checks the three frozen generator/geometry checkpoint pairs before running a guarded evaluator.

```bash
bash scripts/reproduce_sdd.sh --check
# After assembling licensed data, crop caches and frozen checkpoints:
bash scripts/reproduce_sdd.sh --evaluate
```

The SDD evaluator contains its original seal mechanism and refuses another test by default. Repeated held-out testing for model selection is prohibited.

## Reproduce controlled robustness ablation (Table IV)

The original complete controlled runner is preserved in `TMM_TableIV_ControlledRepro_v1`. It requires **60 full trainings** (4 arms × 5 folds × 3 seeds), using the exact four settings and `train_group` model selection. To use the historical runner:

```bash
python TMM_TableIV_ControlledRepro_v1/repro.py --help
```

The 60 original experiment summaries and aggregate values are available as provenance under `results/table_iv`. **Only the new controlled reproduction** (not the obsolete original exploratory values) supports Table IV.

## Efficiency

`experiments/efficiency` preserves audited GPU benchmarking scripts and source-limited run summaries. The 0.660 ms *single-target native inference*, the multi-target GPU-resident inference and the *prepared in-memory pipeline* use different timing scopes. Neither the GitHub README nor plots shall blend these metrics into a single speedup claim. See `docs/EFFICIENCY_PROTOCOL.md`.

## GitHub private staging

The source-only package can be published first to a **private** repository.
For Windows upload, read `docs/PUBLISH_WINDOWS_GITHUB_DESKTOP_ZH.md`.
Git ignores raw datasets and checkpoints but retains the required `data/*.py`
source files; `tools/check_git_index.py` can verify the actual staged source
manifest before pushing.

The AutoDL checkpoint audit reported 51/51 existing files (23.87 MiB), but
this was **inventory-only**; the checkpoint bytes and public redistribution
rights have not been verified. See `docs/CHECKPOINT_INVENTORY_NOTES.md`.

## Weights, data and re-drawing Fig. 6/7

- `tools/export_autodl_checkpoints.py` inventories or exports the known formal checkpoint paths from an AutoDL project, with SHA256 checksums; it excludes raw datasets and third-party weights.
- `docs/DATASETS.md` describes required inputs and their external origins. Follow the respective data terms.
- Figure source scripts are placed in `visualization/`; the native scene frames and prediction arrays must be provided separately under the proper permissions. Fig. 6's historical video timing/body-anchor alignment remains **unverified**, and should be audited before its underlying array bundle is released as verified examples.

## Status and citation

See `docs/RELEASE_CHECKLIST.md` and `docs/PROVENANCE.md` for verified versus pending items. The publication citation can be added after journal acceptance; do not assert publication at this stage.

## License

**No open-source license has been chosen.** A public GitHub repository without a license does not grant a general right to redistribute or reuse its contents. Confirm ownership/affiliation permissions and choose a license before public release. See `docs/LICENSING.md`.
