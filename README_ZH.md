# ContextGuided-Traj

论文 *Context-Guided Trend–Residual Learning for Efficient Multi-Hypothesis
Pedestrian Trajectory Forecasting* 的代码。

包含趋势–残差轨迹生成、社会上下文、候选对齐视觉修正和训练增强，以及
ETH–UCY、SDD 的数据准备、训练和评估脚本。

## 安装

使用 Python 3.10 或 3.11，按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)
安装与 CUDA 环境匹配的 PyTorch/torchvision，然后执行：

```bash
python -m pip install -r requirements.txt
python tools/smoke_model.py
```

Shell 脚本使用 Bash，可在 Linux、WSL 或 Git Bash 中执行。

## 数据与权重

数据来源、目录结构、SDD 视频划分、尺度信息和图像裁剪要求见
[docs/DATASETS.md](docs/DATASETS.md)。数据集与训练权重需单独准备；权重
路径清单见 [docs/CHECKPOINT_INVENTORY_20261014.json](docs/CHECKPOINT_INVENTORY_20261014.json)。

## ETH–UCY

```bash
python prepare_ethucy_5fold_social.py --raw_root data/eth_ucy_raw --out_dir data/ethucy_5fold_social --obs_len 8 --pred_len 12 --sample_stride 1 --neighbor_topk 4
FOLDS=eth SEEDS=42 bash scripts/reproduce_ethucy.sh
```

执行 `bash scripts/reproduce_ethucy.sh` 可运行五折和种子 42、43、44。
脚本按 `val_source=train_group`、`val_ratio=0.1` 划分验证组，用留出场景评估。

## SDD

按数据文档准备数据和图像裁剪缓存，将冻结权重放在
`scripts/reproduce_sdd.sh` 检查的路径，然后执行：

```bash
bash scripts/reproduce_sdd.sh --check
ACKNOWLEDGE_SDD_HELDOUT=1 bash scripts/reproduce_sdd.sh --evaluate
```

SDD 基准结果使用像素，受控消融记录使用米。

## 检查与工具

```bash
python -m unittest discover -s tests -v
python tools/check_release.py
```

`train.py`、`evaluate.py` 提供核心训练与评估；`TMM_TableIV_ControlledRepro_v1/`
提供受控增强实验；`experiments/efficiency/` 提供效率测试。导出权重使用
`tools/export_autodl_checkpoints.py`，完整性核验使用
`tools/verify_staged_checkpoints.py`。`tools/check_git_index.py` 核对 Git 暂存区
文件与暂存区哈希清单。

## 许可证

源码采用 [MIT 许可证](LICENSE)。外部数据集、预训练权重和外部仓库遵循
各自的许可条款。引用信息见 [CITATION.cff](CITATION.cff)。
