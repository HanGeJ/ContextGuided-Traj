# ContextGuided-Traj

论文 *Context-Guided Trend–Residual Learning for Efficient Multi-Hypothesis
Pedestrian Trajectory Forecasting* 的代码。

[English](README.md)

模型将预测轨迹分解为共享运动趋势和各个未来假设的残差。邻居轨迹提供社会
上下文，场景特征提供视觉上下文。仓库包含 ETH–UCY 和 Stanford Drone Dataset
（SDD）的训练与评估脚本。

## 环境配置

使用 Python 3.10 或 3.11，按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)
安装与 CUDA 版本匹配的 PyTorch 和 torchvision，然后安装其余依赖，检查模型前向：

```bash
python -m pip install -r requirements.txt
python tools/smoke_model.py
```

下文的 Shell 命令使用 Bash，可在 Linux、WSL 或 Git Bash 中执行。

## 数据与权重

按 [docs/DATASETS.md](docs/DATASETS.md) 准备数据。文档说明了所需文件、数据划分、
目录结构和 SDD 图像裁剪要求。

训练权重需单独准备，路径见
[权重清单](docs/CHECKPOINT_INVENTORY_20261014.json)。

## 训练与评估

### ETH–UCY

先生成五折数据，再以种子 42 训练 ETH 折：

```bash
python prepare_ethucy_5fold_social.py \
  --raw_root data/eth_ucy_raw \
  --out_dir data/ethucy_5fold_social \
  --obs_len 8 --pred_len 12 --sample_stride 1 --neighbor_topk 4

FOLDS=eth SEEDS=42 bash scripts/reproduce_ethucy.sh
```

执行 `bash scripts/reproduce_ethucy.sh` 可运行全部五折及种子 42、43、44。
脚本使用训练组验证集选择权重（`val_source=train_group`、`val_ratio=0.1`），
留出场景用于评估。

### SDD

准备轨迹数据、图像裁剪缓存和已有权重。以下命令会逐项检查所需文件并显示路径：

```bash
bash scripts/reproduce_sdd.sh --check
```

文件齐备后，运行已有权重的评估：

```bash
ACKNOWLEDGE_SDD_HELDOUT=1 bash scripts/reproduce_sdd.sh --evaluate
```

SDD 基准结果以像素报告，视觉消融记录使用米。

## 其他实验

- [TMM_TableIV_ControlledRepro_v1/](TMM_TableIV_ControlledRepro_v1/)：受控训练增强实验。
- [experiments/efficiency/](experiments/efficiency/)：模型规模与推理耗时测试。
- [results/](results/)：已有实验结果记录。

## 工具测试

```bash
python -m unittest discover -s tests -v
```

## 引用与许可证

引用信息见 [CITATION.cff](CITATION.cff)。

源码采用 [MIT 许可证](LICENSE)。外部数据集和预训练权重遵循各自的许可条款。
