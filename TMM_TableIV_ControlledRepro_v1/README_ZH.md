# TMM — Table IV Controlled Augmentation Reproduction v1.0

目的：在已存在的 `TMM_Traj/source/tmm_v1` 工程内复现一个**可独立追溯**的 ETH–UCY 鲁棒训练消融，填补旧 Table IV 的证据缺口。**不追求拟合旧表的数值，也不修改原模型。**

## 固定实验约定（所有配置一致）

- 模型：`multihyp_social`；20 条轨迹候选；8 步观测 / 12 步预测；五场景留一场景。
- 五个留出测试折：`eth`, `hotel`, `univ`, `zara1`, `zara2`。
- 随机种子：42、43、44；每配置 15 组，四配置一共 60 次训练。
- `--val_source train_group`；从训练组抽取场景/录像组互斥的验证窗口；`val_ratio=0.1`。
- 使用 100 epochs / Adam / LR=0.01 / batch=512，`lambda_cls=0.2`、`lambda_nll=0.1`、Hard CE。
- checkpoint 仅由训练组内部验证 minADE@20 选择；选定后评估该折的独立留出测试集。
- 单视角评估、K=20；聚合规则：每折三种子均值，再对五折宏平均。

| 训练增强配置 | rotation_max_deg | mask_prob | noise_std |
|---|---:|---:|---:|
| `base_losses`（全部关闭）| 0 | 0 | 0 |
| `rotation_only` | 180 | 0 | 0 |
| `mask_noise_only` | 0 | 0.3 | 0.02 |
| `rotation_mask_noise` | 180 | 0.3 | 0.02 |

遮挡最长 4 个观测步；不使用额外中央反转。全部增强仅用于训练批次。模型、损失、候选数量、优化器与验证方法均相同。

## 一次性运行

将增量包上传至 `/root/autodl-tmp/TMM_TableIV_ControlledRepro_v1.zip`：

```bash
cd /root/autodl-tmp/TMM_Traj/source/tmm_v1
unzip -o /root/autodl-tmp/TMM_TableIV_ControlledRepro_v1.zip -d .

bash TMM_TableIV_ControlledRepro_v1/CHECK.sh
bash TMM_TableIV_ControlledRepro_v1/SMOKE.sh
```

通过后，建议在 tmux 中运行（tmux 已安装时）：

```bash
tmux new -s table4
cd /root/autodl-tmp/TMM_Traj/source/tmm_v1
bash TMM_TableIV_ControlledRepro_v1/RUN.sh
```

如 `tmux` 未安装，可使用 `apt-get update && apt-get install -y tmux`；或在 SSH 长连接终端直接运行。

**无需手动 conda activate。** 脚本优先使用 `/root/miniconda3/envs/tmm_traj/bin/python`，再查找已安装 PyTorch 的 Python。出现 `CUDA unavailable` 时，先定位现有 GPU 环境，也可临时指定 `PYTHON=/path/to/your/env/bin/python`。不要在 CPU 上启动 60 次正式训练。

## 断点续跑

`RUN.sh` 逐个执行 60 个固定任务。每个任务独立保存 `history.json`、`summary.json`、`best_model.pt`。重新运行时严格核实配置和结果完整性，完成的任务会跳过。若在某一任务训练中断，该任务会从 epoch 1 重启；其他已完成任务不需要重做。

代码与数据 SHA256 在首次运行时锁定；后续启动若源代码或数据变化会拒绝混用，避免产生不公平表格。全部新文件写入 `runs/tmm_tableIV_controlled_repro_v1/`，不覆盖 Stage 3E 和 ABC/D0/D1 的权重。

## 检查状态和输出

```bash
bash TMM_TableIV_ControlledRepro_v1/STATUS.sh
bash TMM_TableIV_ControlledRepro_v1/PACKAGE.sh
```

全部 60 组结束后，自动生成：

```
runs/tmm_tableIV_controlled_repro_v1/tableIV_report.md
runs/tmm_tableIV_controlled_repro_v1/tableIV_summary.json
runs/tmm_tableIV_controlled_repro_v1/tableIV_runs.csv
runs/tmm_tableIV_controlled_repro_v1/tableIV_byfold.csv
runs/tmm_tableIV_controlled_repro_v1/tableIV_paired_delta.csv
runs/tmm_tableIV_controlled_repro_v1/TABLE_IV_REPRODUCTION_CANDIDATE.tex
runs/TMM_TableIV_ControlledRepro_v1_review_bundle.zip
```

**只有 60/60 验收通过后，才会生成 `TABLE_IV_REPRODUCTION_CANDIDATE.tex`。** 如果某一种增强组合没有改善，应该如实保留结果，并相应调整论文中的 Robust Learning 结论。旧表中的 0.227/0.389、0.213/0.368、0.218/0.377、0.207/0.359 只用于差异检查，程序不会人为调整新数值。

实验会在预先确定的四组设置下使用 ETH–UCY 的留出测试折做最终评价；绝不以测试指标选择 epoch 或重新调节超参数。已封存的 **SDD 独立测试集完全不在本实验范围内**。

## 重要说明

旧 Table IV 的来源仍未验证。新实验生成的是统一配置、可追溯的正式消融，结果可能与旧数字不同。由于增广组和基线从头训练，计算量为最多 60 次 100-epoch 训练，GPU 总耗时取决于实例性能；请预留较长运行时间。运行前的 smoke 是缩小规模的功能性检查，不输出正式论文指标。
