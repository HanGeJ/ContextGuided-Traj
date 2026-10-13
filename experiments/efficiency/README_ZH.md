# v1.3.1：Proposed 邻居预处理严格等价优化对照

目的：v1.3 发现 Proposed N=57 的完整管线延迟达 75.752 ms，但 GPU 前向只有 1.724 ms。此补丁仅替换 Proposed 适配器的 Python 最近邻循环为等价 NumPy 向量化实现，保持现有模型、权重、社交邻居选择和 K=20 不变。其他三个网络适配器和统一计时器完全保留 v1.3 方法。

## 安全边界

- 这是**另一次计时实验**，不能用尚未测量的新数字更新论文；先按同一 GPU 完成实测。
- 本包会解压为独立目录 `TMM_Efficiency_AutoBench_v1_3_1`，产物写入 `runs/tmm_efficiency_autobench_v131/`，不会覆盖 v1.3 的输出或训练 checkpoint。
- 首先逐元素核验随机、多行人、重复轨迹与等距离排序的相对位置、mask 与 reference 代码**完全一致**。
- FlowChain 旧官方测试实际 K=1；本版仍将其标记为未认证，不给出虚假的 K=20 性能。
- 关键输出需按两种测量口径报告：GPU resident forward 与内存预处理+传输+forward，不能混为一谈。

## AutoDL 命令

```bash
cd /root/autodl-tmp/TMM_Traj/source/tmm_v1
unzip -o /root/autodl-tmp/TMM_Efficiency_AutoBench_v1_3_1_PreprocessFix.zip -d .
bash TMM_Efficiency_AutoBench_v1_3_1/CHECK_EQUIVALENCE.sh
bash TMM_Efficiency_AutoBench_v1_3_1/CHECK_GPU_EQUIVALENCE.sh
bash TMM_Efficiency_AutoBench_v1_3_1/CHECK.sh
bash TMM_Efficiency_AutoBench_v1_3_1/RUN_QUICK.sh
```

正式运行：

```bash
tmux new -s tmmeff131
cd /root/autodl-tmp/TMM_Traj/source/tmm_v1
bash TMM_Efficiency_AutoBench_v1_3_1/RUN.sh
```

运行之后上传 `runs/TMM_Efficiency_AutoBench_v1_3_1_review_bundle.zip`。如果未生成该文件，请在 `runs/tmm_efficiency_autobench_v131/` 查找 report 和状态文件。

## 论文处理

已经发现原版复杂场景的 CPU 瓶颈，最终投稿前必须报告完整推理管线的实际效率，不能只选取 GPU 模型前向的有利结果。相同预测目标下，后续可比较 Vectorized-v1.3.1 与原版-v1.3 的两种管线成本，同时保留精度评估使用的 checkpoint 不变。此阶段不增加训练。

补充 GPU 等价核验：若检测到 v1.3 已保存的 UNIV canonical window，还会使用相同 Stage 3E checkpoint 同时运行原始与优化预处理，并检查完整模型预测最大绝对偏差不超过 `1e-6`。失败会中止预检；不会将误差结果认证为改进。
