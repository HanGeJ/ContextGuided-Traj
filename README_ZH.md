# ContextGuided-Traj 开源复现候选包

这是从历史 AutoDL TMM-v1 Stage 3E / Stage SDDS 代码中整理的**论文复现候选版**。目前已包含核心模型、ETH–UCY/SDD 数据转换及评估程序、Table IV 原生实验代码、GPU 效率脚本及审计结果。**尚缺正式权重及原始数据**，因此不能宣称已在新容器完整复现论文指标。

按顺序操作：

1. 在本地验证：`python tools/smoke_model.py && python tools/check_release.py`。
2. 根据 `docs/DATASETS.md` 准备获得授权的数据；运行 `prepare_ethucy_5fold_social.py`。
3. 在 AutoDL 现存项目根目录执行 `python tools/export_autodl_checkpoints.py --source /root/autodl-tmp/TMM_Traj/source/tmm_v1 --destination /root/autodl-tmp/ContextGuided_export`，默认仅输出缺失检查和哈希清单。
4. 如权重具有发布许可，再使用 `--include-weights` 复制正式权重；权重备份与公开发布须分开审核。
5. 通过 `docs/RELEASE_CHECKLIST.md` 中的实际复现检查后，决定 README、许可证与 GitHub 仓库公开状态。

`bash scripts/reproduce_ethucy.sh` 运行五折/三种子正式模型训练；`bash scripts/reproduce_sdd.sh --check` 检查冻结的视觉模型目录。详细命令和评估口径在英文 README 与 docs 内。

**注意：** 图 6 原始帧配准曾被标记为未完全审计。重画 EPS 不会自动解决时间映射问题。不要把该问题标成已验证。

## GitHub 上传修复（RC2）

RC1 的 `.gitignore` 会错误忽略 `data/` 下必须提交的 Python 文件。RC2 已修复，
并增加 `tools/check_git_index.py` 检查 Git 暂存区是否完整。Windows
用户按 `docs/PUBLISH_WINDOWS_GITHUB_DESKTOP_ZH.md` 上传至私有仓库。

上一轮 AutoDL checkpoint 清单核验结果为 51/51（仅库存清点）；详见
`docs/CHECKPOINT_INVENTORY_NOTES.md`。权重尚未经过独立加载和重新评估。
