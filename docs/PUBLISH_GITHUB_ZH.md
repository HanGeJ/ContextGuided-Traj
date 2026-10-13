# GitHub 发布顺序（先私有仓库）

1. 在发布前执行 `python tools/check_release.py`，确认仓库没有数据集、`.pt`、私人路径和 token。**当前包尚无正式权重，也未完成新的独立 GPU 全流程验证。**
2. 使用已连接的 GitHub 工具或已安装 `gh` 的终端：

```bash
git init
git add .
git commit -m "Add paper-aligned reproduction source (release candidate)"
gh auth login
gh repo create ContextGuided-Traj --private --source=. --remote=origin --push
```

3. 建议保持私有，完成权重导出和所有数据/许可审核。
4. 选择许可证和正确实验复现证明后，再按自己的意愿公开仓库，添加 release tag。**不要把评估错误修复、实验日志或原始 SDD 场景帧强制公开。**

如果 ChatGPT 内的 GitHub 连接已配置，可以指定仓库 owner/repo 和可见性，由连接器处理仓库创建及文件上传。

## Windows 图形界面

首次发布的完整 GitHub Desktop 操作参见 `PUBLISH_WINDOWS_GITHUB_DESKTOP_ZH.md`。
ZIP 含 100 个以上文件，GitHub 网页单次上传不方便；不要只上传 ZIP 文件。
