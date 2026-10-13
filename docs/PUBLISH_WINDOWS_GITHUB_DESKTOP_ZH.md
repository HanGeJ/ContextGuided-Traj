# Windows 操作：从 RC2 源码 ZIP 发布到 GitHub 私有仓库

本流程针对首次发布，避免在 GitHub 网页一次上传全部源码文件。

1. 下载 `ContextGuided_Traj_PrivateGitHub_RC2.zip` 并解压。进入包含
   `README.md`、`train.py`、`models/`、`data/`、`tools/` 的
   **ContextGuided-Traj** 文件夹。不要将外层压缩包提交为仓库中的单个 ZIP。
2. 安装并登录 [GitHub Desktop](https://desktop.github.com/)。
3. 在 GitHub Desktop 中选择 **File → Add local repository**，定位到
   解压的 `ContextGuided-Traj` 文件夹。首次可能提示目录没有 Git 仓库，
   点击 **Create a repository here**（根据版本可能为“Create a repository”）。
   若 Desktop 没有显示该选项，可在此文件夹用 Git Bash 执行
   `git init -b main`，再返回 Desktop 添加目录。
4. 在 Changes 中确认源码存在。重点检查
   `data/dataset.py`、`data/visual_dataset.py`、`data/__init__.py`，以及
   `models/components.py`、`train.py`、`README.md` 都会被纳入提交。
   **切勿加入** 原始数据集、任何 `.pt` 模型权重或私有运行目录。
5. 在 Summary 填写 `Initial private paper reproduction candidate`，
   点击 **Commit to main**。
6. 点击 **Publish repository**：名称 `ContextGuided-Traj`，
   勾选 **Keep this code private**；点击 Publish Repository。完成后，
   在 GitHub 仓库网页检查 `data/` 包含 Python 源码且不存在私有数据。
7. 把实际仓库网址发回对话，继续核验目录、README、许可及发布条件。

此 ZIP 只包含源码、受控实验记录和模型哈希清单，不包含 checkpoint。
`docs/CHECKPOINT_INVENTORY_NOTES.md` 解释了 51 个权重当前的证据级别。

**关于公开：** 在权利授权、净环境论文指标复现、Fig.6 帧配准等审核之前，
请勿将 Private 改为 Public。GitHub 仓库可见性可在 **Settings → General →
Danger Zone → Change repository visibility** 调整。
