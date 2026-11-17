# 上传 GitHub

建议仓库名：`intraday-pattern-research`。

建议 Description：`Intraday futures pattern classification with K-Shape clustering and a hybrid GRU-MLP network.`

建议 Topics：`quantitative-finance`、`time-series`、`k-shape`、`gru`、`pytorch`、`futures`、`pattern-recognition`。

## 网页上传

1. 解压项目压缩包，确认进入包含 `README.md` 和 `pyproject.toml` 的目录。
2. GitHub 新建仓库。先用私有仓库检查内容，再按自己的需要设为公开。
3. 使用 Add file → Upload files，上传项目目录内的文件和文件夹。不要额外上传 `runs/`、虚拟环境、原始行情或训练权重。
4. 确保 `.gitignore` 和 `.github/workflows/checks.yml` 也上传；命令行推送对隐藏文件更可靠。

## Git 命令行

先在 GitHub 创建一个空仓库，不勾选生成 README，以避免两个不同的初始提交。

```bash
git init
git add .
git status
git commit -m "Add intraday pattern research pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/intraday-pattern-research.git
git push -u origin main
```

将 `YOUR_USERNAME` 替换为自己的 GitHub 用户名。提交前查看 `git status`，确认只有预期的源码和文档。仓库默认忽略行情、模型与 PDF；项目包不包含真实行情或供应商账号。

当前没有代你创建或推送远程仓库。`.github` 中的检查会在上传后自动运行；本次本地通过不等于 GitHub 托管环境已经验证。

## 对外展示的表述

可写：构建股指期货日内形态研究项目，使用 K-Shape 生成训练期形态标签，融合 GRU 时序编码与 25 维截面因子，并提供日期划分、基准评价和可运行模拟演示。

当前真实行情上的预测表现尚待验证。不要把模拟演示的准确率写成市场预测效果。是否公开真实行情需依据供应商授权；是否开放代码复用取决于你选择的许可证。
