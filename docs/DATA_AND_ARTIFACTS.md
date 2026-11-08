# 数据和产物的保存范围

## Git 内保存

- 兼容训练源码、环境依赖锁文件、实验编排、暂停恢复、监测和分析绘图工具。
- 作者固定版本的 Git 子模块、原始许可证、引用及修改说明。
- 174 次完成训练相关的配置、结果、逐轮日志、进度和全阶段计划/汇总。
- 研究核验 JSON、原作者数值对照、32 个 PNG / SVG 图表文件，以及原有完整流程图。
- 两市场原始股票名单与日期表。

`audit/` 仅选择研究相关记录；本机软件清单、安装日志、下载片段、旧副本及凭据不纳入仓库。历史日志中保留了部分实验路径和硬件信息，它们用于理解当时的运行环境。

## Release 附件保存

版本：[v0.1.0-experiments](https://github.com/jinglangqiu-hub/rsr-reproduction/releases/tag/v0.1.0-experiments)

附件 `rsr-experiment-artifacts-v1.zip` 保存 `runs/` 下除 `.npy` 嵌入、锁文件和临时文件以外的产物，包括：

- 全部已保存的 134 份预测 NPZ（含不同阶段的验证/测试预测）。
- TensorFlow 检查点的索引、参数、元图及检查点指针，保留已有 best / latest 目录。
- 训练/验证搜索缓存以及冻结测试数据缓存，共 4 份 NPZ。
- 同一时点的配置、日志、进度、计划与汇总，便于附件独立核对。

逐文件原始大小、SHA-256 及压缩包 SHA-256 记录于 `docs/artifact-manifest.json`；下载后可验证附件传输完整性。解压位置为项目根目录，归档内路径均为相对 `runs/...`，会恢复同名文件。

```bash
gh release download v0.1.0-experiments --repo jinglangqiu-hub/rsr-reproduction \
  --pattern rsr-experiment-artifacts-v1.zip --dir artifacts
python tools/package_artifacts.py --verify artifacts/rsr-experiment-artifacts-v1.zip
```

确认目标目录内容后再解压，避免覆盖新增实验的同名结果。Python 标准库示例：

```bash
python -m zipfile -e artifacts/rsr-experiment-artifacts-v1.zip .
```

该附件适合结果再分析和权重留存；不包含所有训练依赖，不能据此宣称恢复了全量运行目录。

## 本地保留、未上传

- `compat/data` 中的逐股行情、稠密关系图和作者预训练数组：从作者固定版本/公开归档恢复，步骤见 `REPRODUCING.md`。
- `runs/` 中约 13 GB 的 `.npy` 数组，包括各阶段编码器嵌入：属于可重新生成的大型中间产物。本地原文件完整保留。
- `downloads/` 下载缓存、`original/` 早期重复副本、论文 PDF、本机安装和软件清单。

因此，`tools/analyze_multiseed.py` 中验证编码器数组形状的完整检查，在其他机器仍需补齐嵌入；报告记录的是原机已完成的核验。已有 JSON 和图表可直接阅读，预测 NPZ 可用于独立重算 MSE、MRR 和 IRR。

## 重新制作附件

`tools/package_artifacts.py` 只读训练产物，生成 ZIP 和清单，不加载模型、不执行训练。默认拒绝覆盖已有附件：

```bash
python tools/package_artifacts.py --build
```

重新归档属于新的项目快照，应使用新版本附件和对应清单，避免把历史 Release 与后续结果混合。
