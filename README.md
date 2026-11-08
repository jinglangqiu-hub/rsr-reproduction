# RSR 股票排序实验复现

论文 **Temporal Relational Ranking for Stock Prediction** 的本机 GPU 复现项目，包含 TensorFlow 兼容代码、实验编排、浏览器监测、超参数搜索、五随机种子评估及结果图表。

当前已完成 **174 次训练：48 次 Rank_LSTM 预训练 + 126 次 RSR 训练**。已覆盖 NASDAQ / NYSE × Wiki / Industry × 隐式 / 显式的 8 组 RSR 配置。原作者数值复现取得部分一致；后续搜索和五种子实验采用单独记录的训练边界与归一化协议，结论见 [现阶段结果](docs/EXPERIMENT_RESULTS.md)。

## 原始来源

- 作者仓库：[fulifeng/Temporal_Relational_Stock_Ranking](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking)
- 固定提交：`cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d`，通过 `upstream/` Git 子模块保留。
- [论文](https://arxiv.org/abs/1809.09441)、[引用信息](CITATION.bib)、[上游声明与改动说明](THIRD_PARTY_NOTICES.md)。

## 项目结构

| 路径 | 内容 |
| --- | --- |
| `upstream/` | 固定版本的原作者代码与数据来源，Git 子模块 |
| `compat/` | TensorFlow 2.21 的图模式兼容实现、原始数据整理与核验 |
| `environment/` | GPU 环境锁定依赖、CUDA 链接与 GPU 验证 |
| `experiments/` | 作者种子、单因素搜索、组合验证、冻结测试、五种子训练脚本 |
| `runs/` | 各次训练的配置、结果、逐轮日志、进度及全阶段计划/汇总 |
| `audit/` | 数据校验、训练前核验、论文结果对照等研究证据 |
| `tools/` | 浏览器监测、数值分析、绘图和归档工具 |
| `figures/` | 作者结果对照、超参数阶段、五种子阶段的 PNG / SVG 图表 |
| `docs/` | 环境恢复、数据与附件范围、实验结果说明 |

## 环境与使用

训练环境为 Windows 11 + WSL2 Ubuntu，Python **3.11.18**、TensorFlow **2.21.0**、NumPy **1.26.4**；实测 GPU 为 NVIDIA GeForce RTX 5070 Ti Laptop。依赖使用 TensorFlow 内部兼容接口，应按锁文件恢复版本。

```bash
git clone --recurse-submodules https://github.com/jinglangqiu-hub/rsr-reproduction.git
cd rsr-reproduction
```

请先阅读 [环境与数据恢复步骤](docs/REPRODUCING.md) 和 [数据及 Release 附件说明](docs/DATA_AND_ARTIFACTS.md)。仓库不会自动启动训练；既有实验清单绑定了文件 SHA-256，不能随意修改源码、清单或覆盖历史输出。

只查看已记录的训练过程时，Python 标准库即可运行监测服务：

```bash
python tools/monitor_training.py
```

浏览器打开 **http://127.0.0.1:8766/?view=parameters&batch=6**。页面显示市场、关系集、显式/隐式建模、序列长度 S、隐藏维度 U、排序权重 α、学习率、种子和训练阶段；指标提供完整中英文名称。

## 最终配置与五种子结果

共用设置：Adam 学习率 `0.001`，每次 `50` 轮，预测步长 `1`；Rank_LSTM 预训练排序权重为 `1`。每个种子同时重训两个阶段，共 30 次预训练、40 次关系训练。

| 市场 | 关系 | 建模 | S | U | α | 测试 IRR 均值 ± 样本标准差 | 正收益种子 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| NASDAQ | Wiki | 隐式 RSR_I | 8 | 64 | 0.1 | 54.54% ± 66.47% | 4/5 |
| NASDAQ | Wiki | 显式 RSR_E | 8 | 64 | 0.1 | 42.94% ± 45.40% | 4/5 |
| NASDAQ | Industry | 隐式 RSR_I | 16 | 64 | 10 | 22.04% ± 30.93% | 4/5 |
| NASDAQ | Industry | 显式 RSR_E | 16 | 16 | 0.1 | 24.87% ± 96.01% | 4/5 |
| NYSE | Wiki | 隐式 RSR_I | 16 | 32 | 10 | 117.58% ± 105.64% | 5/5 |
| NYSE | Wiki | 显式 RSR_E | 8 | 32 | 1 | 121.86% ± 68.55% | 5/5 |
| NYSE | Industry | 隐式 RSR_I | 4 | 32 | 10 | 109.16% ± 31.47% | 5/5 |
| NYSE | Industry | 显式 RSR_E | 16 | 32 | 10 | 105.59% ± 67.27% | 5/5 |

五个新种子：`175185289, 425827170, 1846380228, 1772992330, 20885008`。作者公开关系训练种子 `123456789` 用于前期复现与选参，不计入新五种子的统计。

**IRR：累计投资回报率 / Cumulative Investment Return Ratio**，按作者评估器对每日首选股票的简单收益率求和，不是复利净值。误差范围为五次运行的样本标准差（`ddof=1`），不是置信区间。

![五种子测试结果](figures/multiseed/01-test-metrics-mean-std.png)

需结合种子波动与事件敏感性解读收益。NYSE Industry 隐式模型的两个固定交易日合计贡献 126.61 个百分点；剔除这两天后，平均 IRR 为 −17.45%。详见 [完整分析及复现边界](docs/EXPERIMENT_RESULTS.md)。

## 数据与产物保存

Git 保存源码、轻量实验记录与图表。逐股预测、TensorFlow 检查点、评估缓存另存于 [Release 附件](https://github.com/jinglangqiu-hub/rsr-reproduction/releases/tag/v0.1.0-experiments)，逐文件 SHA-256 见 [附件清单](docs/artifact-manifest.json)。原始行情、关系图、作者预训练数据由固定上游版本恢复；运行产生的大型嵌入不随附件上传。具体恢复范围和限制见 [说明](docs/DATA_AND_ARTIFACTS.md)。

## 许可证

沿用原作者仓库的 **GNU AGPL v3** 许可证，完整文本见 [LICENSE](LICENSE)。本项目包含对上游实现的适配与实验工具，原作者及论文出处见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
