# RSR 超参数搜索与冻结测试图表

图表分为实验结果和分析结果两组，共 10 张。每张提供 240 DPI PNG 和矢量 SVG。坐标和图例注明完整中英文指标名称，所有结果来自固定种子 123456789。

## 实验结果

| 编号 | 内容 | PNG | SVG |
|---|---|---|---|
| E01 | 序列长度 S 的独立搜索 | [查看](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/experimental-results/E01-sequence-length-search.png) | [矢量图](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/experimental-results/E01-sequence-length-search.svg) |
| E02 | 隐藏维度 U 的独立搜索 | [查看](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/experimental-results/E02-hidden-dimension-search.png) | [矢量图](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/experimental-results/E02-hidden-dimension-search.svg) |
| E03 | 排序权重 α 的独立搜索 | [查看](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/experimental-results/E03-ranking-weight-search.png) | [矢量图](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/experimental-results/E03-ranking-weight-search.svg) |
| E04 | 冻结配置的测试集 MSE、MRR、IRR | [查看](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/experimental-results/E04-frozen-test-metrics.png) | [矢量图](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/experimental-results/E04-frozen-test-metrics.svg) |
| E05 | 测试期逐日累计 IRR | [查看](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/experimental-results/E05-test-cumulative-irr.png) | [矢量图](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/experimental-results/E05-test-cumulative-irr.svg) |

## 分析结果

| 编号 | 内容 | PNG | SVG |
|---|---|---|---|
| A01 | 验证期与测试期 IRR 对照 | [查看](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/analysis-results/A01-validation-test-irr.png) | [矢量图](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/analysis-results/A01-validation-test-irr.svg) |
| A02 | 6 组新组合与原最优配置对照 | [查看](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/analysis-results/A02-combination-comparison.png) | [矢量图](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/analysis-results/A02-combination-comparison.svg) |
| A03 | 剔除最佳 3 日的收益集中度分析 | [查看](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/analysis-results/A03-return-concentration.png) | [矢量图](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/analysis-results/A03-return-concentration.svg) |
| A04 | 检查点选择规则敏感性 | [查看](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/analysis-results/A04-checkpoint-rule-sensitivity.png) | [矢量图](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/analysis-results/A04-checkpoint-rule-sensitivity.svg) |
| A05 | 三项单因素搜索增益及优胜值 | [查看](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/analysis-results/A05-independent-search-gains.png) | [矢量图](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/analysis-results/A05-independent-search-gains.svg) |

## 复现绘图

在项目目录使用已有独立绘图环境运行。

```powershell
wsl -d Ubuntu --exec /home/luckythrower/.venvs/rsr-plot/bin/python tools/plot_hparam_stage.py
```

[绘图脚本](C:/Users/luckythrower/Desktop/RSR/tools/plot_hparam_stage.py) 使用 NumPy 1.26.4 和 Matplotlib 3.10.8。[来源与图表清单](C:/Users/luckythrower/Desktop/RSR/figures/hparam-stage/figure-manifest.json) 记录输入文件与脚本的 SHA-256。

验证期覆盖 2016 年 252 个交易日，测试期覆盖 2017-01-03 至 2017-12-08 的 237 个交易日。IRR 按每日首选股票的简单收益率加和。E01–E03 固定另外两个参数为各市场基线，A04 的日志峰值仅用于规则敏感性分析，A03 的剔除日期依据各模型自身最佳 3 日。全部图表使用本阶段实际运行数据，误差范围需要后续多种子实验估计。
