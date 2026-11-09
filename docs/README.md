# Evidence and figure index · 证据与图表索引

[Project home](../README.md) · [中文首页](../README.zh-CN.md) · [Research Note](RESEARCH_NOTE.md) · [中文研究笔记](RESEARCH_NOTE.zh-CN.md)

| Evidence layer | What it supports | Sources |
| --- | --- | --- |
| Paper · 论文 | Model definition, dataset description and reported five-run means | [arXiv v2, §§3–5 and Tables 3, 6, 7](https://arxiv.org/abs/1809.09441v2) · [Journal citation](https://doi.org/10.1145/3309547) |
| Released implementation · 作者代码 | Actual scoring, normalization axis, sampling and evaluator | [Pinned commit](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/tree/cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d) · [Source fingerprints](../audit/source-manifest.json) |
| Compatibility port · 兼容实现 | TensorFlow 2.21 port and forward checks | [Port notes](../compat/README.md) · [Synthetic check](../audit/compat-forward.json) · [Author data verification](../audit/author-data-verification.json) |
| A · 作者公开流程 | Eight local runs against paper tables, with settings provenance | [Run manifest](../experiments/manifest.json) · [Sampling preflight](../audit/experiment-preflight.json) · [Metric comparison](../audit/public-seed-comparison.json) |
| B · 独立协议 | Search, frozen selection, test results and seed sensitivity | [Independent search](../runs/hparam-search-summary.json) · [Combination trial](../runs/hparam-combination-summary.json) · [Frozen settings](../runs/frozen-test-manifest.json) · [Frozen test](../runs/frozen-test/analysis.json) · [Five-seed audit](../runs/multiseed-analysis.json) |
| Interpretation · 分析与推论 | Scope of reproduction, mathematical reading and limitations | [English note](RESEARCH_NOTE.md) · [中文笔记](RESEARCH_NOTE.zh-CN.md) · [完整实验结果](EXPERIMENT_RESULTS.md) |

## Figures · 图表

| Stage | Plots | Provenance |
| --- | --- | --- |
| A · Paper comparison | [Test metrics](../figures/public-seed/01-test-metrics.png) · [IRR ordering](../figures/public-seed/02-irr-order.png) · [NYSE Industry return path](../figures/public-seed/03-nyse-industry-daily-irr.png) | [Plotting script](../tools/plot_public_seed.py) · [Comparison JSON](../audit/public-seed-comparison.json) |
| B · Search and frozen test | [Ten figures, PNG and SVG](../figures/hparam-stage/README.md) | [Figure manifest](../figures/hparam-stage/figure-manifest.json) |
| B · Five seeds | [Metrics and dispersion](../figures/multiseed/01-test-metrics-mean-std.png) · [IRR by seed](../figures/multiseed/02-test-irr-by-seed.png) · [Two-date sensitivity](../figures/multiseed/03-event-day-sensitivity.png) | [Figure manifest](../figures/multiseed/figure-manifest.json) |

MSE denotes mean squared error on returns, MRR mean reciprocal rank of the daily top pick, and IRR the cumulative investment return ratio (additive daily simple returns). Figure spreads for the five new seeds are sample standard deviations. IRR displayed as a percentage uses a factor of 100. Differences between percentages are in percentage points.

[Environment and recovery](REPRODUCING.md) · [中文恢复步骤](REPRODUCING.zh-CN.md) · [Release inventory](DATA_AND_ARTIFACTS.md) · [Experiment entry points](../experiments/README.md)
