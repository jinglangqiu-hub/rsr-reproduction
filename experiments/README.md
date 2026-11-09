# RSR 实验入口与冻结清单

[项目首页](../README.zh-CN.md) · [证据索引](../docs/README.md) · [环境与数据恢复](../docs/REPRODUCING.zh-CN.md)

## 协议 A · 作者公开种子流程

本轮清单覆盖两个市场、两套关系数据和两种时序图卷积建模。关系阶段使用作者公开嵌入和种子 `123456789`。Adam 学习率为 0.001，预测步长为 1，训练 50 轮。

| 市场 | 关系 | 建模 | 序列长度 | 隐藏维度 | α | 参数依据 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| NASDAQ | Wiki | RSR_I 隐式 | 16 | 64 | 0.1 | 作者 README |
| NYSE | Industry | RSR_I 隐式 | 8 | 32 | 10 | 作者 README |
| NASDAQ | Wiki | RSR_E 显式 | 16 | 64 | 0.1 | 沿用同市场 README 参数 |
| NASDAQ | Industry | RSR_I 隐式 | 16 | 64 | 0.1 | 沿用同市场 README 参数 |
| NASDAQ | Industry | RSR_E 显式 | 16 | 64 | 0.1 | 沿用同市场 README 参数 |
| NYSE | Wiki | RSR_I 隐式 | 8 | 32 | 10 | 沿用同市场 README 参数 |
| NYSE | Wiki | RSR_E 显式 | 8 | 32 | 10 | 沿用同市场 README 参数 |
| NYSE | Industry | RSR_E 显式 | 8 | 32 | 10 | 沿用同市场 README 参数 |

`manifest.json` 固定源码、依赖、输入文件的 SHA-256，以及八个公开种子运行 ID。`parameter_provenance` 将两条作者 README 命令与六条沿用同市场参数的配置区分开。论文报告了超参数搜索范围和汇总结果，公开材料未给出这六条各自的最终设置。论文主结果的 22 个配置都保留在 `paper_grid`。

种子只作用于关系训练，嵌入保持固定。论文表格汇总了五次不同运行。作者尚未公开对应的完整种子清单。协议 B 已用五个新种子重训，统计单独保存。 [五种子计划](../runs/multiseed-plan.json)

窗口采样沿用发布代码。每轮将起点 `0–755` 原地打乱，再截取前 `756−S` 个。按公开种子模拟 50 轮，NASDAQ 和 NYSE 分别有 781 和 393 次更新会抽到验证期标签。这是发布代码的边界行为，保留在预检记录中。

在 WSL 的 RSR 工作目录运行预检。预检重新核对文件哈希并模拟每个 epoch 的采样，默认没有优化器更新。

```bash
source "$HOME/.venvs/rsr-gpu/bin/activate"
python compat/run_experiment.py --output audit/local-public-seed-preflight.json
```

Windows 启动脚本默认展示配置和执行预检，其 WSL Python 路径指向原机环境。新环境可使用上面的 Bash 入口。若在单独的实验目录启动新训练，每次指定一个运行 ID。脚本会先显示参数，输入 `START` 后才启动该条。沿用同市场参数的六条还要求显式传入 `-AllowMarketTemplate`，表明同意采用清单所列的参数。

```powershell
.\experiments\run_public_seed.ps1
.\experiments\run_public_seed.ps1 -RunId nasdaq-wikidata-rsri-public-seed-123456789 -Execute
```

运行记录存入 `runs/<运行 ID>`。每次保留冻结配置、训练日志、最低验证总损失对应的检查点和预测矩阵。结果报告 MSE、MRR 和 IRR，保留作者原始 `btl` 并按 `btl - 1` 转成论文口径。运行目录具有排他性，重复执行同一 ID 会拒绝覆盖。

## 协议 B · 独立搜索、冻结测试与五种子评估

| 阶段 | 入口 | 归档记录 |
| --- | --- | --- |
| 单因素搜索（批次 1–4） | [run_search.py](run_search.py) | [计划](../runs/hparam-search-plan.json) · [汇总](../runs/hparam-search-summary.json) |
| 组合验证（批次 5） | [run_combination.py](run_combination.py) | [计划](../runs/hparam-combination-plan.json) · [汇总](../runs/hparam-combination-summary.json) |
| 冻结测试（推理） | [evaluate_frozen.py](evaluate_frozen.py) | [清单](../runs/frozen-test-manifest.json) · [测试分析](../runs/frozen-test/analysis.json) |
| 五种子重训（批次 6） | [run_multiseed.py](run_multiseed.py) | [计划](../runs/multiseed-plan.json) · [逐种子分析](../runs/multiseed-analysis.json) |

历史运行目录与完成标记供核查。新实验使用独立输出目录和新的清单记录。 [选择规则与覆盖范围](../docs/EXPERIMENT_RESULTS.md)

## 浏览器训练观察

在项目根目录使用 Python 启动只读监测服务。页面从运行配置显示时序图卷积的显式或隐式建模，以及行业关系或 Wikidata 企业关系。服务每 2 秒刷新训练记录。可切换运行、选中历史轮次或暂停刷新。 [本地训练监测](http://127.0.0.1:8766/?view=parameters&batch=6)

```powershell
python tools/monitor_training.py
```

如果端口已被本项目的监控服务使用，直接打开页面即可。从终端启动的服务可用 Ctrl+C 关闭。
