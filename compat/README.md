# RSR 兼容副本与训练前核验

`compat/training` 从作者提交移植而来，固定版本为 `cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d`。完整恢复后的 `compat/data` 包含股票名单、日期表、名单对应的 2,763 个行情文件、四个关系矩阵以及两份公开预训练嵌入。Git 检出包含名单与日期表，其余按指南恢复。 [恢复步骤](../docs/REPRODUCING.zh-CN.md) · [数据核验](../audit/author-data-verification.json)

适配使用 TensorFlow 2.21.0 内附的旧版 `BasicLSTMCell` 和 `dense`，保留图模式、原来的损失、关系 softmax 轴及协议 A 使用的公开 RSR 种子 `123456789`。依赖内部接口，因此运行时使用 `environment/requirements.lock.txt` 中的固定版本。测试将 TF32 关闭以匹配 FP32 数值对照。原仓库仍可单独用于核对差异。

NYSE 行业关系的 `int64` 文件约 2.6 GB。兼容副本以内存映射读取关系数据，初始化时一次转成原模型使用的 FP32 并上传到 GPU 中的非训练变量。关系矩阵不再序列化为图常量，也不纳入模型参数或检查点。

在 WSL 的工作目录 `RSR` 中使用以下命令验证。每次只计算前向和损失。测试使用 GPU，禁止静默回退至 CPU。

```bash
source "$HOME/.venvs/rsr-gpu/bin/activate"
python environment/run_gpu.py compat/verify_forward.py --output audit/local-compat-forward.json
python environment/run_gpu.py compat/verify_forward.py --market NASDAQ --output audit/local-compat-forward-NASDAQ.json
python environment/run_gpu.py compat/verify_forward.py --market NYSE --output audit/local-compat-forward-NYSE.json
```

合成检查以同一组已初始化的权重，通过 NumPy 对照 LSTM 门控、RSR 两个分支的关系传播、预测、回归损失和成对排序损失。作者数据检查使用 README 指定的市场、关系类型和嵌入，取验证集第一天，覆盖全部股票。

可通过 `compat/fetch_pretrain.py` 从作者公开链接恢复下载，再用 `compat/verify_data.py` 整理和核验文件。预训练归档只有 NASDAQ `seq-16/unit-64` 与 NYSE `seq-8/unit-32` 两份 `float32` 数组，文件名后缀的含义未随归档发布。同一市场的嵌入可以作为不同关系类型的固定输入，但其他序列长度、隐藏维度或重新初始化后的嵌入需要另行生成并记录来源。

两个市场均使用 1,245 个交易日，按 756／252／237 日划分。关系文件的行业通道数为 NASDAQ 97、NYSE 108，包含自环通道。公开预处理器只编码至少包含两只股票的行业，并另加自环，因此通道数不能直接与论文表 4 的行业类型统计等同。

协议 A 沿用作者公开采样，从 `0..755` 打乱偏移再截取，标签日期取 `offset + seq`。按 README 配置，可抽到 NASDAQ 的前 16 个验证日和 NYSE 的前 8 个验证日。协议 B 限制训练标签在训练期内并使用训练期归一化。 [作者采样预检](../audit/experiment-preflight.json) · [独立协议](../docs/EXPERIMENT_RESULTS.md)
