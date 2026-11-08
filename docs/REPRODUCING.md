# 环境恢复与安全入口

本文描述恢复方式，不会触发训练。原训练脚本、实验清单及历史结果保持原字节，以保留已有 SHA-256 链。部分旧目录 README 记录的是最初建立环境时的状态，当前阶段以根目录 README 及 `docs/EXPERIMENT_RESULTS.md` 为准。

## 1. 获取源码和作者版本

```bash
git clone --recurse-submodules https://github.com/jinglangqiu-hub/rsr-reproduction.git
cd rsr-reproduction
git -C upstream rev-parse HEAD
```

上游提交应为 `cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d`。如果本机原有的上游目录使用稀疏检出，恢复数据前先执行 `git -C upstream sparse-checkout disable`；本仓库正常递归克隆无需该步骤。

## 2. 恢复 GPU 环境

已验证平台：Windows 11、WSL2 Ubuntu、NVIDIA GeForce RTX 5070 Ti Laptop。需要 Windows NVIDIA 驱动向 WSL 提供 `/usr/lib/wsl/lib/libcuda.so.1`。使用 CPython 3.11.18 建立名为 `rsr-gpu` 的虚拟环境：

```bash
python3.11 -m venv "$HOME/.venvs/rsr-gpu"
source "$HOME/.venvs/rsr-gpu/bin/activate"
python -m pip install -r environment/requirements.lock.txt
python -m pip check
python environment/configure_cuda_links.py
python environment/run_gpu.py environment/verify_gpu.py --output audit/local-gpu-verification.json
```

其中 `python3.11` 必须实际为 3.11.18；Python 的安装由本机环境负责。验证仅使用合成常量，禁止静默回退 CPU，不执行训练。`run_gpu.py` 为子进程设置 CUDA 库路径，不修改系统环境变量。

## 3. 恢复原始实验数据

完整上游检出包含行情 CSV、名单、日期和 `data/relation.tar.gz`。兼容整理程序会核验并复制所需数据：

```bash
python compat/fetch_pretrain.py
python compat/verify_data.py
```

第一条下载作者公开预训练归档（约 570 MB）；第二条解压关系与预训练数据，整理行情并核验。展开后的关系矩阵和嵌入需数 GB 空间。`verify_data.py` 会重写 `audit/author-data-verification.json`，因此执行前可另行保存已发布记录用于对照。

作者公开归档只包含 NASDAQ `(S=16,U=64)`、NYSE `(S=8,U=32)` 两套嵌入；其他设置由本项目训练产生。下载地址及归档散列已写在 `compat/fetch_pretrain.py` 与 `audit/author-data-verification.json`。

前向数值核验不执行参数更新：

```bash
python environment/run_gpu.py compat/verify_forward.py --output audit/local-forward.json
```

## 4. 恢复已有结果与附件

Git 已包含配置、日志和统计。需要逐股预测、权重或评估缓存时，按 `docs/DATA_AND_ARTIFACTS.md` 下载 Release 附件、校验后解压。附件不含大型 `embedding.npy`、`test_embedding.npy`，完整训练流程/全部审计如需这些文件，需要在原机保留的产物中取回，或在独立输出目录重新生成。

原有清单中记录了 `/home/luckythrower/.venvs/rsr-gpu/bin/python` 等本机运行元数据；GPU 启动器主要基于当前环境定位库。不要直接修改已冻结清单来掩盖新机器差异；新的训练应生成新的计划、哈希和输出记录。

## 5. 查看记录与图表

监测服务仅依赖 Python 标准库，Windows 或 Linux 均可运行：

```bash
python tools/monitor_training.py
```

打开 `http://127.0.0.1:8766/?view=parameters&batch=6`。默认绑定回环地址，读取已保存的运行记录。

绘图采用独立环境，已记录版本为 NumPy 1.26.4、Matplotlib 3.10.8：

```bash
python3.11 -m venv "$HOME/.venvs/rsr-plot"
source "$HOME/.venvs/rsr-plot/bin/activate"
python -m pip install numpy==1.26.4 matplotlib==3.10.8
python tools/plot_multiseed.py
```

五种子图表直接读取已提交的分析 JSON。其他绘图脚本部分需要 Release 中的逐股预测和 `compat/data` 的股票日期表。中文字体默认从本机 Windows / WSL 的 Microsoft YaHei 路径查找；另两阶段脚本也可通过 `--font` 指定字体。字体文件不随项目分发。

## 6. 训练入口说明

训练入口依次为 `experiments/run_confirmed_batch.py`、`run_search.py`、`run_combination.py`、`evaluate_frozen.py` 和 `run_multiseed.py`；具体协议由对应已冻结 JSON 清单定义。`evaluate_frozen.py` 仅推理，其余入口带明确执行参数才训练。

历史 `runs/` 已保存完成标记，且大数组不全部随 Git 分发；不能把已有结果目录当作干净队列直接重跑。后续新一轮训练前应列清配置、种子、协议、预计训练数，取得实验负责人确认，再使用独立输出和新计划。当前上传操作不启动任何训练。
