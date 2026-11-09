# RSR GPU 运行环境

[English setup](../docs/REPRODUCING.md) · [中文恢复步骤](../docs/REPRODUCING.zh-CN.md) · [兼容实现](../compat/README.md)

归档实验在 Windows 11 + WSL2 Ubuntu 和 NVIDIA GeForce RTX 5070 Ti Laptop 上运行，使用 CPython 3.11.18、TensorFlow 2.21.0 及独立虚拟环境。安装输入与实际版本分别保存在 [requirements.in](requirements.in) 和 [requirements.lock.txt](requirements.lock.txt)。

虚拟环境名称为 **rsr-gpu**。按恢复指南安装锁文件后，运行 [configure_cuda_links.py](configure_cuda_links.py) 配置环境内的 CUDA 链接。 [run_gpu.py](run_gpu.py) 仅为子进程指定库搜索路径，需要 Windows NVIDIA 驱动在 WSL 内提供 libcuda.so.1。

在 WSL 中激活环境后，从仓库根目录运行合成验证。

~~~bash
source "$HOME/.venvs/rsr-gpu/bin/activate"
python environment/run_gpu.py environment/verify_gpu.py --output audit/local-gpu-verification.json
python environment/run_gpu.py compat/verify_forward.py --output audit/local-forward.json
~~~

首个程序验证 eager 和 compat.v1 Session 均在 GPU 执行。第二个程序对照 NumPy 检查 LSTM、两种关系分支和损失。原机另核验了作者数据上的前向运行。上述归档实验已使用兼容实现完成，历史检查和实验记录的适用范围见 [合成前向记录](../audit/compat-forward.json)、[NASDAQ 前向记录](../audit/compat-forward-NASDAQ.json)、[NYSE 前向记录](../audit/compat-forward-NYSE.json) 和 [实验结果](../docs/EXPERIMENT_RESULTS.md)。
