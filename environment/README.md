# RSR 独立运行环境

WSL2 Ubuntu 内的环境：`/home/luckythrower/.venvs/rsr-gpu`。
Windows 的 Conda base 和 gcn-repro 不用于安装这些包。rsr-original 已按用户要求删除；此前的包清单仍保存在 audit 中。

在 PowerShell 中打开环境：

```powershell
wsl -d Ubuntu
```

进入 Linux 后：

```bash
source /home/luckythrower/.venvs/rsr-gpu/bin/activate
cd /mnt/c/Users/luckythrower/Desktop/RSR
python environment/run_gpu.py environment/verify_gpu.py --output audit/rsr-gpu-verification.json
```

验证程序仅使用合成常量，不执行训练或参数更新；显式禁止 CPU 回退。eager 和 compat.v1 Session 均已在 RTX 5070 Ti Laptop 上通过，证据见 audit/rsr-gpu-verification.json。

安装输入在 requirements.in，实际安装版本已导出到 requirements.lock.txt，pip check 通过。重建时使用 CPython 3.11.18，在同名 venv 中从锁文件安装，再运行 `python environment/configure_cuda_links.py` 和上面的 GPU 验收。链接用于让 TensorFlow 和工具找到环境内的 CUDA 文件；本机还需要通过 run_gpu.py 设置子进程的 CUDA 库搜索路径，不修改系统变量。

TensorFlow 2.21.0 的 wheel 未预编译 sm_120，首遇算子时由驱动从 PTX 即时编译。合成测试已实测通过；完整模型的算子和显存需求仍需单独检查。

作者旧代码包含 tf.contrib、tf.layers 等接口，此环境不能直接保证原模型可运行。后续需在独立工作副本完成兼容迁移及数值核对，upstream 保持原始提交。本环境的建立不代表论文结果已复现。
