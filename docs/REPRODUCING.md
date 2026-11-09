# Reproduce and inspect the archived experiments

[中文](REPRODUCING.zh-CN.md) · [Project home](../README.md) · [Evidence index](README.md)

## 1. Choose what to restore

The Git checkout contains the analysis JSON, per-run configuration and logs, and figures. It is sufficient to read the results and run the local monitor. Use a separate recovery checkout for commands that regenerate reports, figures or data. The published run directories are records of completed jobs.

~~~bash
git clone https://github.com/jinglangqiu-hub/rsr-reproduction.git
cd rsr-reproduction
python tools/monitor_training.py
~~~

Open [the local monitor](http://127.0.0.1:8766/?view=parameters&batch=6). It reads the saved logs and binds to the loopback address. For original author data, initialize the upstream submodule and confirm its pinned commit.

~~~bash
git submodule update --init upstream
git -C upstream rev-parse HEAD
~~~

The expected commit is cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d. A legacy sparse checkout of upstream may need its data paths enabled with git -C upstream sparse-checkout disable. [Pinned author repository](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/tree/cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d)

## 2. Restore the GPU environment

The recorded platform is Windows 11 with WSL2 Ubuntu and an NVIDIA GeForce RTX 5070 Ti Laptop GPU. The Windows driver must expose /usr/lib/wsl/lib/libcuda.so.1 in WSL. In the WSL shell at the project root, use CPython 3.11.18 to create a virtual environment named rsr-gpu. The launcher scopes CUDA library paths to its child process.

~~~bash
python3.11 --version
python3.11 -m venv "$HOME/.venvs/rsr-gpu"
source "$HOME/.venvs/rsr-gpu/bin/activate"
python -m pip install -r environment/requirements.lock.txt
python -m pip check
python environment/configure_cuda_links.py
python environment/run_gpu.py environment/verify_gpu.py --output audit/local-gpu-verification.json
python environment/run_gpu.py compat/verify_forward.py --output audit/local-forward.json
~~~

The two checks use synthetic inputs and require GPU execution. The forward check compares the Rank_LSTM and RSR calculations against NumPy. Both write new reports at the supplied paths. [Locked requirements](../environment/requirements.lock.txt) · [Published forward audit](../audit/compat-forward.json)

## 3. Recover author inputs and check provenance

The full upstream checkout provides author CSVs and the relation archive. The pretraining download is approximately 570 MB and supplies the two fixed author embeddings. Unpacked inputs take several GB.

~~~bash
python compat/fetch_pretrain.py
python compat/verify_data.py
python compat/run_experiment.py --output audit/local-public-seed-preflight.json
~~~

The data verifier reconstructs compat/data and writes audit/author-data-verification.json in the recovery checkout. The preflight verifies frozen inputs and simulates the released sampler without optimizer updates. The recorded hash chain binds original files by byte content. [Expected input hashes and shapes](../audit/author-data-verification.json) · [Protocol A manifest](../experiments/manifest.json)

## 4. Recover recorded outputs and check metrics

Download the Release archive if you need saved predictions, checkpoints or evaluation caches. Check the SHA-256 and all ZIP members before extracting into the recovery checkout. Extraction restores same-named files under runs/.

~~~bash
gh release download v0.1.0-experiments --repo jinglangqiu-hub/rsr-reproduction \
  --pattern rsr-experiment-artifacts-v1.zip --dir artifacts
python tools/package_artifacts.py --verify artifacts/rsr-experiment-artifacts-v1.zip
python -m zipfile -e artifacts/rsr-experiment-artifacts-v1.zip .
~~~

Use NumPy 1.26.4 when recomputing the recorded metrics. The evaluator sorts without a stable tie rule, so a different NumPy version can change realized ranks when returns are tied.

The frozen-test audit uses archived predictions. The author-seed comparison JSON already contains the paper values. Rebuilding that comparison with the script’s main entry point additionally needs a local paper-text extraction at audit/paper.txt, which is outside the Git and Release inventories. The complete five-seed audit also checks the original large encoder embeddings, which require separate recovery. The scripts write reports to their recorded locations, so use the recovery checkout. [Release inventory and constraints](DATA_AND_ARTIFACTS.md) · [Metric audit code](../tools/analyze_multiseed.py)

## 5. Read or regenerate figures

The repository includes the original PNG and SVG outputs. The plotting environment uses NumPy 1.26.4 and Matplotlib 3.10.8. The figure scripts look for Microsoft YaHei on Windows or WSL. The public-seed and hyperparameter scripts also accept a font path with --font. Use the saved figures if the original font or prediction arrays are unavailable.

~~~bash
python3.11 -m venv "$HOME/.venvs/rsr-plot"
source "$HOME/.venvs/rsr-plot/bin/activate"
python -m pip install numpy==1.26.4 matplotlib==3.10.8
python tools/plot_multiseed.py
~~~

The five-seed figures read the committed analysis JSON. The public-seed and hyperparameter plots also need the corresponding prediction NPZ files from the Release. Plotting regenerates files and provenance metadata at the existing figure paths. [Figure index](README.md#figures--图表)

## 6. Historical training entry points

The historical order was Protocol A, one-factor search, combination verification, frozen inference and five-seed retraining. The existing queues contain completed run IDs and expected hashes. A new training attempt needs a dedicated output directory, explicit settings and new provenance records. The Windows wrapper for Protocol A contains a machine-specific WSL Python path. The verified Bash examples above use the active environment. [Experiment entry points](../experiments/README.md)
