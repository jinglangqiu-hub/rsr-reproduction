"""Run a Python script with CUDA library paths scoped to the child process."""
import os
from pathlib import Path
import sys
import sysconfig

prefix = Path(sys.prefix).resolve()
if prefix.name != 'rsr-gpu' or sys.prefix == sys.base_prefix or len(sys.argv) < 2:
    raise SystemExit('Usage: <rsr-gpu Python> environment/run_gpu.py <script> [args]')
nvidia = Path(sysconfig.get_path('platlib')) / 'nvidia'
libraries = sorted(str(path) for path in nvidia.glob('*/lib') if path.is_dir())
driver = Path('/usr/lib/wsl/lib')
if not (driver / 'libcuda.so.1').is_file() or not libraries:
    raise SystemExit('WSL GPU driver or environment NVIDIA libraries were not found')
env = dict(os.environ)
env['LD_LIBRARY_PATH'] = os.pathsep.join([str(driver), *libraries])
env['PATH'] = os.pathsep.join([str(prefix / 'bin'), env.get('PATH', '')])
os.execve(sys.executable, [sys.executable, *sys.argv[1:]], env)
