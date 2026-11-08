"""Expose pip-installed NVIDIA libraries to TensorFlow inside rsr-gpu only."""
import os
from pathlib import Path
import sys
import sysconfig

prefix = Path(sys.prefix).resolve()
if prefix.name != 'rsr-gpu' or sys.prefix == sys.base_prefix:
    raise RuntimeError(f'Refusing to configure anything outside rsr-gpu: {prefix}')
site = Path(sysconfig.get_path('platlib')).resolve()
if not site.is_relative_to(prefix):
    raise RuntimeError(f'Unexpected site-packages path: {site}')
nvidia = site / 'nvidia'
tensorflow = site / 'tensorflow'
if not tensorflow.is_dir():
    raise RuntimeError('Install TensorFlow before configuring CUDA library links')


def link(target, destination):
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != target.resolve():
            raise RuntimeError(f'Unexpected existing file: {destination}')
    else:
        destination.symlink_to(os.path.relpath(target, destination.parent))


libraries = sorted(nvidia.glob('*/lib/*.so*'))
if not libraries:
    raise RuntimeError('No NVIDIA pip libraries found')
for library in libraries:
    link(library, tensorflow / library.name)
ptxas = nvidia / 'cuda_nvcc/bin/ptxas'
if not ptxas.is_file():
    raise RuntimeError('ptxas not found in NVIDIA pip packages')
link(ptxas, prefix / 'bin/ptxas')
print(f'Linked {len(libraries)} NVIDIA libraries and ptxas within {prefix}')
