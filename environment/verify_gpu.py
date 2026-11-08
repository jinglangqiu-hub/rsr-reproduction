"""No training: validate eager and TF1-style GPU execution on synthetic constants."""
import argparse
import importlib.metadata as metadata
import json
import platform
from pathlib import Path

import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise RuntimeError('GPU not registered; CPU fallback is forbidden')
tf.config.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_soft_device_placement(False)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

with tf.device('/GPU:0'):
    a = tf.constant([[1., 2.], [3., 4.]], dtype=tf.float32)
    product = tf.matmul(a, a)
    weights = tf.nn.softmax(product, axis=0)
    activated = tf.nn.leaky_relu(product - 8., alpha=0.2)
    reduced = tf.reduce_mean(activated)
np.testing.assert_allclose(product.numpy(), [[7., 10.], [15., 22.]])
np.testing.assert_allclose(weights.numpy().sum(axis=0), [1., 1.], atol=1e-6)
np.testing.assert_allclose(reduced.numpy(), 5.7, atol=1e-6)
for tensor in [product, weights, activated, reduced]:
    assert 'GPU:0' in tensor.device, tensor.device

# TF1-style graph with explicitly placed ops; no variables or optimizer.
graph = tf.Graph()
with graph.as_default(), tf.device('/GPU:0'):
    features = tf.compat.v1.placeholder(tf.float32, [4, 2], name='features')
    head = tf.matmul(features, features, transpose_b=True)
    adjacency = tf.nn.softmax(tf.nn.leaky_relu(head, alpha=0.2), axis=0)
    propagated = tf.matmul(adjacency, features)
    loss = tf.reduce_mean(tf.square(propagated))
config = tf.compat.v1.ConfigProto(allow_soft_placement=False,
                                intra_op_parallelism_threads=2,
                                inter_op_parallelism_threads=2)
config.gpu_options.allow_growth = True
sample = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
run_metadata = tf.compat.v1.RunMetadata()
with tf.compat.v1.Session(graph=graph, config=config) as session:
    graph_result, loss_result = session.run([propagated, loss], {features: sample},
                                           options=options, run_metadata=run_metadata)
assert np.isfinite(graph_result).all() and np.isfinite(loss_result)
executed_gpu = [d.device for d in run_metadata.step_stats.dev_stats
                if 'GPU:0' in d.device and len(d.node_stats) > 0]
assert executed_gpu, 'Session did not execute any GPU nodes'
names = ['tensorflow', 'numpy', 'scipy', 'scikit-learn', 'keras',
         'nvidia-cublas-cu12', 'nvidia-cuda-runtime-cu12', 'nvidia-cudnn-cu12']
report = {
    'python': platform.python_version(), 'platform': platform.platform(),
    'packages': {name: metadata.version(name) for name in names},
    'build': tf.sysconfig.get_build_info(),
    'gpu': str(gpus[0]),
    'gpu_details': tf.config.experimental.get_device_details(gpus[0]),
    'eager_device': product.device, 'eager_matmul': product.numpy().tolist(),
    'softmax_passed': True, 'elementwise_reduction_passed': True,
    'session_gpu_execution': executed_gpu,
    'session_output': graph_result.tolist(), 'session_loss': float(loss_result),
    'cpu_fallback_allowed': False, 'optimizer_steps': 0,
    'original_model_verified': False,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
print(json.dumps(report, indent=2, default=str))
