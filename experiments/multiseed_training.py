"""Train one seeded stage, committing a recoverable checkpoint after each epoch."""

import argparse
from contextlib import redirect_stdout
import os
from pathlib import Path
import platform
from time import monotonic

import numpy as np

from run_multiseed import MULTI_PLAN, saved_config
from run_search import FIRST_TARGET, ROOT, RUNS, TRAIN_END, VALID_END, digest, read_json, save_json
from search_training import (SearchRankLSTM, SearchRSR, export_embedding, graph, initialize,
                             metrics_for, session_config, tf)


def validation(session, model, inputs, prediction, losses):
    seq = model.parameters['seq']
    offsets = range(TRAIN_END - seq, VALID_END - seq)
    pred, target, mask = (np.empty((model.batch_size, len(offsets))) for _ in range(3))
    totals = np.zeros(3)
    for column, offset in enumerate(offsets):
        batch = model.get_batch(offset)
        value, *parts = session.run((prediction, *losses), dict(zip(inputs, batch)))
        pred[:, column], target[:, column], mask[:, column] = value[:, 0], batch[3][:, 0], batch[1][:, 0]
        totals += parts
    means = totals / len(offsets)
    if not np.isfinite(means).all() or not np.isfinite(pred).all():
        raise ValueError('Nonfinite validation predictions or loss')
    return means, (pred, target, mask), metrics_for(pred, target, mask)


def test_embedding(session, model, feature, tensor, folder, market):
    with np.load(RUNS / 'frozen-test' / f'{market}.npz', allow_pickle=False) as source:
        eod = source['eod']
    seq = model.parameters['seq']
    shape = (model.batch_size, eod.shape[1] - VALID_END, model.parameters['unit'])
    temporary = folder / 'test_embedding.tmp.npy'
    result = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=shape)
    for column, day in enumerate(range(VALID_END, eod.shape[1])):
        value = session.run(tensor, {feature: eod[:, day - seq:day, :]})
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite test embedding at day {day}')
        result[:, column, :] = value
    result.flush()
    del result
    temporary.replace(folder / 'test_embedding.npy')
    return list(shape)


def test_predictions(session, model, config, inputs, prediction):
    embedding = np.load(RUNS / config['embedding_run'] / 'test_embedding.npy', mmap_mode='r', allow_pickle=False)
    with np.load(RUNS / 'frozen-test' / f'{config["market"]}.npz', allow_pickle=False) as source:
        close = source['eod'][:, :, -1]
        masks, targets = source['mask'], source['targets']
        dates, tickers = source['dates'][VALID_END:], source['tickers']
    if embedding.shape != (model.batch_size, len(dates), config['unit']) or len(dates) != 237:
        raise ValueError('Test representation or period does not match the configuration')
    pred, gt, mask = (np.empty((model.batch_size, len(dates))) for _ in range(3))
    for column, day in enumerate(range(VALID_END, close.shape[1])):
        day_mask = np.min(masks[:, day - config['seq']:day + 1], axis=1)
        batch = (embedding[:, column, :], day_mask[:, None], close[:, day - 1, None], targets[:, day, None])
        value = session.run(prediction, dict(zip(inputs, batch)))
        pred[:, column], gt[:, column], mask[:, column] = value[:, 0], targets[:, day], day_mask
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite test predictions')
    return (pred, gt, mask), dates, tickers


def restore_log(path, offset):
    if not path.exists():
        if offset:
            raise ValueError('Committed training log is missing')
        return
    data = path.read_bytes()
    if len(data) < offset:
        raise ValueError('The training log is shorter than its committed checkpoint')
    if len(data) > offset:
        with path.with_name('uncommitted-log-tail.txt').open('ab') as archive:
            archive.write(data[offset:])
        with path.open('r+b') as output:
            output.truncate(offset)


def checkpoint_path(folder, relative):
    path = (folder / relative).resolve()
    if folder.resolve() not in path.parents:
        raise ValueError('Recovery checkpoint is outside the current run')
    return str(path)


def train(plan, run):
    config = plan['configurations'][run['configuration']]
    folder = RUNS / run['id']
    plan_hash = digest(MULTI_PLAN)
    if read_json(folder / 'config.json') != saved_config(plan, run):
        raise ValueError('The run configuration differs from the fixed five-seed plan')
    progress_path = folder / 'progress.json'
    progress = read_json(progress_path) if progress_path.exists() else {
        'completed_epochs': 0, 'log_offset': 0, 'best': None,
        'multiseed_plan_sha256': plan_hash,
    }
    if progress['multiseed_plan_sha256'] != plan_hash or not 0 <= progress['completed_epochs'] <= config['epochs']:
        raise ValueError('Recovery metadata differs from this run')
    is_embedding = config['stage'] == 'embedding'
    if not is_embedding:
        dependency = read_json(RUNS / config['embedding_run'] / 'result.json')
        if dependency['seed'] != run['seed'] or dependency['multiseed_plan_sha256'] != plan_hash:
            raise ValueError('The RSR stage must use the encoder from the same random seed')
    model = (SearchRankLSTM if is_embedding else SearchRSR)(config, run['seed'])
    inputs, (prediction, loss, regression, ranking), optimizer, embedding = graph(model, is_embedding)
    latest_saver = tf.train.Saver(max_to_keep=2)
    best_saver = tf.train.Saver(max_to_keep=2)
    for directory in ('latest', 'best'):
        (folder / directory).mkdir(exist_ok=True)
    offsets = np.arange(FIRST_TARGET - config['seq'], TRAIN_END - config['seq'])
    rng = np.random.RandomState(run['seed'])
    for _ in range(progress['completed_epochs']):
        rng.shuffle(offsets)
    best = progress['best']
    restore_log(folder / 'train.log', progress['log_offset'])
    with tf.Session(config=session_config()) as session:
        initialize(session, model, is_embedding)
        if progress['completed_epochs']:
            latest_saver.restore(session, checkpoint_path(folder, progress['latest_checkpoint']))
        with (folder / 'train.log').open('a', encoding='utf-8', buffering=1) as log:
            with redirect_stdout(log):
                print(f'Run {run["id"]}; seed={run["seed"]}; completed_epochs={progress["completed_epochs"]}')
                for epoch in range(progress['completed_epochs'], config['epochs']):
                    start = monotonic()
                    rng.shuffle(offsets)
                    totals = np.zeros(3)
                    for offset in offsets:
                        values = session.run((loss, regression, ranking, optimizer),
                                             dict(zip(inputs, model.get_batch(offset))))
                        totals += values[:3]
                    train_means = totals / len(offsets)
                    if not np.isfinite(train_means).all():
                        raise ValueError(f'Nonfinite training loss in epoch {epoch + 1}')
                    valid_means, _, metrics = validation(
                        session, model, inputs, prediction, (loss, regression, ranking))
                    print('Train Loss:', *train_means)
                    print('Valid MSE:', *valid_means)
                    print('Valid performance:', {'mse': metrics['mse'], 'mrrt': metrics['mrr'],
                                                 'btl': metrics['author_btl']})
                    if best is None or valid_means[0] < best['best_validation_loss']:
                        best = {'manifest_sha256': plan['base_plan_sha256'],
                                'multiseed_plan_sha256': plan_hash, 'id': run['id'], 'seed': run['seed'],
                                'best_epoch': epoch + 1, 'best_validation_loss': float(valid_means[0]),
                                'validation': metrics}
                        best_saver.save(session, str(folder / 'best' / 'epoch'),
                                        global_step=epoch + 1, write_meta_graph=False)
                        print('Better valid loss:', best['best_validation_loss'])
                    print(f'epoch: {epoch} time: {monotonic() - start:.4f}')
                    log.flush()
                    os.fsync(log.fileno())
                    latest = latest_saver.save(session, str(folder / 'latest' / 'epoch'),
                                                global_step=epoch + 1, write_meta_graph=False)
                    progress = {'multiseed_plan_sha256': plan_hash, 'completed_epochs': epoch + 1,
                                'best': best, 'log_offset': log.tell(),
                                'latest_checkpoint': str(Path(latest).relative_to(folder)),
                                'best_checkpoint': f'best/epoch-{best["best_epoch"]}'}
                    save_json(progress_path, progress)
        best_saver.restore(session, checkpoint_path(folder, progress['best_checkpoint']))
        restored_loss, valid_arrays, restored_metrics = validation(
            session, model, inputs, prediction, (loss, regression, ranking))
        np.testing.assert_allclose(restored_loss[0], best['best_validation_loss'], rtol=1e-5, atol=1e-10)
        best['validation'] = restored_metrics
        best['checkpoint'] = progress['best_checkpoint']
        if is_embedding:
            best['embedding_shape'] = export_embedding(session, embedding, inputs[0], model, folder)
            best['test_embedding_shape'] = test_embedding(session, model, inputs[0], embedding, folder, config['market'])
        else:
            arrays, dates, tickers = test_predictions(session, model, config, inputs, prediction)
            best['test'] = metrics_for(*arrays)
            np.savez_compressed(folder / 'predictions.npz',
                                **dict(zip(('valid_pred', 'valid_gt', 'valid_mask'), valid_arrays)),
                                **dict(zip(('test_pred', 'test_gt', 'test_mask'), arrays)),
                                dates=dates, tickers=tickers)
    save_json(folder / 'result.json', best)
    print(f'Completed {run["id"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    plan = read_json(MULTI_PLAN)
    if platform.python_version() != plan['python'] or tf.__version__ != plan['tensorflow']:
        raise RuntimeError('Python or TensorFlow differs from the frozen environment')
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('A registered GPU is required')
    for name, sha in plan['code_sha256'].items():
        if digest(ROOT / name) != sha:
            raise ValueError(f'Training code differs from the five-seed plan: {name}')
    run = next(run for run in plan['runs'] if run['id'] == args.run)
    train(plan, run)


if __name__ == '__main__':
    main()
