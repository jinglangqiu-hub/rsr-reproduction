"""Freeze validation-selected RSR checkpoints and evaluate them on 2017 prices."""

import argparse
import fcntl
import hashlib
import io
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from run_search import (CACHE, PLAN, ROOT, RUNS, SOURCE, TRAIN_END, VALID_END,
                        author_manifest, digest, read_json, save_json)


FROZEN = RUNS / 'frozen-test-manifest.json'
OUTPUT = RUNS / 'frozen-test'
END = 1245


def fingerprint_checkpoint(run_id):
    folder = RUNS / run_id
    return {str(path.relative_to(ROOT)): digest(path) for path in (
        folder / 'best.index', folder / 'best.data-00000-of-00001',
        folder / 'config.json', folder / 'result.json')}


def freeze():
    summary_path = RUNS / 'hparam-combination-summary.json'
    if read_json(RUNS / 'hparam-combination-status.json')['status'] != 'completed':
        raise ValueError('Combination validation must be complete before test selection')
    base = read_json(PLAN)
    base_hash = digest(PLAN)
    author = author_manifest()
    manifest = {
        'selection_summary_sha256': digest(summary_path),
        'base_plan_sha256': base_hash,
        'combination_plan_sha256': digest(RUNS / 'hparam-combination-plan.json'),
        'selection_rule': 'minimum validation total loss checkpoint, maximum validation IRR configuration',
        'normalization': 'training-period maximum closing price per ticker',
        'validation_targets': [VALID_END - 252, VALID_END - 1],
        'test_targets': [VALID_END, END - 1],
        'test_first': '2017-01-03', 'test_last': '2017-12-08',
        'python': base['python'], 'tensorflow': base['tensorflow'],
        'code_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (
            Path(__file__), ROOT / 'experiments' / 'search_training.py',
            ROOT / 'compat' / 'training' / 'rank_lstm.py',
            ROOT / 'compat' / 'training' / 'relation_rank_lstm.py',
            ROOT / 'compat' / 'training' / 'load_data.py',
            ROOT / 'compat' / 'training' / 'evaluator.py')},
        'input_sha256': {}, 'markets': {}, 'embeddings': {}, 'runs': [],
    }
    for row in read_json(summary_path):
        selected = row['selected']
        run_id = selected['id']
        saved = read_json(RUNS / run_id / 'config.json')
        config = saved['configuration']
        expected = read_json(RUNS / run_id / 'result.json')
        if (saved['manifest_sha256'] != base_hash
                or tuple(config[key] for key in ('market', 'relation', 'method')) !=
                   tuple(row[key] for key in ('market', 'relation', 'method'))
                or any(selected[key] != expected[key] for key in expected)):
            raise ValueError(f'Invalid selection provenance: {run_id}')
        manifest['input_sha256'].update(fingerprint_checkpoint(run_id))
        emb_id = config['embedding_run']
        emb_config = read_json(RUNS / emb_id / 'config.json')['configuration']
        if (emb_config['market'], emb_config['seq'], emb_config['unit']) != (
                config['market'], config['seq'], config['unit']):
            raise ValueError(f'Pretrained representation mismatch: {run_id}')
        manifest['input_sha256'].update(fingerprint_checkpoint(emb_id))
        manifest['embeddings'][emb_id] = emb_config
        market = config['market']
        template = next(c for c in author['configurations'].values()
                        if c['market'] == market and c['relation'] == config['relation'])
        for name, sha in template['file_sha256'].items():
            if not name.startswith('pretrain/'):
                manifest['input_sha256'][str((SOURCE / name).relative_to(ROOT))] = sha
        manifest['markets'][market] = {key: template[key] for key in
                                     ('ticker_file', 'stocks', 'eod_manifest_sha256')}
        manifest['input_sha256'][str((CACHE / f'{market}.npz').relative_to(ROOT))] = read_json(
            CACHE / 'preflight.json')['markets'][market]['file_sha256']
        manifest['runs'].append({'id': run_id, 'seed': saved['run']['seed'],
                                 'configuration': config, 'best_epoch': selected['best_epoch'],
                                 'validation': selected['validation']})
    groups = {(r['configuration']['market'], r['configuration']['relation'],
               r['configuration']['method']) for r in manifest['runs']}
    if len(manifest['runs']) != len(groups) or len(groups) != 8 or len(manifest['embeddings']) != 6:
        raise ValueError('Expected eight selected configurations and six distinct encoders')
    if FROZEN.exists():
        if read_json(FROZEN) != manifest:
            raise ValueError('A different selection has already been frozen')
    else:
        save_json(FROZEN, manifest)
    print(f'Frozen eight configurations and six encoders: {digest(FROZEN)}', flush=True)


def verify_frozen(manifest):
    for name, sha in (manifest['code_sha256'] | manifest['input_sha256']).items():
        if digest(ROOT / name) != sha:
            raise ValueError(f'Frozen input changed: {name}')


def prepare_prices(manifest):
    for market, config in manifest['markets'].items():
        with np.load(CACHE / f'{market}.npz', allow_pickle=False) as cached:
            scales, tickers = cached['scales'], cached['tickers']
            expected_eod, expected_mask, expected_gt = (cached[key] for key in ('eod', 'mask', 'targets'))
        eod = np.empty((len(tickers), END, 5), dtype=np.float32)
        source_hash = hashlib.sha256()
        for i, ticker in enumerate(tickers):
            name = f'{market}_{ticker}_1.csv'
            source = (SOURCE / '2013-01-01' / name).read_bytes()
            source_hash.update(f'{name} {hashlib.sha256(source).hexdigest()}\n'.encode())
            values = np.loadtxt(io.BytesIO(source), delimiter=',', dtype=np.float32)
            if values.shape != (END + (market == 'NASDAQ'), 6):
                raise ValueError(f'Price row count changed: {name}')
            if not np.array_equal(values[:END, 0], np.arange(END)):
                raise ValueError(f'Unexpected price indices: {name}')
            eod[i] = values[:END, 1:]
        if source_hash.hexdigest() != config['eod_manifest_sha256']:
            raise ValueError(f'Author price files changed: {market}')
        missing = eod == -1234.0
        mask = ~missing[:, :, -1]
        close = eod[:, :, -1]
        if np.any(close[mask] <= 0) or not np.isfinite(eod).all():
            raise ValueError(f'Invalid price data: {market}')
        targets = np.zeros_like(close)
        pairs = mask[:, 1:] & mask[:, :-1]
        np.divide(close[:, 1:] - close[:, :-1], close[:, :-1], out=targets[:, 1:], where=pairs)
        eod /= scales[:, None, None]
        eod[missing] = 1.1
        np.testing.assert_array_equal(eod[:, :VALID_END], expected_eod)
        np.testing.assert_array_equal(mask[:, :VALID_END], expected_mask)
        np.testing.assert_array_equal(targets[:, :VALID_END], expected_gt)
        dates = (SOURCE / f'{market}_aver_line_dates.csv').read_text().splitlines()[29:29 + END]
        if dates[VALID_END][:10] != manifest['test_first'] or dates[-1][:10] != manifest['test_last']:
            raise ValueError(f'Test dates changed: {market}')
        path = OUTPUT / f'{market}.npz'
        temporary = OUTPUT / f'{market}.tmp.npz'
        np.savez(temporary, eod=eod, mask=mask.astype(np.float32), targets=targets,
                 dates=dates, tickers=tickers)
        temporary.replace(path)
        print(f'Prepared {market}: {END - VALID_END} test dates, training-only normalization', flush=True)


def encode(manifest, run_id):
    from search_training import SearchRankLSTM, session_config, tf

    config = manifest['embeddings'][run_id]
    model = SearchRankLSTM(config, 123456789)
    feature, *rest = model.build_graph(training=False)
    embedding_tensor = rest[-2]
    seq = config['seq']
    with np.load(OUTPUT / f'{config["market"]}.npz', allow_pickle=False) as data:
        eod = data['eod']
    reference = np.load(RUNS / run_id / 'embedding.npy', mmap_mode='r', allow_pickle=False)
    shape = (model.batch_size, END - TRAIN_END, config['unit'])
    path = OUTPUT / f'{run_id}.npy'
    temporary = path.with_suffix('.tmp.npy')
    embedding = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=shape)
    saver = tf.train.Saver()
    maximum_difference = 0.0
    with tf.Session(config=session_config()) as session:
        saver.restore(session, str(RUNS / run_id / 'best'))
        for column, day in enumerate(range(TRAIN_END, END)):
            offset = day - seq
            value = session.run(embedding_tensor, {feature: eod[:, offset:day, :]})
            if not np.isfinite(value).all():
                raise ValueError(f'Nonfinite embedding: {run_id}, day {day}')
            if day < VALID_END:
                np.testing.assert_allclose(value, reference[:, offset, :], rtol=1e-5, atol=1e-7)
                maximum_difference = max(maximum_difference, float(np.max(np.abs(value - reference[:, offset, :]))))
            embedding[:, column, :] = value
    embedding.flush()
    del embedding
    temporary.replace(path)
    save_json(OUTPUT / f'{run_id}.json', {'manifest_sha256': digest(FROZEN),
                                         'maximum_validation_embedding_difference': maximum_difference,
                                         'file_sha256': digest(path)})


def infer(manifest, run_id):
    from search_training import SearchRSR, metrics_for, session_config, tf

    selected = next(run for run in manifest['runs'] if run['id'] == run_id)
    config = selected['configuration']
    model = SearchRSR(config, selected['seed'], probe=True)
    feature, mask_input, price, target_input, prediction, *_ = model.build_graph(training=False)
    embedding = np.load(OUTPUT / f'{config["embedding_run"]}.npy', mmap_mode='r', allow_pickle=False)
    with np.load(OUTPUT / f'{config["market"]}.npz', allow_pickle=False) as data:
        close = data['eod'][:, :, -1]
        masks, targets = data['mask'], data['targets']
        dates, tickers = data['dates'], data['tickers']
    saver = tf.train.Saver()
    partitions = {}
    with tf.Session(config=session_config()) as session:
        saver.restore(session, str(RUNS / run_id / 'best'))
        model.initialize_relation(session)
        for partition, first, stop in (('validation', TRAIN_END, VALID_END), ('test', VALID_END, END)):
            pred, gt, mask = (np.empty((model.batch_size, stop - first)) for _ in range(3))
            for column, day in enumerate(range(first, stop)):
                day_mask = np.min(masks[:, day - config['seq']:day + 1], axis=1)
                value = session.run(prediction, {
                    feature: embedding[:, day - TRAIN_END, :], mask_input: day_mask[:, None],
                    price: close[:, day - 1, None], target_input: targets[:, day, None],
                })
                pred[:, column], gt[:, column], mask[:, column] = value[:, 0], targets[:, day], day_mask
            metrics = metrics_for(pred, gt, mask)
            if partition == 'validation':
                with np.load(RUNS / run_id / 'predictions.npz', allow_pickle=False) as saved:
                    np.testing.assert_array_equal(gt, saved['valid_gt'])
                    np.testing.assert_array_equal(mask, saved['valid_mask'])
                    np.testing.assert_allclose(pred, saved['valid_pred'], rtol=1e-5, atol=1e-7)
                np.testing.assert_allclose([metrics[key] for key in ('mse', 'mrr', 'irr')],
                                           [selected['validation'][key] for key in ('mse', 'mrr', 'irr')],
                                           rtol=1e-4, atol=1e-9)
                print(f'{run_id}: validation checkpoint reproduced', flush=True)
            partitions[partition] = metrics
            if partition == 'test':
                np.savez_compressed(OUTPUT / f'{run_id}.predictions.npz', test_pred=pred, test_gt=gt,
                                    test_mask=mask, dates=dates[first:stop], tickers=tickers)
    save_json(OUTPUT / f'{run_id}.json', {'manifest_sha256': digest(FROZEN), 'id': run_id,
                                         'configuration': config, 'best_epoch': selected['best_epoch'],
                                         **partitions, 'optimizer_steps_executed': 0})


def worker(manifest, kind, run_id):
    from search_training import tf

    if platform.python_version() != manifest['python'] or tf.__version__ != manifest['tensorflow']:
        raise RuntimeError('Python or TensorFlow differs from the frozen environment')
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('A registered GPU is required')
    (encode if kind == 'embedding' else infer)(manifest, run_id)


def evaluate(manifest):
    verify_frozen(manifest)
    OUTPUT.mkdir(exist_ok=True)
    jobs = ([('embedding', run_id) for run_id in manifest['embeddings']]
            + [('relation', run['id']) for run in manifest['runs']])
    state = {'status': 'preparing', 'manifest_sha256': digest(FROZEN),
             'active': None, 'completed': [], 'optimizer_steps_executed': 0}
    state_path = OUTPUT / 'status.json'
    save_json(state_path, state)
    try:
        prepare_prices(manifest)
        for kind, run_id in jobs:
            marker = OUTPUT / f'{run_id}.json'
            if marker.exists():
                if read_json(marker)['manifest_sha256'] != digest(FROZEN):
                    raise ValueError(f'Existing evaluation belongs to a different selection: {run_id}')
            else:
                state.update(status='running', active=run_id)
                save_json(state_path, state)
                print(f'Evaluating {kind}: {run_id}', flush=True)
                with (OUTPUT / f'{run_id}.log').open('ab') as log:
                    subprocess.run([sys.executable, '-u', __file__, '--worker', kind, run_id],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            state['completed'].append(run_id)
            save_json(state_path, state)
        results = [read_json(OUTPUT / f'{run["id"]}.json') for run in manifest['runs']]
        save_json(OUTPUT / 'summary.json', {'manifest_sha256': digest(FROZEN),
                                           'test_first': manifest['test_first'], 'test_last': manifest['test_last'],
                                           'test_days': END - VALID_END, 'optimizer_steps_executed': 0,
                                           'runs': results})
        state.update(status='completed', active=None)
    except (OSError, ValueError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
        state.update(status='stopped', message=str(error))
        raise
    finally:
        save_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--evaluate', action='store_true')
    mode.add_argument('--worker', nargs=2, metavar=('KIND', 'RUN'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(read_json(FROZEN), *args.worker)
        return
    with (RUNS / 'hparam-search.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.freeze:
            freeze()
        else:
            evaluate(read_json(FROZEN))


if __name__ == '__main__':
    main()
