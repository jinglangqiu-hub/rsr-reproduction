"""Train a single candidate with the original model graphs and isolated labels."""

import argparse
import hashlib
import io
from pathlib import Path
import platform
import random
import sys
from time import monotonic

import numpy as np

from run_search import (CACHE, FIRST_TARGET, PLAN, ROOT, RUNS, SOURCE, TRAIN_END,
                        VALID_END, author_manifest, digest, read_json, save_json)

sys.path.insert(0, str(ROOT / 'compat' / 'training'))
from evaluator import evaluate
from load_data import load_relation_data
from rank_lstm import RankLSTM, tf
from relation_rank_lstm import ReRaLSTM


RELATION_TAIL = {'wikidata': 'wiki_relation.npy', 'sector_industry': 'industry_relation.npy'}


def relation_path(config):
    relation = config['relation']
    return SOURCE / 'relation' / relation / f'{config["market"]}_{RELATION_TAIL[relation]}'


def prepare_market(config):
    market = config['market']
    dates = (SOURCE / f'{market}_aver_line_dates.csv').read_text().splitlines()[29:]
    boundaries = {0: '2013-01-02', TRAIN_END - 1: '2015-12-31', TRAIN_END: '2016-01-04',
                  VALID_END - 1: '2016-12-30', VALID_END: '2017-01-03'}
    for index, expected in boundaries.items():
        if dates[index][:10] != expected:
            raise ValueError(f'{market} calendar mismatch at index {index}')
    tickers = (SOURCE / config['ticker_file']).read_text().splitlines()
    if len(tickers) != config['stocks']:
        raise ValueError(f'{market} ticker count changed')
    eod = np.empty((len(tickers), VALID_END, 5), dtype=np.float32)
    data_hash = hashlib.sha256()
    for i, ticker in enumerate(tickers):
        name = f'{market}_{ticker}_1.csv'
        source = (SOURCE / '2013-01-01' / name).read_bytes()
        data_hash.update(f'{name} {hashlib.sha256(source).hexdigest()}\n'.encode())
        values = np.loadtxt(io.BytesIO(source), delimiter=',', dtype=np.float32, max_rows=VALID_END)
        if values.shape != (VALID_END, 6) or not np.array_equal(values[:, 0], np.arange(VALID_END)):
            raise ValueError(f'Unexpected day indices: {name}')
        eod[i] = values[:, 1:]
    if data_hash.hexdigest() != config['eod_manifest_sha256']:
        raise ValueError(f'{market} author price files changed')

    close = eod[:, :, -1]
    missing = eod == -1234.0
    mask = ~missing[:, :, -1]
    if not np.isfinite(eod).all() or np.any(close[mask] <= 0):
        raise ValueError(f'{market} has invalid closing prices')
    scales = np.max(np.where(mask[:, :TRAIN_END], close[:, :TRAIN_END], 0), axis=1)
    if np.any(scales <= 0):
        raise ValueError(f'{market} has a ticker without a positive training-period maximum')
    targets = np.zeros_like(close)
    pairs = mask[:, 1:] & mask[:, :-1]
    np.divide(close[:, 1:] - close[:, :-1], close[:, :-1], out=targets[:, 1:], where=pairs)
    eod /= scales[:, None, None]
    eod[missing] = 1.1
    if not np.allclose(np.max(np.where(mask[:, :TRAIN_END], close[:, :TRAIN_END], 0), axis=1), 1):
        raise ValueError(f'{market} training price normalization failed')
    path = CACHE / f'{market}.npz'
    temporary = path.with_suffix('.tmp.npz')
    np.savez(temporary, eod=eod, mask=mask.astype(np.float32), targets=targets,
             scales=scales, tickers=tickers, dates=dates[:VALID_END])
    temporary.replace(path)
    print(f'{market}: {len(tickers)} tickers, {eod.shape[1]} pre-test dates', flush=True)
    return {'file_sha256': digest(path), 'stocks': len(tickers),
            'train_first': dates[0][:10], 'train_last': dates[TRAIN_END - 1][:10],
            'valid_first': dates[TRAIN_END][:10], 'valid_last': dates[VALID_END - 1][:10]}


class SearchInputs:
    def __init__(self, config, seed):
        with np.load(CACHE / f'{config["market"]}.npz', allow_pickle=False) as source:
            self.tickers = source['tickers']
            self.eod_data = source['eod']
            self.mask_data = source['mask']
            self.gt_data = source['targets']
        self.price_data = self.eod_data[:, :, -1]
        self.parameters = {key: config[key] for key in ('seq', 'unit', 'alpha', 'lr')}
        self.batch_size = len(self.tickers)
        self.fea_dim = self.eod_data.shape[2]
        self.steps = config['steps']
        self.valid_index = TRAIN_END
        self.test_index = VALID_END
        self.seed = seed
        self.gpu = True


class SearchRankLSTM(SearchInputs, RankLSTM):
    pass


class SearchRSR(SearchInputs, ReRaLSTM):
    def __init__(self, config, seed, *, probe=False):
        super().__init__(config, seed)
        self.flat = False
        self.inner_prod = config['inner_prod']
        self.rel_encoding, self.rel_mask = load_relation_data(relation_path(config))
        shape = (self.batch_size, VALID_END - config['seq'], config['unit'])
        if probe:
            self.embedding = np.zeros((self.batch_size, 1, config['unit']), dtype=np.float32)
        else:
            folder = RUNS / config['embedding_run']
            result = read_json(folder / 'result.json')
            if result['manifest_sha256'] != digest(PLAN):
                raise ValueError(f'Embedding belongs to a different search: {folder.name}')
            self.embedding = np.load(folder / 'embedding.npy', mmap_mode='r', allow_pickle=False)
            if self.embedding.shape != shape or self.embedding.dtype != np.float32:
                raise ValueError(f'Embedding shape or type mismatch: {folder.name}')


def graph(model, embedding_stage):
    random.seed(model.seed)
    np.random.seed(model.seed)
    outputs = model.build_graph(training=True)
    return outputs[:4], outputs[4:8], outputs[-1], outputs[-2] if embedding_stage else None


def initialize(session, model, embedding_stage):
    session.run(tf.global_variables_initializer())
    if embedding_stage:
        # The legacy Rank_LSTM resets its graph before building variables.
        # Seed the original Glorot kernels and zero biases explicitly.
        assignments = []
        for index, variable in enumerate(tf.trainable_variables()):
            name = variable.op.name.rsplit('/', 1)[-1]
            if name == 'kernel':
                initial = tf.glorot_uniform_initializer(seed=model.seed + index)(variable.shape, tf.float32)
            elif name == 'bias':
                initial = tf.zeros(variable.shape, dtype=tf.float32)
            else:
                raise ValueError(f'Unexpected Rank_LSTM trainable variable: {variable.name}')
            assignments.append(variable.assign(initial))
        session.run(assignments)
    else:
        model.initialize_relation(session)


def session_config():
    config = tf.ConfigProto(allow_soft_placement=False)
    config.gpu_options.allow_growth = True
    return config


def metrics_for(pred, target, mask):
    raw = evaluate(pred, target, mask)
    return {'mse': float(raw['mse']), 'mrr': float(raw['mrrt']),
            'irr': float(raw['btl']) - 1.0, 'author_btl': float(raw['btl'])}


def export_embedding(session, tensor, input_feature, model, folder):
    seq = model.parameters['seq']
    shape = (model.batch_size, VALID_END - seq, model.parameters['unit'])
    temporary = folder / 'embedding.tmp.npy'
    output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=shape)
    for offset in range(shape[1]):
        value = session.run(tensor, {input_feature: model.eod_data[:, offset:offset + seq, :]})
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite embedding at offset {offset}')
        output[:, offset, :] = value
    output.flush()
    del output
    temporary.replace(folder / 'embedding.npy')
    return list(shape)


def train(config, run, folder):
    print({'run': run, 'configuration': config}, flush=True)
    is_embedding = config['stage'] == 'embedding'
    model = (SearchRankLSTM if is_embedding else SearchRSR)(config, run['seed'])
    inputs, (prediction, loss, regression, ranking), optimizer, embedding = graph(model, is_embedding)
    seq = config['seq']
    train_offsets = np.arange(FIRST_TARGET - seq, TRAIN_END - seq)
    valid_offsets = range(TRAIN_END - seq, VALID_END - seq)
    rng = np.random.RandomState(run['seed'])
    checkpoint = str(folder / 'best')
    best = None
    saver = tf.train.Saver()
    with tf.Session(config=session_config()) as session:
        initialize(session, model, is_embedding)
        for epoch in range(config['epochs']):
            start = monotonic()
            rng.shuffle(train_offsets)
            train_totals = np.zeros(3)
            for offset in train_offsets:
                values = session.run((loss, regression, ranking, optimizer),
                                     dict(zip(inputs, model.get_batch(offset))))
                train_totals += values[:3]
            train_means = train_totals / len(train_offsets)
            print('Train Loss:', *train_means)
            valid_totals = np.zeros(3)
            pred, target, mask = (np.empty((model.batch_size, len(valid_offsets))) for _ in range(3))
            for index, offset in enumerate(valid_offsets):
                batch = model.get_batch(offset)
                value, *parts = session.run((prediction, loss, regression, ranking), dict(zip(inputs, batch)))
                valid_totals += parts
                pred[:, index], target[:, index], mask[:, index] = value[:, 0], batch[3][:, 0], batch[1][:, 0]
            valid_means = valid_totals / len(valid_offsets)
            if not np.isfinite(train_means).all() or not np.isfinite(valid_means).all():
                raise ValueError(f'Nonfinite loss in epoch {epoch + 1}')
            metrics = metrics_for(pred, target, mask)
            print('Valid MSE:', *valid_means)
            print('Valid performance:', {'mse': metrics['mse'], 'mrrt': metrics['mrr'],
                                         'btl': metrics['author_btl']})
            if best is None or valid_means[0] < best['best_validation_loss']:
                best = {'manifest_sha256': digest(PLAN), 'id': run['id'], 'best_epoch': epoch + 1,
                        'best_validation_loss': float(valid_means[0]), 'validation': metrics}
                saver.save(session, checkpoint, write_meta_graph=False)
                if not is_embedding:
                    best_predictions = (pred, target, mask)
                print('Better valid loss:', best['best_validation_loss'])
            print(f'epoch: {epoch} time: {monotonic() - start:.4f}')
        if is_embedding:
            saver.restore(session, checkpoint)
            best['embedding_shape'] = export_embedding(session, embedding, inputs[0], model, folder)
        else:
            np.savez_compressed(folder / 'predictions.npz',
                                **dict(zip(('valid_pred', 'valid_gt', 'valid_mask'), best_predictions)))
    save_json(folder / 'result.json', best)


def preflight(plan):
    report_path = CACHE / 'preflight.json'
    if report_path.exists():
        report_path.unlink()
    author = author_manifest()
    for relative, expected in author['file_sha256'].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f'Frozen source changed: {relative}')
    files = {}
    markets = {}
    for config in author['configurations'].values():
        relative = relation_path(config).relative_to(SOURCE).as_posix()
        for name in (relative, config['ticker_file'], f'{config["market"]}_aver_line_dates.csv'):
            if name not in files:
                files[name] = digest(SOURCE / name)
            if files[name] != config['file_sha256'][name]:
                raise ValueError(f'Frozen data changed: {name}')
        relation = np.load(SOURCE / relative, mmap_mode='r', allow_pickle=False)
        if relation.ndim != 3 or relation.shape[:2] != (config['stocks'],) * 2:
            raise ValueError(f'Relation matrix has incorrect stock axes: {relative}')
        if config['market'] not in markets:
            markets[config['market']] = prepare_market(config)
    configs = list(plan['configurations'].values())
    for conf in configs:
        lookback = conf['seq']
        if conf['steps'] != 1 or not 0 < lookback <= FIRST_TARGET:
            raise ValueError(f'Unsupported search window: {conf}')
    probes = []
    for embedding_stage in (True, False):
        config = next(conf for conf in configs if conf['market'] == 'NYSE' and conf['unit'] == 128
                      and (conf['stage'] == 'embedding' if embedding_stage
                           else conf['relation'] == 'sector_industry' and conf['method'] == 'RSR_I'))
        model = SearchRankLSTM(config, plan['runs'][0]['seed']) if embedding_stage else SearchRSR(
            config, plan['runs'][0]['seed'], probe=True)
        inputs, (prediction, loss, _, _), _, _ = graph(model, embedding_stage)
        with tf.Session(config=session_config()) as session:
            initialize(session, model, embedding_stage)
            offset = FIRST_TARGET - config['seq'] if embedding_stage else 0
            value, error = session.run((prediction, loss), dict(zip(inputs, model.get_batch(offset))))
            if not np.isfinite(value).all() or not np.isfinite(error):
                raise ValueError(f'Nonfinite GPU preflight: {config}')
        probes.append({key: config[key] for key in ('stage', 'market', 'method', 'relation', 'seq', 'unit')})
        print(f'GPU forward pass OK: {probes[-1]}', flush=True)
    save_json(report_path, {'plan_sha256': digest(PLAN), 'markets': markets, 'gpu_probes': probes,
                           'optimizer_steps_executed': 0, 'training_targets_per_epoch': TRAIN_END - FIRST_TARGET,
                           'validation_targets': VALID_END - TRAIN_END})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--run')
    args = parser.parse_args()
    plan = read_json(PLAN)
    if platform.python_version() != plan['python'] or tf.__version__ != plan['tensorflow']:
        raise RuntimeError('Python or TensorFlow differs from the frozen environment')
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('A registered GPU is required')
    if args.preflight:
        preflight(plan)
    else:
        run = next(item for item in plan['runs'] if item['id'] == args.run)
        conf = plan['configurations'][run['configuration']]
        if digest(CACHE / f'{conf["market"]}.npz') != read_json(
                CACHE / 'preflight.json')['markets'][conf['market']]['file_sha256']:
            raise ValueError(f'Prepared data changed: {conf["market"]}')
        train(conf, run, RUNS / args.run)


if __name__ == '__main__':
    main()

