"""Inspect the frozen RSR plan or explicitly execute one registered run."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'compat' / 'data'
PLAN_PATH = ROOT / 'experiments' / 'manifest.json'
sys.path.insert(0, str(ROOT / 'compat' / 'training'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def verify_inputs(plan, configurations):
    for relative_path, expected in plan['file_sha256'].items():
        if digest(ROOT / relative_path) != expected:
            raise ValueError(f'Frozen file changed: {relative_path}')
    checked_files = {}
    checked_markets = set()
    for configuration in configurations:
        for relative_path, expected in configuration['file_sha256'].items():
            if relative_path not in checked_files:
                checked_files[relative_path] = digest(DATA / relative_path)
            if checked_files[relative_path] != expected:
                raise ValueError(f'Frozen data changed: {relative_path}')
        market_data = (configuration['market'], configuration['ticker_file'],
                       configuration['stocks'], configuration['eod_manifest_sha256'])
        if market_data in checked_markets:
            continue
        tickers = (DATA / configuration['ticker_file']).read_text().splitlines()
        if len(tickers) != configuration['stocks']:
            raise ValueError(f'Ticker count changed: {configuration["market"]}')
        source_manifest = hashlib.sha256()
        for ticker in tickers:
            name = f'{configuration["market"]}_{ticker}_1.csv'
            source_manifest.update(f'{name} {digest(DATA / "2013-01-01" / name)}\n'.encode())
        if source_manifest.hexdigest() != configuration['eod_manifest_sha256']:
            raise ValueError(f'Frozen EOD data changed: {configuration["market"]}')
        checked_markets.add(market_data)


def verify_relation_configuration(plan, run, configuration):
    expected_method = 'RSR_E' if configuration['inner_prod'] else 'RSR_I'
    if configuration['method'] != expected_method or run['seed'] != plan['public_seed']:
        raise ValueError(f'Model branch or public seed mismatch: {run["id"]}')
    market = configuration['market']
    relation_paths = {'wikidata': f'relation/wikidata/{market}_wiki_relation.npy',
                      'sector_industry': f'relation/sector_industry/{market}_industry_relation.npy'}
    relation_path = relation_paths[configuration['relation']]
    embedding_path = f'pretrain/{configuration["embedding"]}'
    if not {relation_path, embedding_path}.issubset(configuration['file_sha256']):
        raise ValueError(f'Input fingerprints missing for {run["id"]}')
    relation = np.load(DATA / relation_path, mmap_mode='r', allow_pickle=False)
    embedding = np.load(DATA / embedding_path, mmap_mode='r', allow_pickle=False)
    stocks = configuration['stocks']
    if relation.ndim != 3 or relation.shape[:2] != (stocks, stocks):
        raise ValueError(f'Relation axes do not match stock count: {run["id"]}')
    if embedding.shape != (stocks, 1245, configuration['unit']):
        raise ValueError(f'Embedding shape does not match configuration: {run["id"]}')
    return {'relation_shape': list(relation.shape), 'embedding_shape': list(embedding.shape)}


def check_author_windows(configuration, run):
    train_days = 756
    lookahead = configuration['seq'] + configuration['steps'] - 1
    updates = train_days - lookahead
    if updates <= 0:
        raise ValueError('No training windows in the author protocol')
    offsets = np.arange(train_days)
    rng = np.random.RandomState(run['seed'])
    validation_labels = []
    for _ in range(configuration['epochs']):
        rng.shuffle(offsets)
        labels = offsets[:updates] + lookahead
        validation_labels.append(int(np.count_nonzero(labels >= train_days)))
    return {'id': run['id'], 'seed': run['seed'],
            'optimizer_steps_planned': updates * configuration['epochs'],
            'first_epoch_validation_labels_in_train': validation_labels[0],
            'validation_labels_in_train_over_all_epochs': sum(validation_labels)}


def execute(plan, run, configuration):
    from evaluator import evaluate
    from relation_rank_lstm import ReRaLSTM, tf

    if platform.python_version() != plan['python'] or tf.__version__ != plan['tensorflow']:
        raise RuntimeError('Python or TensorFlow differs from the frozen environment')
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('The frozen GPU experiment requires a registered GPU')
    root = ROOT / 'runs'
    plan_sha = digest(PLAN_PATH)
    output = root / run['id']
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'config.json', {'manifest_sha256': plan_sha, 'run': run,
                                       'configuration': configuration})
    print(f'Run log: {output / "train.log"}', flush=True)
    with (output / 'train.log').open('w', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            model = ReRaLSTM(
                str(DATA / '2013-01-01'), configuration['market'], configuration['ticker_file'],
                configuration['relation'], configuration['embedding'],
                {key: configuration[key] for key in ('seq', 'unit', 'alpha', 'lr')},
                steps=configuration['steps'], epochs=configuration['epochs'],
                gpu=True, in_pro=configuration['inner_prod'],
                seed=run['seed'],
            )
            arrays = model.train(checkpoint_prefix=str(output / 'best'))
    if model.best_epoch is None:
        raise RuntimeError('No epoch produced a finite best validation loss')
    names = ('valid_pred', 'valid_gt', 'valid_mask', 'test_pred', 'test_gt', 'test_mask')
    np.savez_compressed(output / 'predictions.npz', **dict(zip(names, arrays)))
    result = {'manifest_sha256': plan_sha, 'id': run['id'], 'best_epoch': model.best_epoch,
              'best_validation_loss': model.best_valid_loss}
    for partition, start in [('validation', 0), ('test', 3)]:
        raw = evaluate(*arrays[start:start + 3])
        result[partition] = {'mse': raw['mse'], 'mrr': raw['mrrt'],
                             'irr': raw['btl'] - 1.0, 'author_btl': raw['btl']}
    save_json(output / 'result.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', help='Exact run ID from the frozen manifest')
    parser.add_argument('--execute', action='store_true', help='Start the selected training run')
    parser.add_argument('--allow-market-template', action='store_true',
                        help='Allow a run whose settings are adapted from the market README example')
    parser.add_argument('--output', type=Path, help='Save the no-training preflight report')
    args = parser.parse_args()
    if args.execute and (not args.run or args.output):
        parser.error('--execute requires --run and cannot be combined with --output')
    plan = json.loads(PLAN_PATH.read_text(encoding='utf-8'))
    runs = {run['id']: run for run in plan['runs']}
    if args.run and args.run not in runs:
        parser.error(f'Unknown run ID: {args.run}')
    selected = [runs[args.run]] if args.run else list(runs.values())
    configurations = plan['configurations']
    if args.execute and configurations[selected[0]['configuration']]['parameter_provenance'] == 'market_template' and not args.allow_market_template:
        parser.error('Exact author settings are unavailable for this combination. '
                     'Use --allow-market-template only to run the documented market-template configuration.')
    verify_inputs(plan, [configurations[key] for key in sorted({r['configuration'] for r in selected})])
    report = {'manifest_sha256': digest(PLAN_PATH), 'input_hashes_verified': True,
              'optimizer_steps_executed': 0,
              'runs': []}
    for run in selected:
        config = configurations[run['configuration']]
        report['runs'].append({**check_author_windows(config, run),
                               **verify_relation_configuration(plan, run, config),
                               'parameter_provenance': config['parameter_provenance']})
    if args.execute:
        run = selected[0]
        execute(plan, run, configurations[run['configuration']])
    else:
        if args.output:
            save_json(args.output, report)
        print(json.dumps(report, indent=2))
