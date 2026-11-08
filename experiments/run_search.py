"""Run the confirmed baseline, S, U and alpha searches on the WSL GPU."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
STATE = RUNS / 'hparam-search-status.json'
PLAN = RUNS / 'hparam-search-plan.json'
CACHE = RUNS / 'hparam-search-data'
SOURCE = ROOT / 'compat' / 'data'
SEED = 123456789
TRAIN_END = 756
VALID_END = 1008
FIRST_TARGET = 16
EPOCHS = 50
LR = 0.001
BASELINES = {'NASDAQ': (16, 64, 0.1), 'NYSE': (8, 32, 10)}
AXES = {'seq': (2, 4, 8, 16), 'unit': (16, 32, 64, 128), 'alpha': (0.1, 1, 10)}
WORKER = Path(__file__).with_name('search_training.py')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def author_manifest():
    return read_json(ROOT / 'experiments' / 'manifest.json')


def build_plan():
    author = author_manifest()
    plan = {
        'scope': 'baseline_and_independent_axes_only',
        'python': author['python'], 'tensorflow': author['tensorflow'],
        'train_targets': [FIRST_TARGET, TRAIN_END - 1],
        'validation_targets': [TRAIN_END, VALID_END - 1],
        'normalization': 'training-period maximum close per ticker',
        'checkpoint': 'minimum validation total loss',
        'ranking': 'maximum validation IRR at selected checkpoint',
        'code_sha256': {str(path.relative_to(ROOT)): digest(path) for path in
                        (Path(__file__).resolve(), WORKER, ROOT / 'experiments' / 'manifest.json')},
        'configurations': {}, 'runs': [],
    }
    embeddings = {}
    for batch, axis in enumerate((None, *AXES), start=1):
        for market, baseline in BASELINES.items():
            settings = [baseline]
            if axis:
                column = tuple(AXES).index(axis)
                settings = [tuple(value if i == column else original
                                  for i, original in enumerate(baseline))
                            for value in AXES[axis] if value != baseline[column]]
            relations = [conf for conf in author['configurations'].values() if conf['market'] == market]
            for seq, unit, alpha in settings:
                pair = (market, seq, unit)
                common = {'market': market, 'seq': seq, 'unit': unit, 'lr': LR, 'steps': 1,
                          'epochs': EPOCHS, 'ticker_file': relations[0]['ticker_file'],
                          'stocks': relations[0]['stocks'], 'batch': batch,
                          'axis': axis or 'baseline', 'parameter_provenance': 'independent_search'}
                prefix = f'hparam-{market.lower()}-s{seq}-u{unit}'

                def register(run_id, config):
                    plan['configurations'][run_id] = {**common, **config}
                    plan['runs'].append({'id': run_id, 'configuration': run_id, 'seed': SEED})

                if pair not in embeddings:
                    embeddings[pair] = f'{prefix}-rank-lstm'
                    register(embeddings[pair], {'stage': 'embedding', 'alpha': 1, 'method': 'Rank_LSTM',
                                                'relation': None, 'inner_prod': None})
                for relation in relations:
                    run_id = f'{prefix}-a{alpha:g}-{relation["relation"]}-{relation["method"].lower()}'
                    register(run_id, {'stage': 'relation', 'alpha': alpha, 'method': relation['method'],
                                      'relation': relation['relation'], 'inner_prod': relation['inner_prod'],
                                      'embedding_run': embeddings[pair]})
    expected = ((2, 8), (6, 24), (6, 24), (0, 16))
    observed = tuple(tuple(sum(conf['batch'] == batch and conf['stage'] == stage
                               for conf in plan['configurations'].values())
                           for stage in ('embedding', 'relation')) for batch in range(1, 5))
    if len(plan['runs']) != len(plan['configurations']) or observed != expected:
        raise ValueError(f'Search matrix does not match the approved batches: {observed}')
    return plan


def run_checked(args, log_path):
    with log_path.open('ab') as log:
        subprocess.run([sys.executable, '-u', str(WORKER), *args],
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def summarize(plan):
    groups = {}
    for run_id, conf in plan['configurations'].items():
        if conf['stage'] != 'relation':
            continue
        result = read_json(RUNS / run_id / 'result.json')
        group = (conf['market'], conf['relation'], conf['method'])
        groups.setdefault(group, []).append({
            'id': run_id, 'axis': conf['axis'],
            'parameters': {key: conf[key] for key in AXES}, **result,
        })
    summary = []
    for (market, relation, method), candidates in groups.items():
        baseline = next(row for row in candidates if row['axis'] == 'baseline')
        def score(row):
            metrics = row['validation']
            return (metrics['irr'], metrics['mrr'], -metrics['mse'], row is baseline)

        winners = {axis: max((row for row in candidates if row['axis'] in ('baseline', axis)), key=score)
                   for axis in AXES}
        combined = {axis: row['parameters'][axis] for axis, row in winners.items()}
        summary.append({'market': market, 'relation': relation, 'method': method,
                        'baseline': baseline, 'winners': winners, 'combined': combined,
                        'needs_combination_run': not any(row['parameters'] == combined for row in candidates),
                        'best_observed': max(candidates, key=score),
                        'candidates': sorted(candidates, key=score, reverse=True)})
    save_json(RUNS / 'hparam-search-summary.json', summary)


def execute(plan):
    if not CACHE.joinpath('preflight.json').is_file():
        raise SystemExit('Complete the no-training preflight first.')
    if plan != read_json(PLAN) or read_json(CACHE / 'preflight.json')['plan_sha256'] != digest(PLAN):
        raise SystemExit('The checked plan differs from the current plan.')
    state = {'status': 'running', 'plan_sha256': digest(PLAN), 'active': None, 'completed': []}
    save_json(STATE, state)
    try:
        for run in plan['runs']:
            run_id = run['id']
            conf = plan['configurations'][run_id]
            folder = RUNS / run_id
            if (folder / 'result.json').exists():
                if read_json(folder / 'config.json') != {
                    'manifest_sha256': state['plan_sha256'], 'run': run, 'configuration': conf
                }:
                    raise ValueError(f'Existing result does not match the plan: {run_id}')
            else:
                folder.mkdir(exist_ok=False)
                save_json(folder / 'config.json', {'manifest_sha256': state['plan_sha256'],
                                                   'run': run, 'configuration': conf})
                state.update(active=run_id, batch=conf['batch'])
                save_json(STATE, state)
                print(f'Batch {conf["batch"]}: {run_id}', flush=True)
                run_checked(['--run', run_id], folder / 'train.log')
            state['completed'].append(run_id)
            save_json(STATE, state)
        summarize(plan)
        state.update(status='completed', active=None)
    except (OSError, ValueError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
        state.update(status='stopped', message=str(error))
        raise
    finally:
        save_json(STATE, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--show-plan', action='store_true')
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--execute-batches-1-4', action='store_true')
    args = parser.parse_args()
    plan = build_plan()
    if args.show_plan:
        print(json.dumps(plan, indent=2))
        return
    with (RUNS / 'hparam-search.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.preflight:
            if PLAN.exists() and read_json(PLAN) != plan and any(
                    (RUNS / run['id']).exists() for run in plan['runs']):
                raise SystemExit('Existing training outputs are bound to a different plan.')
            save_json(PLAN, plan)
            CACHE.mkdir(exist_ok=True)
            run_checked(['--preflight'], CACHE / 'preflight.log')
        else:
            execute(plan)


if __name__ == '__main__':
    main()
