"""Execute the six approved combination runs without changing batches 1–4."""

import argparse
import fcntl
from pathlib import Path
import platform
import subprocess
import sys

from run_search import (AXES, CACHE, PLAN, RUNS, SEED, STATE, VALID_END,
                        build_plan, digest, read_json, save_json)


COMBINATION_PLAN = RUNS / 'hparam-combination-plan.json'
COMBINATION_STATE = RUNS / 'hparam-combination-status.json'
BASE_SUMMARY = RUNS / 'hparam-search-summary.json'
APPROVED = {
    ('NASDAQ', 'wikidata', 'RSR_I'): (8, 128, 10),
    ('NASDAQ', 'wikidata', 'RSR_E'): (8, 32, 10),
    ('NASDAQ', 'sector_industry', 'RSR_I'): (8, 32, 10),
    ('NASDAQ', 'sector_industry', 'RSR_E'): (4, 16, 10),
    ('NYSE', 'sector_industry', 'RSR_I'): (4, 128, 10),
    ('NYSE', 'wikidata', 'RSR_E'): (16, 32, 1),
}


def group_key(config):
    return tuple(config[key] for key in ('market', 'relation', 'method'))


def make_plan():
    base = read_json(PLAN)
    base_hash = digest(PLAN)
    status = read_json(STATE)
    if (base != build_plan() or status['status'] != 'completed'
            or status['plan_sha256'] != base_hash
            or status['completed'] != [run['id'] for run in base['runs']]
            or read_json(CACHE / 'preflight.json')['plan_sha256'] != base_hash):
        raise ValueError('The completed batches do not match the checked base plan')
    summary = read_json(BASE_SUMMARY)
    combinations = {group_key(row): tuple(row['combined'][key] for key in AXES)
                    for row in summary if row['needs_combination_run']}
    if len(summary) != 8 or combinations != APPROVED:
        raise ValueError('The proposed combinations differ from the approved six')

    plan = {'batch': 5, 'base_plan_sha256': base_hash, 'base_summary_sha256': digest(BASE_SUMMARY),
            'runner_sha256': digest(Path(__file__)), 'runs': [], 'configurations': {},
            'relations': {}}
    baseline_configs = base['configurations']
    embeddings = {(config['market'], config['seq'], config['unit']): run_id
                  for run_id, config in baseline_configs.items() if config['stage'] == 'embedding'}

    def register(run_id, config):
        if run_id in baseline_configs or run_id in plan['configurations']:
            raise ValueError(f'Run already exists in a plan: {run_id}')
        plan['runs'].append({'id': run_id, 'configuration': run_id, 'seed': SEED})
        plan['configurations'][run_id] = {**config, 'batch': 5, 'axis': 'combination',
                                          'parameter_provenance': 'combination_validation'}

    for row in summary:
        group = group_key(row)
        if group not in combinations:
            continue
        market, relation, method = group
        seq, unit, alpha = combinations[group]
        template = baseline_configs[row['baseline']['id']]
        if group_key(template) != group or template['axis'] != 'baseline':
            raise ValueError(f'Baseline does not match the proposed run: {group}')
        prefix = f'hparam-{market.lower()}-s{seq}-u{unit}'
        pair = (market, seq, unit)
        common = {**template, 'seq': seq, 'unit': unit}
        if pair not in embeddings:
            embedding_id = f'{prefix}-rank-lstm'
            embedding_config = {key: value for key, value in common.items() if key != 'embedding_run'}
            register(embedding_id, {**embedding_config, 'stage': 'embedding', 'method': 'Rank_LSTM',
                                    'alpha': 1, 'relation': None, 'inner_prod': None})
            embeddings[pair] = embedding_id
        run_id = f'{prefix}-a{alpha:g}-{relation}-{method.lower()}'
        register(run_id, {**common, 'alpha': alpha, 'embedding_run': embeddings[pair]})
        plan['relations'][row['baseline']['id']] = run_id
    if len(plan['runs']) != 10 or len(plan['relations']) != 6:
        raise ValueError('Expected four embedding runs and six relation runs')
    return plan


def verify_inputs(plan):
    import numpy as np

    fingerprints = read_json(CACHE / 'preflight.json')['markets']
    for market in fingerprints:
        if digest(CACHE / f'{market}.npz') != fingerprints[market]['file_sha256']:
            raise ValueError(f'Prepared data changed: {market}')
    available = set()
    for run in plan['runs']:
        run_id = run['id']
        config = plan['configurations'][run_id]
        if config['stage'] == 'embedding':
            available.add(run_id)
            continue
        dep = config['embedding_run']
        if dep in available:
            continue
        folder = RUNS / dep
        result = read_json(folder / 'result.json')
        embedding = np.load(folder / 'embedding.npy', mmap_mode='r', allow_pickle=False)
        shape = (config['stocks'], VALID_END - config['seq'], config['unit'])
        if (result['manifest_sha256'] != plan['base_plan_sha256'] or result['id'] != dep
                or embedding.shape != shape or embedding.dtype != np.float32):
            raise ValueError(f'Existing embedding does not match the combination: {dep}')
        available.add(dep)


def run_config(plan, run):
    return {'manifest_sha256': plan['base_plan_sha256'], 'combination_plan_sha256': digest(COMBINATION_PLAN),
            'run': run, 'configuration': plan['configurations'][run['configuration']]}


def checked_result(folder, expected):
    result = read_json(folder / 'result.json')
    if (read_json(folder / 'config.json') != expected
            or result['manifest_sha256'] != expected['manifest_sha256']
            or result['id'] != expected['run']['id']):
        raise ValueError(f'Run does not match the approved plan: {folder.name}')
    return result


def worker(plan, run_id):
    from search_training import tf, train

    base = read_json(PLAN)
    if platform.python_version() != base['python'] or tf.__version__ != base['tensorflow']:
        raise RuntimeError('Python or TensorFlow differs from the frozen environment')
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('A registered GPU is required')
    run = next(run for run in plan['runs'] if run['id'] == run_id)
    expected = run_config(plan, run)
    folder = RUNS / run_id
    if read_json(COMBINATION_STATE)['active'] != run_id or read_json(folder / 'config.json') != expected:
        raise ValueError(f'Run is not active in the approved queue: {run_id}')
    train(expected['configuration'], run, folder)


def summarize(plan):
    rows = []
    for row in read_json(BASE_SUMMARY):
        observed = row['best_observed']
        combination_id = plan['relations'].get(row['baseline']['id'])
        if combination_id:
            config = plan['configurations'][combination_id]
            combination = {'parameters': {key: config[key] for key in AXES},
                           **read_json(RUNS / combination_id / 'result.json')}
        else:
            combination = next(run for run in row['candidates'] if run['parameters'] == row['combined'])

        def score(run):
            metrics = run['validation']
            return metrics['irr'], metrics['mrr'], -metrics['mse']

        rows.append({key: row[key] for key in ('market', 'relation', 'method', 'baseline')} |
                    {'best_observed': observed, 'combination': combination,
                     'combination_run_in_batch_5': combination_id is not None,
                     'selected': max((observed, combination), key=score)})
    save_json(RUNS / 'hparam-combination-summary.json', rows)


def execute(plan):
    state = {'status': 'running', 'batch': 5, 'plan_sha256': digest(COMBINATION_PLAN),
             'active': None, 'completed': []}
    try:
        for run in plan['runs']:
            run_id = run['id']
            folder = RUNS / run_id
            expected = run_config(plan, run)
            if not (folder / 'result.json').exists():
                folder.mkdir(exist_ok=False)
                save_json(folder / 'config.json', expected)
                state['active'] = run_id
                save_json(COMBINATION_STATE, state)
                print(f'Batch 5: {run_id}', flush=True)
                with (folder / 'train.log').open('ab') as log:
                    subprocess.run([sys.executable, '-u', __file__, '--worker', run_id],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            checked_result(folder, expected)
            state['completed'].append(run_id)
            save_json(COMBINATION_STATE, state)
        summarize(plan)
        state.update(status='completed', active=None)
    except (OSError, ValueError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
        state.update(status='stopped', message=str(error))
        raise
    finally:
        save_json(COMBINATION_STATE, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--worker', help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = make_plan()
    if args.worker:
        if plan != read_json(COMBINATION_PLAN):
            raise ValueError('The combination plan changed')
        worker(plan, args.worker)
        return
    with (RUNS / 'hparam-search.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare:
            verify_inputs(plan)
            if COMBINATION_PLAN.exists() and plan != read_json(COMBINATION_PLAN):
                raise ValueError('The saved combination plan differs')
            save_json(COMBINATION_PLAN, plan)
            print('Batch 5 prepared: four Rank_LSTM runs, six RSR runs, zero optimizer steps')
        else:
            if plan != read_json(COMBINATION_PLAN):
                raise ValueError('The prepared combination plan changed')
            verify_inputs(plan)
            execute(plan)


if __name__ == '__main__':
    main()
