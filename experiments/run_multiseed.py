"""Train five new seeds using the eight validation-selected RSR configurations."""

import argparse
import fcntl
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from run_search import CACHE, PLAN, ROOT, RUNS, digest, read_json, save_json


FROZEN = RUNS / 'frozen-test-manifest.json'
MULTI_PLAN = RUNS / 'multiseed-plan.json'
MULTI_STATE = RUNS / 'multiseed-status.json'
WORKER = Path(__file__).with_name('multiseed_training.py')
SEEDS = (175185289, 425827170, 1846380228, 1772992330, 20885008)


def make_plan():
    frozen = read_json(FROZEN)
    if digest(PLAN) != frozen['base_plan_sha256']:
        raise ValueError('The previous protocol differs from the frozen selection')
    expected = {(market, relation, method) for market in ('NASDAQ', 'NYSE')
                for relation in ('wikidata', 'sector_industry') for method in ('RSR_I', 'RSR_E')}
    groups = {(run['configuration']['market'], run['configuration']['relation'],
               run['configuration']['method']) for run in frozen['runs']}
    if groups != expected or len(frozen['runs']) != 8:
        raise ValueError('Expected the eight previously frozen configurations')
    files = (Path(__file__), WORKER, ROOT / 'experiments' / 'search_training.py',
             ROOT / 'compat' / 'training' / 'rank_lstm.py',
             ROOT / 'compat' / 'training' / 'relation_rank_lstm.py',
             ROOT / 'compat' / 'training' / 'load_data.py',
             ROOT / 'compat' / 'training' / 'evaluator.py', ROOT / 'environment' / 'run_gpu.py')
    plan = {'scope': 'five_seeds_fixed_hyperparameters', 'seed_scope': 'both_training_stages',
            'seeds': list(SEEDS), 'python': frozen['python'], 'tensorflow': frozen['tensorflow'],
            'base_plan_sha256': digest(PLAN), 'frozen_selection_sha256': digest(FROZEN),
            'checkpoint_rule': 'minimum validation total loss',
            'standard_deviation': 'sample standard deviation, ddof=1',
            'train_target_indices': [16, 755], 'validation_target_indices': [756, 1007],
            'test_target_indices': [1008, 1244], 'stage1_alpha': 1,
            'code_sha256': {str(path.relative_to(ROOT)): digest(path) for path in files},
            'input_sha256': {}, 'configurations': {}, 'runs': []}
    for name, sha in frozen['input_sha256'].items():
        if name.startswith('compat/data/') or name.startswith('runs/hparam-search-data/'):
            plan['input_sha256'][name] = sha
    for market in ('NASDAQ', 'NYSE'):
        path = RUNS / 'frozen-test' / f'{market}.npz'
        plan['input_sha256'][str(path.relative_to(ROOT))] = digest(path)

    def register(run_id, seed, config):
        if run_id in plan['configurations']:
            raise ValueError(f'Duplicate run: {run_id}')
        plan['configurations'][run_id] = {
            **config, 'batch': 6, 'axis': 'multiseed', 'epochs': 50, 'lr': .001, 'steps': 1,
            'parameter_provenance': 'five_seed_fixed_hyperparameters',
        }
        plan['runs'].append({'id': run_id, 'configuration': run_id, 'seed': seed})

    for seed in SEEDS:
        embeddings = {}
        for selected in frozen['runs']:
            config = selected['configuration']
            market, seq, unit = (config[key] for key in ('market', 'seq', 'unit'))
            pair = (market, seq, unit)
            if pair not in embeddings:
                embedding_id = f'ms-{market.lower()}-s{seq}-u{unit}-seed-{seed}-rank-lstm'
                emb_config = {key: value for key, value in config.items() if key != 'embedding_run'}
                register(embedding_id, seed, {**emb_config, 'stage': 'embedding', 'alpha': 1,
                                             'method': 'Rank_LSTM', 'relation': None, 'inner_prod': None})
                embeddings[pair] = embedding_id
            run_id = (f'ms-{market.lower()}-{config["relation"]}-{config["method"].lower()}'
                      f'-s{seq}-u{unit}-a{config["alpha"]:g}-seed-{seed}')
            register(run_id, seed, {**config, 'embedding_run': embeddings[pair]})
    if len(plan['runs']) != 70 or len(plan['configurations']) != 70:
        raise ValueError('Expected 30 encoder fits and 40 RSR fits')
    for seed in SEEDS:
        configs = [plan['configurations'][run['configuration']] for run in plan['runs'] if run['seed'] == seed]
        if sum(config['stage'] == 'embedding' for config in configs) != 6 or len(configs) != 14:
            raise ValueError(f'Unexpected model count for seed {seed}')
    return plan


def verify_files(plan):
    for name, sha in (plan['code_sha256'] | plan['input_sha256']).items():
        if digest(ROOT / name) != sha:
            raise ValueError(f'Frozen source or data changed: {name}')
    if digest(FROZEN) != plan['frozen_selection_sha256'] or digest(PLAN) != plan['base_plan_sha256']:
        raise ValueError('The selected hyperparameters or training protocol changed')
    if shutil.disk_usage(RUNS).free < 10 * 1024**3:
        raise ValueError('At least 10 GiB of free space is required for the full five-seed queue')


def saved_config(plan, run):
    return {'manifest_sha256': plan['base_plan_sha256'], 'multiseed_plan_sha256': digest(MULTI_PLAN),
            'run': run, 'configuration': plan['configurations'][run['configuration']]}


def summarize(plan):
    groups = {}
    for run in plan['runs']:
        config = plan['configurations'][run['configuration']]
        if config['stage'] != 'relation' or not (RUNS / run['id'] / 'result.json').exists():
            continue
        result = read_json(RUNS / run['id'] / 'result.json')
        key = tuple(config[field] for field in ('market', 'relation', 'method'))
        groups.setdefault(key, []).append({'seed': run['seed'], **result})
    rows = []
    for group, results in groups.items():
        metrics = {}
        for partition in ('validation', 'test'):
            metrics[partition] = {}
            for metric in ('mse', 'mrr', 'irr'):
                values = np.array([result[partition][metric] for result in results])
                metrics[partition][metric] = {
                    'mean': float(values.mean()),
                    'std': float(values.std(ddof=1)) if len(values) > 1 else None,
                    'min': float(values.min()), 'max': float(values.max()),
                }
        rows.append({'market': group[0], 'relation': group[1], 'method': group[2],
                     'completed_seed_count': len(results), 'results': results, 'metrics': metrics})
    save_json(RUNS / 'multiseed-summary.json', {'multiseed_plan_sha256': digest(MULTI_PLAN),
                                               'standard_deviation_ddof': 1, 'groups': rows})


def execute(plan):
    previous = read_json(MULTI_STATE) if MULTI_STATE.exists() else {}
    if previous and previous['multiseed_plan_sha256'] != digest(MULTI_PLAN):
        raise ValueError('Existing progress belongs to another plan')
    state = {'status': 'running', 'batch': 6, 'multiseed_plan_sha256': digest(MULTI_PLAN),
             'active': None, 'completed': []}
    save_json(MULTI_STATE, state)
    try:
        for run in plan['runs']:
            run_id = run['id']
            folder = RUNS / run_id
            expected = saved_config(plan, run)
            if folder.exists():
                if read_json(folder / 'config.json') != expected:
                    raise ValueError(f'Existing configuration differs: {run_id}')
            else:
                folder.mkdir()
                save_json(folder / 'config.json', expected)
            result_path = folder / 'result.json'
            if not result_path.exists():
                state.update(active=run_id, seed=run['seed'])
                save_json(MULTI_STATE, state)
                print(f'Multiseed: {run_id}', flush=True)
                with (folder / 'console.log').open('ab') as log:
                    subprocess.run([sys.executable, '-u', str(WORKER), '--run', run_id],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
            result = read_json(result_path)
            if (result['id'] != run_id or result['seed'] != run['seed']
                    or result['multiseed_plan_sha256'] != digest(MULTI_PLAN)):
                raise ValueError(f'Unexpected result: {run_id}')
            state['completed'].append(run_id)
            save_json(MULTI_STATE, state)
            summarize(plan)
        state.update(status='completed', active=None)
    except (OSError, ValueError, subprocess.CalledProcessError, KeyboardInterrupt) as error:
        state.update(status='stopped', message=str(error))
        raise
    finally:
        save_json(MULTI_STATE, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan = make_plan()
    with (RUNS / 'hparam-search.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if MULTI_PLAN.exists() and read_json(MULTI_PLAN) != plan:
            raise ValueError('The five-seed plan differs from its saved version')
        verify_files(plan)
        if args.prepare:
            save_json(MULTI_PLAN, plan)
            save_json(RUNS / 'multiseed-preflight.json',
                      {'multiseed_plan_sha256': digest(MULTI_PLAN), 'optimizer_steps_executed': 0,
                       'encoder_fits': 30, 'rsr_fits': 40, 'verified_input_hashes': True})
            print('Prepared 30 Rank_LSTM fits and 40 RSR fits with five fixed seeds')
        else:
            if not MULTI_PLAN.exists() or read_json(RUNS / 'multiseed-preflight.json')[
                    'multiseed_plan_sha256'] != digest(MULTI_PLAN):
                raise ValueError('Complete the no-training preflight first')
            execute(plan)


if __name__ == '__main__':
    main()
