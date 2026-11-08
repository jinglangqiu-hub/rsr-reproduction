"""Verify all five-seed runs and independently recompute their ranking metrics."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
PLAN = RUNS / 'multiseed-plan.json'
FIELDS = ('market', 'relation', 'method')
METRICS = ('mse', 'mrr', 'irr')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def group_key(row):
    return tuple(row[field] for field in FIELDS)


def describe(values):
    return {'mean': statistics.mean(values), 'std': statistics.stdev(values),
            'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def verify_training(folder, run, config, plan_hash):
    saved = read_json(folder / 'config.json')
    result = read_json(folder / 'result.json')
    progress = read_json(folder / 'progress.json')
    if (saved['multiseed_plan_sha256'] != plan_hash or saved['configuration'] != config
            or saved['run'] != run or result['multiseed_plan_sha256'] != plan_hash
            or result['id'] != run['id'] or result['seed'] != run['seed']
            or progress['completed_epochs'] != 50):
        raise ValueError(f'Training provenance mismatch: {run["id"]}')
    losses = []
    for line in (folder / 'train.log').read_text().splitlines():
        if line.startswith('Valid MSE:'):
            current = float(line.split()[2])
        elif line.startswith('epoch:'):
            if int(line.split()[1]) != len(losses):
                raise ValueError(f'Incomplete or duplicated epochs: {run["id"]}')
            losses.append(current)
    if (len(losses) != 50 or np.argmin(losses) + 1 != result['best_epoch']
            or min(losses) != result['best_validation_loss']
            or progress['best_checkpoint'] != result['checkpoint']):
        raise ValueError(f'Checkpoint does not match minimum validation loss: {run["id"]}')
    for prefix in (result['checkpoint'], progress['latest_checkpoint']):
        if not all((folder / f'{prefix}{suffix}').is_file()
                   for suffix in ('.index', '.data-00000-of-00001')):
            raise ValueError(f'Checkpoint files are missing: {run["id"]}')
    return result


def evaluate_arrays(pred, target, mask, dates, tickers):
    if (pred.shape != target.shape or pred.shape != mask.shape or pred.shape[1] != len(dates)
            or not np.isfinite(pred).all() or not np.isfinite(target).all()
            or not np.isin(mask, (0, 1)).all() or np.any(mask.sum(axis=0) == 0)):
        raise ValueError('Saved prediction arrays are invalid')
    returns, ranks, picks = [], [], []
    for day in range(len(dates)):
        order = np.argsort(pred[:, day])[::-1]
        chosen = order[mask[order, day].astype(bool)][0]
        truth = np.argsort(target[:, day])[::-1]
        truth = truth[mask[truth, day].astype(bool)]
        returns.append(float(target[chosen, day]))
        ranks.append(int(np.flatnonzero(truth == chosen)[0]) + 1)
        picks.append(str(tickers[chosen]))
    metrics = {'mse': float(np.square((pred - target) * mask).sum() / mask.sum()),
               'mrr': statistics.mean(1 / rank for rank in ranks), 'irr': sum(returns)}
    daily = [{'date': str(day)[:10], 'ticker': ticker, 'return': ret, 'rank': rank}
             for day, ticker, ret, rank in zip(dates, picks, returns, ranks)]
    largest = sorted(returns, reverse=True)
    return metrics, {
        'positive_days': sum(ret > 0 for ret in returns),
        'irr_without_best_day': metrics['irr'] - largest[0],
        'irr_without_best_three_days': metrics['irr'] - sum(largest[:3]),
        'top_positive_days': sorted(daily, key=lambda day: -day['return'])[:3],
        'top_negative_days': sorted(daily, key=lambda day: day['return'])[:3],
        'most_selected_stocks': Counter(picks).most_common(3),
        'two_dates': [day for day in daily if day['date'] in ('2017-10-25', '2017-10-30')],
    }


def main():
    plan = read_json(PLAN)
    plan_hash = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    state = read_json(RUNS / 'multiseed-status.json')
    summary = read_json(RUNS / 'multiseed-summary.json')
    frozen = read_json(RUNS / 'frozen-test-manifest.json')
    prior = {group_key(row): row for row in read_json(RUNS / 'frozen-test' / 'analysis.json')['runs']}
    templates = {group_key(row['configuration']): row['configuration'] for row in frozen['runs']}
    recorded = {group_key(row): row for row in summary['groups']}
    if (state['status'] != 'completed' or len(plan['runs']) != 70
            or state['completed'] != [run['id'] for run in plan['runs']]
            or state['multiseed_plan_sha256'] != plan_hash
            or summary['multiseed_plan_sha256'] != plan_hash or summary['standard_deviation_ddof'] != 1):
        raise ValueError('The complete five-seed plan is not available')
    for name, sha in plan['code_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'Training code changed: {name}')
    markets = {}
    for market in ('NASDAQ', 'NYSE'):
        path = RUNS / 'frozen-test' / f'{market}.npz'
        if hashlib.sha256(path.read_bytes()).hexdigest() != plan['input_sha256'][str(path.relative_to(ROOT))]:
            raise ValueError(f'Reference price arrays changed: {market}')
        with np.load(path, allow_pickle=False) as data:
            markets[market] = {key: data[key] for key in ('mask', 'targets', 'dates', 'tickers')}
    results, encoders = {}, {}
    for run in plan['runs']:
        config = plan['configurations'][run['configuration']]
        folder = RUNS / run['id']
        result = verify_training(folder, run, config, plan_hash)
        if config['stage'] == 'embedding':
            for name, days in (('embedding.npy', 1008 - config['seq']), ('test_embedding.npy', 237)):
                values = np.load(folder / name, mmap_mode='r', allow_pickle=False)
                if values.shape != (config['stocks'], days, config['unit']) or values.dtype != np.float32:
                    raise ValueError(f'Encoder shape changed: {run["id"]}')
            encoders[run['id']] = result
            continue
        group = group_key(config)
        if any(config[field] != templates[group][field] for field in ('seq', 'unit', 'alpha', 'inner_prod')):
            raise ValueError(f'Hyperparameters differ from the frozen selection: {run["id"]}')
        dependency = encoders[config['embedding_run']]
        if dependency['seed'] != run['seed']:
            raise ValueError(f'Encoder and RSR seeds differ: {run["id"]}')
        data = markets[config['market']]
        row = {'id': run['id'], 'seed': run['seed'], 'best_epoch': result['best_epoch'],
               'parameters': {key: config[key] for key in ('seq', 'unit', 'alpha')}}
        with np.load(folder / 'predictions.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['dates'], data['dates'][1008:])
            np.testing.assert_array_equal(saved['tickers'], data['tickers'])
            for partition, prefix, first, last in (
                    ('validation', 'valid', 756, 1008), ('test', 'test', 1008, 1245)):
                pred, gt, mask = (saved[f'{prefix}_{field}'] for field in ('pred', 'gt', 'mask'))
                days = np.arange(first, last)
                expected_mask = np.minimum.reduce(
                    [data['mask'][:, days - shift] for shift in range(config['seq'] + 1)])
                np.testing.assert_array_equal(gt, data['targets'][:, first:last])
                np.testing.assert_array_equal(mask, expected_mask)
                metrics, diagnostics = evaluate_arrays(pred, gt, mask, data['dates'][first:last], data['tickers'])
                np.testing.assert_allclose([metrics[key] for key in METRICS],
                                           [result[partition][key] for key in METRICS], rtol=1e-10, atol=1e-12)
                row[partition] = metrics
                if partition == 'test':
                    row['diagnostics'] = diagnostics
        results.setdefault(group, []).append(row)

    groups = []
    for group, rows in results.items():
        if [row['seed'] for row in rows] != plan['seeds']:
            raise ValueError(f'The five fixed seeds are not all present: {group}')
        metrics = {partition: {metric: describe([row[partition][metric] for row in rows])
                               for metric in METRICS} for partition in ('validation', 'test')}
        for partition in metrics:
            for metric in METRICS:
                for statistic in ('mean', 'std', 'min', 'max'):
                    np.testing.assert_allclose(metrics[partition][metric][statistic],
                                               recorded[group]['metrics'][partition][metric][statistic],
                                               rtol=1e-10, atol=1e-12)
        without_three = [row['diagnostics']['irr_without_best_three_days'] for row in rows]
        events = [sum(day['return'] for day in row['diagnostics']['two_dates']) for row in rows]
        without_two_dates = [row['test']['irr'] - contribution for row, contribution in zip(rows, events)]
        groups.append(dict(zip(FIELDS, group)) | {
            'parameters': rows[0]['parameters'], 'results': rows, 'metrics': metrics,
            'positive_test_seeds': sum(row['test']['irr'] > 0 for row in rows),
            'positive_seeds_without_best_three_days': sum(value > 0 for value in without_three),
            'irr_without_best_three_days': describe(without_three),
            'irr_without_two_event_dates': describe(without_two_dates),
            'mean_event_dates_contribution': statistics.mean(events),
            'two_event_dates_ticker_TWO_picks': sum(
                day['ticker'] == 'TWO' for row in rows for day in row['diagnostics']['two_dates']),
            'previous_single_seed_test': prior[group]['test'],
        })

    indexed = {group_key(group): group for group in groups}
    paired = []
    for market in ('NASDAQ', 'NYSE'):
        for relation in ('wikidata', 'sector_industry'):
            implicit = indexed[market, relation, 'RSR_I']['results']
            explicit = indexed[market, relation, 'RSR_E']['results']
            differences = [i['test']['irr'] - e['test']['irr'] for i, e in zip(implicit, explicit)]
            paired.append({'market': market, 'relation': relation, 'comparison': 'RSR_I minus RSR_E',
                           'differences': differences, 'statistics': describe(differences),
                           'implicit_wins': sum(value > 0 for value in differences)})
    seed_losses = [{'seed': seed, 'market': market,
                    'negative_groups': [dict(zip(FIELDS, group)) for group, rows in results.items()
                                        if group[0] == market and next(
                                            row for row in rows if row['seed'] == seed)['test']['irr'] < 0]}
                   for seed in plan['seeds'] for market in ('NASDAQ', 'NYSE')]
    output = {'multiseed_plan_sha256': plan_hash, 'verified_training_runs': 70,
              'verified_rsr_predictions': 40, 'verified_partitions': 80,
              'test_days_per_run': 237, 'standard_deviation_ddof': 1, 'seeds': plan['seeds'],
              'groups': groups, 'paired_model_comparisons': paired, 'losses_by_seed_and_market': seed_losses}
    path = RUNS / 'multiseed-analysis.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)
    compact = {key: value for key, value in output.items() if key != 'groups'}
    compact['groups'] = [{key: value for key, value in group.items() if key != 'results'} for group in groups]
    print(json.dumps(compact, allow_nan=False))


if __name__ == '__main__':
    main()
