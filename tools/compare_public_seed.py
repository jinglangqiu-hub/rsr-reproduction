"""Compare the two public-seed RSR runs with the local paper's reported rows."""

import hashlib
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'experiments' / 'manifest.json'
METRICS = ('mse', 'mrr', 'irr')


def paper_row(text, table, market, method):
    section = text.split(f'Table {table}. Performance comparison among relational ranking methods', 1)[1]
    row = re.search(rf'^{re.escape(method)}\s+(.+)$', section, re.MULTILINE)[1]
    pairs = re.findall(r'([\d.eE+-]+)\s*±\s*([\d.eE+-]+)', row)
    if len(pairs) != 6:
        raise ValueError(f'Unexpected paper row for Table {table}')
    offset = 0 if market == 'NASDAQ' else 3
    return {key: {'mean': float(mean), 'reported_spread': float(spread)}
            for key, (mean, spread) in zip(METRICS, pairs[offset:offset + 3])}


def evaluate_saved_predictions(folder, config, *, include_daily_picks=False):
    market = config['market']
    data = ROOT / 'compat' / 'data'
    tickers = (data / config['ticker_file']).read_text().splitlines()
    dates = (data / f'{market}_aver_line_dates.csv').read_text().splitlines()[29:]
    if market == 'NASDAQ':
        dates = dates[:-1]
    dates = dates[1008:]
    with np.load(folder / 'predictions.npz', allow_pickle=False) as source:
        prediction, target, mask = (source[f'test_{key}'] for key in ('pred', 'gt', 'mask'))
    assert prediction.shape == target.shape == mask.shape == (len(tickers), len(dates))
    picks = []
    for index, date in enumerate(dates):
        predicted_order = np.argsort(prediction[:, index])
        top = next(stock for stock in predicted_order[::-1] if mask[stock, index] >= 0.5)
        realized_order = np.argsort(target[:, index])[::-1]
        valid_order = realized_order[mask[realized_order, index] >= 0.5]
        rank = int(np.flatnonzero(valid_order == top)[0]) + 1
        picks.append({'day': index + 1, 'date': date[:10], 'ticker': tickers[top],
                      'realized_rank': rank, 'daily_return': float(target[top, index])})
    returns = np.array([pick['daily_return'] for pick in picks])
    metrics = {'mse': float(np.sum(((prediction - target) * mask) ** 2) / mask.sum()),
               'mrr': float(np.mean([1 / pick['realized_rank'] for pick in picks])),
               'irr': float(returns.sum())}
    diagnostics = {
        'trading_days': len(picks), 'positive_days': int((returns > 0).sum()),
        'top_positive_days': sorted(picks, key=lambda item: -item['daily_return'])[:3],
        'top_negative_days': sorted(picks, key=lambda item: item['daily_return'])[:3],
        'paper_figure_6_days': [picks[index - 1] for index in (206, 209)] if market == 'NYSE' else [],
    }
    if include_daily_picks:
        diagnostics['daily_picks'] = picks
    return metrics, diagnostics


def main():
    manifest_bytes = MANIFEST.read_bytes()
    plan = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    paper = (ROOT / 'audit' / 'paper.txt').read_text(encoding='utf-8')
    comparisons = []
    for run in plan['runs']:
        if run['stage'] != 'author_seed':
            continue
        config = plan['configurations'][run['configuration']]
        folder = ROOT / 'runs' / run['id']
        result = json.loads((folder / 'result.json').read_text())
        assert result['manifest_sha256'] == manifest_hash and result['id'] == run['id']
        recomputed, diagnostics = evaluate_saved_predictions(folder, config)
        np.testing.assert_allclose([recomputed[key] for key in METRICS],
                                   [result['test'][key] for key in METRICS], rtol=1e-10, atol=1e-12)
        table = 7 if config['relation'] == 'wikidata' else 6
        baseline = paper_row(paper, table, config['market'], config['method'])
        comparison = {}
        for key, values in baseline.items():
            difference = result['test'][key] - values['mean']
            comparison[key] = {**values, 'local': result['test'][key], 'difference': difference,
                               'difference_over_reported_spread': difference / values['reported_spread']}
        comparisons.append({'id': run['id'], 'seed': run['seed'], 'market': config['market'],
                            'relation': config['relation'], 'method': config['method'], 'paper_table': table,
                            'parameter_provenance': config['parameter_provenance'],
                            'best_epoch': result['best_epoch'], 'test': comparison, 'diagnostics': diagnostics})
    report = {'manifest_sha256': manifest_hash, 'prediction_metrics_rechecked': True,
              'paper_repetitions': plan['paper_repeat_count'], 'comparisons': comparisons}
    output = ROOT / 'audit' / 'public-seed-comparison.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
