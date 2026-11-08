"""Independently verify frozen-test metrics and report daily return concentration."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs' / 'frozen-test'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    manifest_path = ROOT / 'runs' / 'frozen-test-manifest.json'
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = read_json(manifest_path)
    summary = read_json(OUTPUT / 'summary.json')
    if (summary['manifest_sha256'] != manifest_hash or summary['optimizer_steps_executed'] != 0
            or len(summary['runs']) != 8):
        raise ValueError('Test summary does not match the eight frozen configurations')
    rows = []
    for result, selected in zip(summary['runs'], manifest['runs']):
        run_id = result['id']
        if run_id != selected['id'] or result['configuration'] != selected['configuration']:
            raise ValueError(f'Test configuration differs from the frozen selection: {run_id}')
        with np.load(OUTPUT / f'{run_id}.predictions.npz', allow_pickle=False) as source:
            pred, gt, mask = (source[f'test_{key}'] for key in ('pred', 'gt', 'mask'))
            dates, tickers = source['dates'], source['tickers']
        if (pred.shape != gt.shape or pred.shape != mask.shape or pred.shape[1] != 237
                or dates[0][:10] != '2017-01-03' or dates[-1][:10] != '2017-12-08'
                or not np.isfinite(pred).all() or not np.isfinite(gt).all()
                or not np.isin(mask, (0, 1)).all()):
            raise ValueError(f'Invalid saved test predictions: {run_id}')
        daily = []
        for day in range(len(dates)):
            ranking = np.argsort(pred[:, day])[::-1]
            valid = ranking[mask[ranking, day].astype(bool)]
            truth = np.argsort(gt[:, day])[::-1]
            truth = truth[mask[truth, day].astype(bool)]
            if not len(valid):
                raise ValueError(f'No eligible stocks: {run_id}, {dates[day]}')
            chosen = valid[0]
            daily.append({'date': dates[day][:10], 'ticker': str(tickers[chosen]),
                          'return': float(gt[chosen, day]),
                          'rank': int(np.flatnonzero(truth == chosen)[0]) + 1})
        returns = np.array([day['return'] for day in daily])
        metrics = {'mse': float(np.square((pred - gt) * mask).sum() / mask.sum()),
                   'mrr': float(np.mean([1 / day['rank'] for day in daily])),
                   'irr': float(returns.sum())}
        np.testing.assert_allclose([metrics[key] for key in metrics],
                                   [result['test'][key] for key in metrics],
                                   rtol=1e-10, atol=1e-12)
        config = result['configuration']
        rows.append({'id': run_id, 'market': config['market'], 'relation': config['relation'],
                     'method': config['method'],
                     'parameters': {key: config[key] for key in ('seq', 'unit', 'alpha')},
                     'best_epoch': result['best_epoch'], 'validation': result['validation'],
                     'test': metrics, 'positive_days': int((returns > 0).sum()),
                     'top_positive_days': sorted(daily, key=lambda day: -day['return'])[:3],
                     'top_negative_days': sorted(daily, key=lambda day: day['return'])[:3],
                     'irr_without_best_day': float(returns.sum() - returns.max()),
                     'irr_without_best_three_days': float(returns.sum() - np.sort(returns)[-3:].sum())})
    output = {'manifest_sha256': manifest_hash, 'runs': rows, 'verified_predictions': 8,
              'trading_days_per_run': 237, 'optimizer_steps_executed': 0}
    path = OUTPUT / 'analysis.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)
    print(json.dumps(output, allow_nan=False))


if __name__ == '__main__':
    main()
