"""Plot the verified hyperparameter-search stage with Matplotlib.

Run in the existing isolated plotting environment:
  /home/luckythrower/.venvs/rsr-plot/bin/python tools/plot_hparam_stage.py
"""

import argparse
import ast
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

from matplotlib import dates as mdates, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, PercentFormatter, ScalarFormatter
import numpy as np

from plot_public_seed import configure_style


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'runs'
OUTPUT = ROOT / 'figures' / 'hparam-stage'
MARKETS = ('NASDAQ', 'NYSE')
RELATIONS = {'wikidata': 'Wiki 企业关系', 'sector_industry': '行业关系 Industry'}
METHODS = {'RSR_I': '隐式 Implicit', 'RSR_E': '显式 Explicit'}
BLUE, GOLD, GRAY = '#246A8D', '#B87B29', '#647082'
INK, MUTED, GRID = '#192D3B', '#4B5C69', '#DDE5E9'
PERCENT = PercentFormatter(xmax=100)
RECORDS = []


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return tuple(row[name] for name in ('market', 'relation', 'method'))


def label(row):
    relation = 'Wiki' if row['relation'] == 'wikidata' else '行业 Industry'
    return f'{row["market"]} · {relation} · {METHODS[row["method"]]}'


def parameter_label(parameters):
    return f'S={parameters["seq"]}, U={parameters["unit"]}, α={parameters["alpha"]:g}'


def load_inputs():
    manifest = read_json(RUNS / 'frozen-test-manifest.json')
    analysis = read_json(RUNS / 'frozen-test' / 'analysis.json')
    if analysis['manifest_sha256'] != digest(RUNS / 'frozen-test-manifest.json'):
        raise ValueError('Test analysis differs from the frozen manifest')
    search = {key(row): row for row in read_json(RUNS / 'hparam-search-summary.json')}
    combined = {key(row): row for row in read_json(RUNS / 'hparam-combination-summary.json')}
    measured = {key(row): row for row in analysis['runs']}
    order = [(market, relation, method)
             for market in MARKETS for relation in RELATIONS for method in METHODS]
    if set(search) != set(order) or set(combined) != set(order) or set(measured) != set(order):
        raise ValueError('Expected all eight market/relation/model configurations')
    rows, daily = [], {}
    for group in order:
        row = measured[group]
        if row['id'] != combined[group]['selected']['id']:
            raise ValueError(f'Test selection mismatch: {group}')
        with np.load(RUNS / 'frozen-test' / f'{row["id"]}.predictions.npz', allow_pickle=False) as saved:
            prediction, target, mask = (saved[f'test_{name}'] for name in ('pred', 'gt', 'mask'))
            dates = [date.fromisoformat(value[:10]) for value in saved['dates']]
        if prediction.shape != target.shape or prediction.shape != mask.shape or len(dates) != 237:
            raise ValueError(f'Test shape mismatch: {group}')
        returns = []
        for day in range(len(dates)):
            ranking = np.argsort(prediction[:, day])[::-1]
            chosen = ranking[mask[ranking, day].astype(bool)][0]
            returns.append(target[chosen, day])
        returns = np.asarray(returns)
        np.testing.assert_allclose(returns.sum(), row['test']['irr'], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(returns.sum() - np.sort(returns)[-3:].sum(),
                                   row['irr_without_best_three_days'], rtol=1e-10, atol=1e-12)
        daily[group] = (dates, returns)
        rows.append(row)
    return manifest, search, combined, rows, daily


def heading(fig, identifier, title, subtitle):
    fig.text(.055, .973, f'RSR / 超参数搜索与冻结测试 / {identifier}', color=BLUE,
             fontsize=10, weight='bold', va='top')
    fig.text(.055, .929, title, fontsize=20, weight='bold', va='top')
    fig.text(.055, .875, subtitle, fontsize=10.5, color=MUTED, va='top')


def footnote(fig, text):
    fig.text(.055, .025, text, fontsize=9, color=MUTED, va='bottom', linespacing=1.5)


def save(fig, category, stem, title):
    directory = OUTPUT / category
    directory.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(directory / f'{stem}.png', dpi=240)
    fig.savefig(directory / f'{stem}.svg', metadata={'Date': None})
    RECORDS.append({'category': category, 'stem': stem, 'title': title,
                    'files': [f'{category}/{stem}.png', f'{category}/{stem}.svg']})
    plt.close(fig)


def method_legend(fig):
    handles = [Line2D([], [], color=BLUE, marker='o', label='隐式建模 Implicit Modeling (RSR_I)'),
               Line2D([], [], color=GOLD, marker='s', linestyle='--',
                      label='显式建模 Explicit Modeling (RSR_E)')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.055, .819),
               frameon=False, ncol=2)


def plot_search(search, field, values, stem, title, english):
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.6), sharey=True)
    fig.subplots_adjust(left=.09, right=.965, top=.718, bottom=.145, hspace=.55, wspace=.18)
    heading(fig, stem[:3], title, '2016 年验证集 · 固定另外两项为各市场基线 · 种子 123456789')
    method_legend(fig)
    all_irr = [candidate['validation']['irr'] * 100
               for group in search.values() for candidate in group['candidates']]
    bottom, top = min(0, min(all_irr)), max(all_irr)
    margin = (top - bottom) * .15
    for i, market in enumerate(MARKETS):
        for j, relation in enumerate(RELATIONS):
            ax = axes[i, j]
            for method, color, marker, style in (
                    ('RSR_I', BLUE, 'o', '-'), ('RSR_E', GOLD, 's', '--')):
                candidates = [candidate for candidate in search[market, relation, method]['candidates']
                              if candidate['axis'] in ('baseline', field)]
                candidates.sort(key=lambda candidate: candidate['parameters'][field])
                x = [candidate['parameters'][field] for candidate in candidates]
                y = [candidate['validation']['irr'] * 100 for candidate in candidates]
                if x != values:
                    raise ValueError(f'Unexpected {field} candidate grid: {market}, {relation}, {method}')
                ax.plot(x, y, color=color, marker=marker, linestyle=style, lw=1.7, ms=5)
                best = int(np.argmax(y))
                other_method = 'RSR_E' if method == 'RSR_I' else 'RSR_I'
                other_at_peak = next(candidate['validation']['irr'] * 100
                                     for candidate in search[market, relation, other_method]['candidates']
                                     if candidate['axis'] in ('baseline', field)
                                     and candidate['parameters'][field] == x[best])
                direction = 1 if y[best] >= other_at_peak else -1
                ax.scatter([x[best]], [y[best]], s=110, facecolors='none', edgecolors=INK, lw=1.1, zorder=5)
                ax.annotate(f'{y[best]:.2f}%', (x[best], y[best]),
                            xytext=(0, direction * 13), textcoords='offset points',
                            ha='center', va='bottom' if direction > 0 else 'top', fontsize=9)
            ax.set_xscale('log', base=10 if field == 'alpha' else 2)
            ax.set_xticks(values)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(values[0] / 1.25, values[-1] * 1.25)
            ax.set_ylim(bottom - margin, top + margin)
            ax.yaxis.set_major_formatter(PERCENT)
            ax.grid(axis='y')
            ax.axhline(0, color=GRAY, lw=.7)
            ax.set_title(f'{market} · {RELATIONS[relation]}', loc='left', fontsize=12, pad=10)
            ax.set_xlabel(english)
    for ax in axes[:, 0]:
        ax.set_ylabel('验证累计投资回报率 IRR (%)\nCumulative Investment Return Ratio')
    footnote(fig, '空心圆标出各曲线的最佳验证 IRR。横轴采用对数刻度，显示全部实际候选值。\n'
             'NASDAQ 基线 (S, U, α)=(16, 64, 0.1)，NYSE 基线为 (8, 32, 10)。每候选取最低验证总损失检查点。')
    save(fig, 'experimental-results', stem, title)


def plot_test_metrics(rows):
    fig, axes = plt.subplots(1, 3, figsize=(15, 8.8), sharey=True)
    fig.subplots_adjust(left=.285, right=.955, top=.735, bottom=.19, wspace=.21)
    heading(fig, 'E04', '冻结配置的测试集指标', '2017-01-03 至 2017-12-08 · 237 个交易日 · 固定权重前向评估')
    labels = [label(row) + '\n' + parameter_label(row['parameters']) for row in rows]
    metrics = [('mse', '均方误差 MSE', 'Mean Squared Error', 1e4),
               ('mrr', '平均倒数排名 MRR', 'Mean Reciprocal Rank', 1),
               ('irr', '累计投资回报率 IRR', 'Cumulative Investment\nReturn Ratio', 100)]
    y = np.arange(len(rows))
    for ax, (metric, chinese, english, scale) in zip(axes, metrics):
        values = np.array([row['test'][metric] * scale for row in rows])
        ax.barh(y, values, color=BLUE, height=.55)
        ax.set_yticks(y, labels)
        ax.set_ylim(len(rows) - .5, -.5)
        ax.set_title(chinese, fontsize=12, pad=12)
        ax.set_xlabel(english + ('\n数值单位 ' + r'$\times 10^{-4}$' if metric == 'mse' else ''))
        low, high = min(0, float(values.min())), float(values.max())
        span = high - low
        ax.set_xlim(low - span * (.17 if low < 0 else 0), high + span * .33)
        for position, value in zip(y, values):
            text = f'{value:.2f}%' if metric == 'irr' else f'{value:.2f}' if metric == 'mse' else f'{value:.5f}'
            ax.annotate(text, (value, position), xytext=(4 if value >= 0 else -4, 0),
                        textcoords='offset points', ha='left' if value >= 0 else 'right',
                        va='center', fontsize=9)
        ax.grid(axis='x')
        ax.axvline(0, color=GRAY, lw=.8)
        ax.axhline(3.5, color=GRID, lw=1)
        if metric == 'irr':
            ax.xaxis.set_major_formatter(PERCENT)
    footnote(fig, '单种子结果，种子为 123456789。IRR 按作者口径累加每日首选股票的简单收益率。\n'
             '参数及检查点均在测试前由验证集结果冻结。本图展示单次运行的点估计。')
    save(fig, 'experimental-results', 'E04-frozen-test-metrics', '冻结配置的测试集指标')


def plot_cumulative(daily):
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.095, right=.955, top=.718, bottom=.15, hspace=.41, wspace=.16)
    heading(fig, 'E05', '测试期逐日累计投资回报率', 'Cumulative Investment Return Ratio · 每条曲线对应一组冻结配置')
    method_legend(fig)
    curves = [100 * returns.cumsum() for _, returns in daily.values()]
    lower, upper = min(0, min(curve.min() for curve in curves)), max(curve.max() for curve in curves)
    for i, market in enumerate(MARKETS):
        for j, relation in enumerate(RELATIONS):
            ax = axes[i, j]
            for method, color, style in (('RSR_I', BLUE, '-'), ('RSR_E', GOLD, '--')):
                dates, returns = daily[market, relation, method]
                curve = 100 * returns.cumsum()
                ax.plot(dates, curve, color=color, linestyle=style, lw=1.8)
                ax.annotate(f'{curve[-1]:.2f}%', (dates[-1], curve[-1]),
                            xytext=(6, 0), textcoords='offset points', color=INK,
                            va='center', fontsize=9)
            ax.set_title(f'{market} · {RELATIONS[relation]}', loc='left', fontsize=12)
            ax.set_xticks([dates[0], date(2017, 4, 1), date(2017, 7, 1), date(2017, 10, 1), dates[-1]])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            ax.tick_params(axis='x', labelbottom=True)
            ax.set_xlim(dates[0], dates[-1] + timedelta(days=48))
            ax.set_ylim(lower - 18, upper + 18)
            ax.yaxis.set_major_formatter(PERCENT)
            ax.grid(axis='y')
            ax.axhline(0, color=GRAY, lw=.8)
            ax.set_xlabel('2017 年日期 / Date')
    for ax in axes[:, 0]:
        ax.set_ylabel('累计投资回报率 IRR (%)\nCumulative Investment Return Ratio')
    footnote(fig, '横轴采用实际交易日期。全部面板采用相同纵轴范围，右侧留白用于标注最终值。\n'
             '累计曲线为每日收益的加和。数据来自 8 组冻结模型保存的测试预测。')
    save(fig, 'experimental-results', 'E05-test-cumulative-irr', '测试期逐日累计投资回报率')


def grouped_comparison(rows, series, title, subtitle, stem, footer):
    fig, ax = plt.subplots(figsize=(13.8, 8.8))
    fig.subplots_adjust(left=.31, right=.92, top=.72, bottom=.16)
    heading(fig, stem[:3], title, subtitle)
    y = np.arange(len(rows))
    height = .72 / len(series)
    all_values = []
    for index, (name, values, color, hatch) in enumerate(series):
        values = np.asarray(values, dtype=float)
        all_values.extend(values)
        positions = y + (index - (len(series) - 1) / 2) * height
        bars = ax.barh(positions, values, height=height * .91, color=color,
                       hatch=hatch, edgecolor='white', linewidth=.4, label=name)
        for bar, value in zip(bars, values):
            ax.annotate(f'{value:.2f}%', (value, bar.get_y() + bar.get_height() / 2),
                        xytext=(5 if value >= 0 else -5, 0), textcoords='offset points',
                        ha='left' if value >= 0 else 'right', va='center', fontsize=9)
    lower, upper = min(0, min(all_values)), max(all_values)
    span = upper - lower
    ax.set_xlim(lower - span * (.18 if lower < 0 else 0), upper + span * .21)
    ax.set_yticks(y, [label(row) for row in rows])
    ax.set_ylim(len(rows) - .6, -.6)
    ax.xaxis.set_major_formatter(PERCENT)
    ax.set_xlabel('累计投资回报率 (%) / Cumulative Investment Return Ratio (IRR)')
    ax.grid(axis='x')
    ax.axvline(0, color=GRAY, lw=.8)
    fig.legend(*ax.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.055, .819),
               ncol=min(len(series), 3), frameon=False, fontsize=10)
    footnote(fig, footer)
    save(fig, 'analysis-results', stem, title)


def plot_analysis(search, combined, rows):
    grouped_comparison(rows, [
        ('验证集 Validation · 252 日', [r['validation']['irr'] * 100 for r in rows], BLUE, ''),
        ('测试集 Test · 237 日', [r['test']['irr'] * 100 for r in rows], GOLD, '//')],
        '验证期与测试期的累计回报对照', '相同冻结配置和检查点 · 两个时期的交易日数量不同',
        'A01-validation-test-irr',
        '验证期为 2016-01-04 至 2016-12-30，测试期为 2017-01-03 至 2017-12-08。\n'
        '两期 IRR 均为日收益加和。本图展示观察差异，差值同时包含时期及交易日数量的影响。')

    new = [combined[key(row)] for row in rows if combined[key(row)]['combination_run_in_batch_5']]
    grouped_comparison(new, [
        ('原最优 Prior best', [r['best_observed']['validation']['irr'] * 100 for r in new], BLUE, ''),
        ('新组合 Combination', [r['combination']['validation']['irr'] * 100 for r in new], GOLD, '//')],
        '6 组补跑与原最优配置的验证回报对照', '均取各候选最低验证总损失检查点 · 相同 252 个验证交易日',
        'A02-combination-comparison',
        '新组合采用三个单因素搜索优胜值的拼接。全部 6 组新组合的验证 IRR 低于对应原最优配置。\n'
        '数据来自第 5 批补跑汇总，原最优配置来自前四批独立搜索。')

    grouped_comparison(rows, [
        ('全测试期 Full test', [r['test']['irr'] * 100 for r in rows], BLUE, ''),
        ('剔除最佳 3 日 Exclude best 3 days', [r['irr_without_best_three_days'] * 100 for r in rows], GOLD, '//')],
        '测试收益对最佳 3 个交易日的敏感性', '全测试期 237 日，对照为各模型剔除自身最佳 3 日后的 234 日',
        'A03-return-concentration',
        '保留各模型原有预测和选股，仅在敏感性计算中去掉其实际收益最高的 3 天。\n'
        '该对照用于衡量收益集中度。主结果采用全测试期 IRR，选参使用验证集。')

    highest = []
    for row in new:
        values = []
        for line in (RUNS / row['combination']['id'] / 'train.log').read_text().splitlines():
            if line.startswith('Valid performance:'):
                values.append(ast.literal_eval(line.split(':', 1)[1])['btl'] - 1)
        if len(values) != 50:
            raise ValueError('Expected 50 validation IRR values in each combination log')
        highest.append(max(values) * 100)
    grouped_comparison(new, [
        ('原最优 Prior best', [r['best_observed']['validation']['irr'] * 100 for r in new], BLUE, ''),
        ('新组合既定检查点 Selected', [r['combination']['validation']['irr'] * 100 for r in new], GOLD, '//'),
        ('新组合日志峰值 Logged peak', highest, GRAY, '..')],
        '检查点选择规则的敏感性', '比较新组合的最低验证总损失检查点与 50 轮日志中的最高验证 IRR',
        'A04-checkpoint-rule-sensitivity',
        '灰色条为历史日志峰值，用于检查规则敏感性。正式选参依据为最低验证总损失对应的权重。\n'
        '训练产物保留最低验证总损失检查点，日志峰值对应的其他轮次权重未保留。')

    fields = ('seq', 'unit', 'alpha')
    gains = np.array([[(search[key(row)]['winners'][field]['validation']['irr']
                       - search[key(row)]['baseline']['validation']['irr']) * 100
                      for field in fields] for row in rows])
    fig, ax = plt.subplots(figsize=(12.2, 8.8))
    fig.subplots_adjust(left=.325, right=.85, top=.745, bottom=.145)
    heading(fig, 'A05', '单项搜索相对市场基线的验证回报增益', '每格显示 IRR 增益（百分点）及该组的单项优胜值')
    image = ax.imshow(gains, cmap='Blues', vmin=0, vmax=gains.max(), aspect='auto')
    ax.set_yticks(np.arange(len(rows)), [label(row) for row in rows])
    ax.set_xticks(range(3), ['序列长度 S\nSequence Length', '隐藏维度 U\nHidden Dimension',
                            '排序权重 α\nRanking Loss Weight'])
    ax.tick_params(axis='both', length=0, pad=8)
    for i, row in enumerate(rows):
        for j, field in enumerate(fields):
            winner = search[key(row)]['winners'][field]['parameters'][field]
            name = ('S', 'U', 'α')[j]
            ax.text(j, i, f'+{gains[i, j]:.2f}\n{name}={winner:g}', ha='center', va='center',
                    color='white' if gains[i, j] > gains.max() * .65 else INK, fontsize=10)
    colorbar = fig.colorbar(image, ax=ax, fraction=.045, pad=.055)
    colorbar.set_label('IRR 增益（百分点）\nIRR gain (percentage points)')
    footnote(fig, '增益为每项搜索的最佳验证 IRR 减去该组基线 IRR。最优值包括基线自身，因此增益下界为 0。\n'
             '该图描述单因素搜索结果，模型之间的数值来自同一验证期和固定随机种子。')
    save(fig, 'analysis-results', 'A05-independent-search-gains', '单项搜索相对市场基线的验证回报增益')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, help='Optional Chinese font path')
    args = parser.parse_args()
    configure_style(args.font)
    _, search, combined, rows, daily = load_inputs()
    plot_search(search, 'seq', [2, 4, 8, 16], 'E01-sequence-length-search',
                '序列长度 S 的独立搜索结果', '序列长度 / Sequence Length S（交易日，对数刻度）')
    plot_search(search, 'unit', [16, 32, 64, 128], 'E02-hidden-dimension-search',
                '隐藏维度 U 的独立搜索结果', '隐藏维度 / Hidden Dimension U（对数刻度）')
    plot_search(search, 'alpha', [.1, 1, 10], 'E03-ranking-weight-search',
                '排序权重 α 的独立搜索结果', '排序权重 / Ranking Loss Weight α（对数刻度）')
    plot_test_metrics(rows)
    plot_cumulative(daily)
    plot_analysis(search, combined, rows)
    sources = ['runs/hparam-search-summary.json', 'runs/hparam-combination-summary.json',
               'runs/frozen-test/analysis.json', 'runs/frozen-test-manifest.json']
    sources.extend(f'runs/frozen-test/{row["id"]}.predictions.npz' for row in rows)
    sources.extend(f'runs/{row["combination"]["id"]}/train.log'
                   for row in combined.values() if row['combination_run_in_batch_5'])
    manifest = {'script': str(Path(__file__).relative_to(ROOT)), 'script_sha256': digest(Path(__file__)),
                'numpy_version': np.__version__, 'matplotlib_version': matplotlib.__version__,
                'source_sha256': {name: digest(ROOT / name) for name in sources},
                'figures': RECORDS}
    (OUTPUT / 'figure-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                                                encoding='utf-8')
    print(f'Saved {len(RECORDS)} PNG and {len(RECORDS)} SVG figures to {OUTPUT}')


if __name__ == '__main__':
    main()
