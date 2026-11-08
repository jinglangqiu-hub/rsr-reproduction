"""Render the independently verified five-seed results with Matplotlib."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

from matplotlib import colors, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from plot_public_seed import configure_style


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'runs' / 'multiseed-analysis.json'
OUTPUT = ROOT / 'figures' / 'multiseed'
BLUE, GOLD, GRAY = '#246A8D', '#B87B29', '#77838C'
INK, MUTED, GRID = '#192D3B', '#4B5C69', '#DDE5E9'
RELATIONS = {'wikidata': 'Wiki', 'sector_industry': '行业 Industry'}
METHODS = {'RSR_I': '隐式 Implicit', 'RSR_E': '显式 Explicit'}


def label(row):
    return f'{row["market"]} · {RELATIONS[row["relation"]]} · {METHODS[row["method"]]}'


def header(fig, number, title, subtitle):
    fig.text(.055, .968, f'RSR / 冻结参数五种子实验 / {number}', color=BLUE,
             fontsize=10, weight='bold', va='top')
    fig.text(.055, .92, title, fontsize=20, weight='bold', va='top')
    fig.text(.055, .865, subtitle, fontsize=10.5, color=MUTED, va='top')


def footer(fig, text):
    fig.text(.055, .025, text, fontsize=9, color=MUTED, va='bottom', linespacing=1.5)


def save(fig, stem):
    fig.canvas.draw()
    fig.savefig(OUTPUT / f'{stem}.png', dpi=240)
    fig.savefig(OUTPUT / f'{stem}.svg', metadata={'Date': None})
    plt.close(fig)


def plot_metrics(groups):
    fig, axes = plt.subplots(1, 3, figsize=(15.3, 8.8), sharey=True)
    fig.subplots_adjust(left=.245, right=.95, top=.71, bottom=.18, wspace=.2)
    header(fig, '01', '五随机种子的测试指标与波动', '测试期 2017-01-03 至 2017-12-08 · 每种子 237 个交易日')
    handles = [
        Line2D([], [], color=BLUE, marker='D', label='五种子均值及 ±1 样本标准差'),
        Line2D([], [], color=GRAY, marker='o', linestyle='none', label='本轮单次运行'),
        Line2D([], [], color=GOLD, marker='s', markerfacecolor='none', linestyle='none',
               label='历史调参种子 123456789'),
    ]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.055, .815), frameon=False, ncol=3)
    y = np.arange(len(groups))
    specs = [('mse', '均方误差 MSE', 'Mean Squared Error\n数值单位 × 10^(-4)', 1e4),
             ('mrr', '平均倒数排名 MRR', 'Mean Reciprocal Rank', 1),
             ('irr', '累计投资回报率 IRR', 'Cumulative Investment\nReturn Ratio', 100)]
    for ax, (metric, cn, en, scale) in zip(axes, specs):
        means = np.array([g['metrics']['test'][metric]['mean'] * scale for g in groups])
        std = np.array([g['metrics']['test'][metric]['std'] * scale for g in groups])
        individual = np.array([[r['test'][metric] * scale for r in g['results']] for g in groups])
        prior = np.array([g['previous_single_seed_test'][metric] * scale for g in groups])
        for seed_index, jitter in enumerate(np.linspace(-.17, .17, 5)):
            ax.scatter(individual[:, seed_index], y + jitter, s=17, color=GRAY, alpha=.8, zorder=2)
        ax.errorbar(means, y, xerr=std, fmt='D', color=BLUE, ms=5, capsize=3, lw=1.3, zorder=4)
        ax.scatter(prior, y, facecolors='none', edgecolors=GOLD, marker='s', s=52, lw=1.4, zorder=5)
        values = np.concatenate((individual.ravel(), prior, means - std, means + std))
        lower, upper = min(0, values.min()), values.max()
        pad = (upper - lower) * .1
        ax.set_xlim(lower - pad if lower < 0 else 0, upper + pad)
        ax.set_ylim(len(groups) - .5, -.5)
        ax.set_yticks(y, [label(g) for g in groups])
        ax.set_title(cn, fontsize=12, pad=12)
        ax.set_xlabel(en)
        ax.grid(axis='x')
        ax.axvline(0, color=GRAY, lw=.75)
        ax.axhline(3.5, color=GRID, lw=1)
        if metric == 'irr':
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    footer(fig, '误差线为 5 次运行的样本标准差（ddof=1），用于描述种子间波动。灰点展示全部 5 次观测。\n'
            '各组固定上一阶段的最优参数，随机性同时覆盖 Rank_LSTM 和 RSR。历史种子作为调参结果参考。')
    save(fig, '01-test-metrics-mean-std')


def plot_seed_matrix(groups, seeds):
    values = np.array([[r['test']['irr'] * 100 for r in g['results']] for g in groups])
    fig, ax = plt.subplots(figsize=(13.6, 8.8))
    fig.subplots_adjust(left=.29, right=.86, top=.74, bottom=.17)
    header(fig, '02', '逐种子的测试累计投资回报率', 'Cumulative Investment Return Ratio (IRR) · 每格对应一次独立训练结果')
    cmap = colors.LinearSegmentedColormap.from_list('seed-irr', [GOLD, '#F4F6F7', BLUE])
    norm = colors.TwoSlopeNorm(vmin=values.min(), vcenter=0, vmax=values.max())
    image = ax.imshow(values, norm=norm, cmap=cmap, aspect='auto')
    ax.set_yticks(np.arange(len(groups)), [label(g) for g in groups])
    ax.set_xticks(range(5), [f'S{i + 1}\n{seed}' for i, seed in enumerate(seeds)])
    ax.set_xlabel('随机种子 / Random Seed', labelpad=10)
    ax.tick_params(axis='both', length=0, pad=8)
    for i in range(len(groups)):
        for j in range(5):
            rgb = cmap(norm(values[i, j]))[:3]
            luminance = .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2]
            ax.text(j, i, f'{values[i, j]:.2f}%', ha='center', va='center',
                    color='white' if luminance < .52 else INK, fontsize=10)
    colorbar = fig.colorbar(image, ax=ax, fraction=.04, pad=.045)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    colorbar.set_label('累计投资回报率 (%)\nCumulative Investment Return Ratio')
    footer(fig, '同一列使用同一个种子。NASDAQ 的全部 4 组在种子 1846380228 下出现负 IRR。\n'
            'IRR 按每日简单收益率加和，因此该指标可低于 -100%。本轮全部 40 次测试均列于图中。')
    save(fig, '02-test-irr-by-seed')


def plot_event_sensitivity(groups):
    groups = [g for g in groups if g['market'] == 'NYSE']
    fig, ax = plt.subplots(figsize=(13.6, 7.8))
    fig.subplots_adjust(left=.29, right=.91, top=.68, bottom=.2)
    header(fig, '03', 'NYSE 对两个固定大涨日期的敏感性', '比较完整测试期 237 日与去掉 2017-10-25、2017-10-30 后的 235 日')
    y = np.arange(len(groups))
    all_values = []
    for offset, field, name, color, marker in (
            (-.16, 'metrics', '完整测试期 Full test', BLUE, 'o'),
            (.16, 'irr_without_two_event_dates', '剔除两个日期 Exclude two dates', GOLD, 's')):
        metrics = [g['metrics']['test']['irr'] if field == 'metrics' else g[field] for g in groups]
        mean, std = (np.array([m[key] * 100 for m in metrics]) for key in ('mean', 'std'))
        all_values.extend(mean - std)
        all_values.extend(mean + std)
        ax.errorbar(mean, y + offset, xerr=std, fmt=marker, color=color,
                    capsize=4, ms=6, lw=1.5, label=name)
        for position, value, spread in zip(y + offset, mean, std):
            ax.annotate(f'{value:.2f}% ± {spread:.2f}', (value + spread, position), xytext=(7, 0),
                        textcoords='offset points', va='center', fontsize=9)
    lo, hi = min(all_values), max(all_values)
    span = hi - lo
    ax.set_xlim(min(0, lo) - span * .08, hi + span * .38)
    ax.set_ylim(len(groups) - .5, -.5)
    ax.set_yticks(y, [label(g) for g in groups])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_xlabel('累计投资回报率 (%) / Cumulative Investment Return Ratio (IRR)')
    ax.axvline(0, color=GRAY, lw=.8)
    ax.grid(axis='x')
    fig.legend(*ax.get_legend_handles_labels(), loc='upper left', bbox_to_anchor=(.055, .79),
               frameon=False, ncol=2)
    footer(fig, '两种口径保留相同的模型与每日选股，只在敏感性计算中去掉两个固定交易日。误差线为样本标准差。\n'
            '行业隐式模型在全部 5 个种子中都在这两天选中 TWO，两日对每次运行合计贡献 126.61 个百分点。')
    save(fig, '03-event-day-sensitivity')


def main():
    report = json.loads(SOURCE.read_text(encoding='utf-8'))
    indexed = {(g['market'], g['relation'], g['method']): g for g in report['groups']}
    order = [(market, relation, method) for market in ('NASDAQ', 'NYSE')
             for relation in ('wikidata', 'sector_industry') for method in ('RSR_I', 'RSR_E')]
    groups = [indexed[group] for group in order]
    configure_style(None)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plot_metrics(groups)
    plot_seed_matrix(groups, report['seeds'])
    plot_event_sensitivity(groups)
    (OUTPUT / 'figure-manifest.json').write_text(json.dumps({
        'analysis_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'figures': ['01-test-metrics-mean-std', '02-test-irr-by-seed', '03-event-day-sensitivity'],
        'formats': ['png', 'svg'], 'png_dpi': 240,
    }, indent=2) + '\n', encoding='utf-8')
    print('Saved 3 PNG and 3 SVG charts to', OUTPUT)


if __name__ == '__main__':
    main()
