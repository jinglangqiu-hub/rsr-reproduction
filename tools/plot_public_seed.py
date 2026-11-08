"""Render the verified eight-run RSR comparison to PNG and SVG with Matplotlib.

Dependencies: NumPy 1.26.4 (matches the evaluator) and Matplotlib 3.10.8.
Run with: python tools/plot_public_seed.py
Use --font PATH to supply a Chinese font if Microsoft YaHei is unavailable.
"""

import argparse
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.dates as mdates
from matplotlib import font_manager, pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np

from compare_public_seed import MANIFEST, METRICS, ROOT, evaluate_saved_predictions


MARKETS = ('NASDAQ', 'NYSE')
RELATIONS = {'sector_industry': '行业关系', 'wikidata': 'Wikidata 企业关系'}
METHODS = {'RSR_I': '隐式', 'RSR_E': '显式'}
METHOD_COLORS = {'RSR_I': '#006A70', 'RSR_E': '#945100'}
PAPER = '#465875'
LOCAL = '#006A70'
INK = '#192D3B'
MUTED = '#4B5C69'
GRID = '#DDE5E9'
OUT = ROOT / 'figures' / 'public-seed'
FIGURE_DAYS = (206, 209)


def read_inputs():
    report = json.loads((ROOT / 'audit/public-seed-comparison.json').read_text(encoding='utf-8'))
    manifest = MANIFEST.read_bytes()
    if not report['prediction_metrics_rechecked'] or report['manifest_sha256'] != hashlib.sha256(manifest).hexdigest():
        raise ValueError('The comparison must be verified against the current experiment manifest.')

    plan = json.loads(manifest)
    rows = {(row['market'], row['relation'], row['method']): row for row in report['comparisons']}
    expected = {(market, relation, method)
                for market in MARKETS for relation in RELATIONS for method in METHODS}
    if rows.keys() != expected or len(report['comparisons']) != len(expected):
        raise ValueError('Expected all eight unique public-seed configurations.')
    if {row['seed'] for row in rows.values()} != {plan['public_seed']}:
        raise ValueError('Unexpected seed in the comparison.')
    if report['paper_repetitions'] != plan['paper_repeat_count']:
        raise ValueError('The reported paper repetition count differs from the manifest.')

    runs = {run['id']: run for run in plan['runs'] if run['stage'] == 'author_seed'}
    daily = {}
    for method in METHODS:
        row = rows['NYSE', 'sector_industry', method]
        run = runs[row['id']]
        config = plan['configurations'][run['configuration']]
        metrics, diagnostics = evaluate_saved_predictions(
            ROOT / 'runs' / run['id'], config, include_daily_picks=True
        )
        np.testing.assert_allclose(
            [metrics[key] for key in METRICS], [row['test'][key]['local'] for key in METRICS],
            rtol=1e-10, atol=1e-12, err_msg=f"Metrics differ for {row['id']}"
        )
        if diagnostics['paper_figure_6_days'] != row['diagnostics']['paper_figure_6_days']:
            raise ValueError(f"Selected event days differ for {row['id']}")
        daily[method] = diagnostics['daily_picks']
    return report, rows, daily


def configure_style(font):
    if font is None:
        font = next((path for path in (Path('C:/Windows/Fonts/msyh.ttc'),
                                      Path('/mnt/c/Windows/Fonts/msyh.ttc')) if path.is_file()), None)
    if font is not None:
        font_manager.fontManager.addfont(font)
        family = font_manager.FontProperties(fname=font).get_name()
    else:
        family = 'Noto Sans CJK SC'
        font_manager.findfont(family, fallback_to_default=False)
    plt.rcParams.update({
        'font.family': family, 'font.size': 10, 'text.color': INK,
        'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.axisbelow': True, 'axes.unicode_minus': False,
        'xtick.color': MUTED, 'ytick.color': MUTED, 'grid.color': GRID,
        'grid.linewidth': 0.65, 'figure.facecolor': 'white', 'svg.fonttype': 'path',
    })


def heading(fig, number, title, subtitle):
    fig.text(0.055, 0.956, f'RSR  /  公开种子复现实验  /  {number}', color=LOCAL,
             fontsize=10, va='top', weight='bold')
    fig.text(0.055, 0.909, title, fontsize=21, va='top', weight='bold')
    fig.text(0.055, 0.844, subtitle, fontsize=10.5, va='top', color=MUTED)


def note(fig, lines):
    fig.text(0.055, 0.033, '\n'.join(lines), fontsize=9, color=MUTED,
             linespacing=1.55, va='bottom')


def save(fig, stem):
    fig.savefig(OUT / f'{stem}.png', dpi=240)
    fig.savefig(OUT / f'{stem}.svg', metadata={'Date': None})
    plt.close(fig)


def plot_metrics(rows, repetitions):
    ordered = [rows[market, relation, method]
               for market in MARKETS for relation in RELATIONS for method in METHODS]
    labels = [
        f"{row['market']}  ·  {'行业' if row['relation'] == 'sector_industry' else 'Wiki'}"
        f"  ·  {METHODS[row['method']]}"
        f"{' *' if row['parameter_provenance'] == 'author_readme' else ''}"
        for row in ordered
    ]
    specs = (
        ('mse', '均方误差 (MSE)', r'Mean Squared Error  ·  $\times 10^{-4}$', 1e4, (2.05, 4.15)),
        ('mrr', '平均倒数排名 (MRR)', 'Mean Reciprocal Rank', 1, (0.021, 0.057)),
        ('irr', '累计投资回报率 (IRR)', 'Cumulative Investment Return Ratio', 100, (-30, 190)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 6.8), sharey=True)
    fig.subplots_adjust(left=0.225, right=0.956, top=0.674, bottom=0.17, wspace=0.17)
    seed = ordered[0]['seed']
    heading(fig, '01', '8 组模型的测试集指标对照',
            f'论文表 6（行业）与表 7（Wiki）的 {repetitions} 次运行均值，对比本地单次运行（种子 {seed}）')
    fig.legend(handles=[
        Line2D([], [], marker='o', linestyle='-', color=PAPER, label='论文均值及原表 ± 值'),
        Line2D([], [], marker='D', linestyle='none', color=LOCAL, label='本地单次运行'),
    ], loc='center left', bbox_to_anchor=(0.05, 0.773), frameon=False, ncol=2)

    y = np.arange(len(ordered))
    for ax, (metric, title, english, scale, limits) in zip(axes, specs):
        records = [row['test'][metric] for row in ordered]
        mean = [record['mean'] * scale for record in records]
        spread = [record['reported_spread'] * scale for record in records]
        local = [record['local'] * scale for record in records]
        ax.errorbar(mean, y - 0.14, xerr=spread, fmt='o', color=PAPER, markersize=5,
                    elinewidth=1.2, capsize=3, zorder=3)
        ax.scatter(local, y + 0.14, marker='D', s=32, color=LOCAL, zorder=4)
        ax.set_xlim(*limits)
        ax.set_yticks(y, labels)
        ax.tick_params(axis='y', length=0, pad=8)
        ax.set_ylim(len(y) - 0.5, -0.5)
        ax.grid(axis='x')
        ax.axhline(3.5, color='#ADBCC5', lw=0.9)
        ax.text(0, 1.093, title, transform=ax.transAxes, weight='bold', fontsize=12)
        ax.text(0, 1.038, english, transform=ax.transAxes, color=MUTED, fontsize=8.8)
        if metric == 'irr':
            ax.xaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax.axvline(0, color='#778997', lw=0.8)
    mse_gap = max(abs(row['test']['mse']['difference']) / row['test']['mse']['mean'] for row in ordered)
    note(fig, [f'8 组 MSE 相对论文均值的最大绝对偏差为 {mse_gap:.2%}。误差线按论文原表 ± 数值绘制。',
               '* 使用作者 README 明确公开的配置。其余 6 组沿用同市场的参数模板。'])
    save(fig, '01-test-metrics')


def plot_irr_order(rows):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.9), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.948, top=0.692, bottom=0.151, wspace=0.22, hspace=0.60)
    heading(fig, '02', '行业关系的 IRR 排序一致，Wiki 关系的排序在两市均反转',
            '累计投资回报率（Cumulative Investment Return Ratio, IRR）')
    fig.legend(handles=[
        Line2D([], [], marker='o', color=METHOD_COLORS['RSR_I'], label='隐式建模 (RSR_I)'),
        Line2D([], [], marker='s', linestyle='--', color=METHOD_COLORS['RSR_E'], label='显式建模 (RSR_E)'),
    ], loc='center left', bbox_to_anchor=(0.05, 0.771), frameon=False, ncol=2)

    for row_index, market in enumerate(MARKETS):
        for col_index, relation in enumerate(RELATIONS):
            ax = axes[row_index, col_index]
            ax.set_title(f'{market}  ·  {RELATIONS[relation]}', loc='left', fontsize=12,
                         weight='bold', pad=8)
            values = {method: [rows[market, relation, method]['test']['irr'][key] * 100
                               for key in ('mean', 'local')] for method in METHODS}
            for method, endpoints in values.items():
                color = METHOD_COLORS[method]
                implicit = method == 'RSR_I'
                other = values['RSR_E' if implicit else 'RSR_I']
                ax.plot([0, 1], endpoints, marker='o' if implicit else 's', markersize=4.2,
                        linestyle='-' if implicit else '--', color=color, lw=1.8, zorder=3)
                for x, value in enumerate(endpoints):
                    offset = 4 if value >= other[x] else -4
                    ax.annotate(f'{value:.2f}%', (x, value), xytext=(0, offset),
                                textcoords='offset points', ha='center',
                                va='bottom' if offset > 0 else 'top', color=color, fontsize=9)
            ax.set_xlim(-0.26, 1.26)
            ax.set_ylim(-27, 180)
            ax.set_xticks([0, 1], ['论文均值', '本地单次运行'])
            ax.set_yticks([0, 50, 100, 150])
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax.tick_params(axis='both', length=0, labelbottom=True)
            ax.grid(axis='y')
            ax.axhline(0, color='#778997', lw=0.8)
    note(fig, ['连线连接同一配置的论文均值与本地结果。交叉表示隐式与显式建模的 IRR 排序反转。',
               '来源为论文表 6、表 7 与本地公开种子实验。'])
    save(fig, '02-irr-order')


def plot_daily_irr(daily):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.8), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.96, top=0.682, bottom=0.2, wspace=0.15)
    implicit_picks = daily['RSR_I']
    events = [implicit_picks[day - 1] for day in FIGURE_DAYS]
    event_note = '、'.join(f"{pick['date'][5:]}（{pick['daily_return']:+.2%}）" for pick in events)
    heading(fig, '03', 'NYSE 行业关系模型的累计回报集中在两个交易日',
            f'测试集共 {len(implicit_picks)} 个交易日。两种模型均在 {event_note} 选中 TWO。')
    fig.legend(handles=[
        Line2D([], [], color=LOCAL, lw=2, label='原始累计回报'),
        Line2D([], [], color=PAPER, lw=1.7, linestyle='--', label='仅将两天的收益置零'),
    ], loc='center left', bbox_to_anchor=(0.05, 0.767), frameon=False, ncol=2)

    for ax, (method, picks) in zip(axes, daily.items()):
        dates = [date.fromisoformat(pick['date']) for pick in picks]
        returns = np.array([pick['daily_return'] for pick in picks]) * 100
        event_mask = np.isin([pick['day'] for pick in picks], FIGURE_DAYS)
        actual = returns.cumsum()
        sensitivity = np.where(event_mask, 0, returns).cumsum()
        event_dates = [day for day, is_event in zip(dates, event_mask) if is_event]
        ax.axvspan(event_dates[0] - timedelta(days=1), event_dates[-1] + timedelta(days=1),
                   color='#DAE9EB', zorder=0)
        ax.plot(dates, actual, color=LOCAL, lw=1.9, zorder=4)
        ax.plot(dates, sensitivity, color=PAPER, lw=1.6, ls='--', zorder=3)
        ax.scatter(event_dates, actual[event_mask], color=LOCAL, edgecolor='white',
                   linewidth=0.9, s=31, zorder=5)
        for series, color in ((actual, LOCAL), (sensitivity, PAPER)):
            ax.scatter(dates[-1], series[-1], color=color, s=20, zorder=5)
            ax.annotate(f'{series[-1]:.2f}%', (dates[-1], series[-1]), xytext=(6, 0),
                        textcoords='offset points', va='center', color=color, weight='bold')
        ax.set_title(f'{METHODS[method]}建模 ({method})', loc='left', fontsize=12, weight='bold')
        ax.set_xlim(dates[0] - timedelta(days=4), dates[-1] + timedelta(days=62))
        ax.set_xticks([dates[0], date(dates[0].year, 4, 1), date(dates[0].year, 7, 1),
                       date(dates[0].year, 10, 1), dates[-1]])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
        ax.set_xlabel(f'{dates[0].year} 年测试期日期')
        ax.grid(axis='y')
        ax.axhline(0, color='#778997', lw=0.8)
    axes[0].set_ylabel('累计投资回报率 (IRR)')
    note(fig, ['按论文口径对每日首选股票的简单收益率求和。阴影标示 10-25 至 10-30。',
               '虚线保留每日选股结果，仅把标记的两天收益置零，用于衡量累计回报对这两天的敏感度。',
               '来源为本地保存的测试集预测、实际收益与可交易掩码，交易日对应论文图 6。'])
    save(fig, '03-nyse-industry-daily-irr')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--font', type=Path, help='Path to a Chinese .ttf or .ttc font')
    args = parser.parse_args()
    configure_style(args.font)
    report, rows, daily = read_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    plot_metrics(rows, report['paper_repetitions'])
    plot_irr_order(rows)
    plot_daily_irr(daily)
    print(f'Saved 3 PNG and 3 SVG figures to {OUT}')


if __name__ == '__main__':
    main()
