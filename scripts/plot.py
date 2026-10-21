"""Recreate the blog comparisons from portable records, keeping resume segments separate."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
COLORS = {'handcrafted': '#2878B5', 'standardized': '#E69F00',
          'xyz0.1': '#CC79A7', 'xyz10': '#009E73',
          'fourier1': '#7B61A8', 'fourier4': '#A65E2E'}
LABELS = {'handcrafted': 'Handcrafted', 'standardized': 'Standardized',
          'xyz0.1': 'Standardized · XYZ ×0.1', 'xyz10': 'Standardized · XYZ ×10',
          'fourier1': 'Fourier: 1 frequency', 'fourier4': 'Fourier: 4 frequencies'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'bold', 'axes.labelcolor': '#334155',
                     'text.color': '#172B4D', 'axes.edgecolor': '#CBD5E1',
                     'savefig.facecolor': 'white'})


def records(name, stream):
    path = OUT / 'runs' / name / stream
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def setup(ax, title, success=False):
    ax.set_title(title, loc='left', pad=12, fontsize=12)
    ax.set_xlabel('Training updates')
    ax.set_xlim(0, 2_000_000)
    ax.set_xticks([0, 500_000, 1_000_000, 1_500_000, 2_000_000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '0' if x == 0 else f'{x/1e6:g}M'))
    ax.grid(axis='y', color='#E2E8F0', linewidth=.7)
    ax.set_axisbelow(True)
    if success:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_yticks([0, .25, .5, .75, 1])
    else:
        ax.set_yscale('log')
        ax.set_ylabel('Loss (log scale)')


def curve(ax, rows, key, color, label, loss=False, linestyle='-'):
    rows = [r for r in rows if key in r and r[key] is not None
            and (not loss or r['step'] >= 10_000)]
    if not rows:
        return None
    x = np.array([r['step'] for r in rows]); y = np.array([r[key] for r in rows])
    if not loss:
        return ax.plot(x, y, color=color, lw=2, marker='o', markersize=3,
                       label=label, linestyle=linestyle)[0]
    ax.plot(x, y, color=color, lw=.5, alpha=.15, linestyle=linestyle)
    # Nonoverlapping 50K-update means, never pooled across runs or resume boundaries.
    bins = (x - 1) // 50_000
    xs = [x[bins == b].mean() for b in np.unique(bins)]
    ys = [y[bins == b].mean() for b in np.unique(bins)]
    return ax.plot(xs, ys, color=color, lw=2, marker='o', markersize=2,
                   label=label, linestyle=linestyle)[0]


def finish(fig, axes, title, subtitle, footnote, filename):
    fig.suptitle(title, x=.065, y=.98, ha='left', fontsize=17, weight='bold')
    fig.text(.065, .90, subtitle, fontsize=10, color='#475569')
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle); labels.append(label)
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .09),
                   ncol=min(4, len(handles)), frameon=False, fontsize=9)
    fig.text(.065, .035, footnote, fontsize=8.5, color='#64748B')
    fig.subplots_adjust(left=.065, right=.975, top=.78, bottom=.29, wspace=.30)
    for extension in ['png', 'svg']:
        path = OUT / f'{filename}.{extension}'
        fig.savefig(path, dpi=190)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    return fig


def mystery1():
    name = 'm1-double-n10000-s0'
    m = records(name, 'metrics.jsonl'); e = records(name, 'eval.jsonl')
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    setup(axes[0], 'Robot success', True)
    curve(axes[0], e, 'success', '#2878B5', 'Policy success')
    for ax, metric, title in zip(axes[1:], ['flow_loss', 'action_mse'], ['Flow loss', 'Action MSE']):
        setup(ax, title)
        for split, color, style in [('training', '#64748B', '-'), ('validation', '#D55E00', '--')]:
            curve(ax, m, f'{split}/{metric}', color, split.capitalize(), True, style)
    if m and e:
        low = min(m, key=lambda r: r['validation/flow_loss'])
        axes[1].axvline(low['step'], color='#94A3B8', lw=1, ls=':')
        axes[1].text(.05, .05, f"Validation minimum: {low['step']/1000:g}K", transform=axes[1].transAxes, fontsize=9)
        axes[0].annotate(f"Final: {e[-1]['success']:.1%}", (e[-1]['step'], e[-1]['success']),
                         xytext=(-8, 14), textcoords='offset points', ha='right', weight='bold')
    return finish(fig, axes, 'Mystery 1 · Validation loss rises while control improves',
                  'Block-double · fixed nominal 10K demonstrations · seed 0 · completed at 2M updates',
                  'One training seed; 500 episodes per success point. Losses: faint raw records + 50K-update means; first 10K updates omitted from loss panels.',
                  'mystery1')


def mystery4(features, filename, title, split='training'):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    setup(axes[0], 'Robot success', True)
    setup(axes[1], f'{split.capitalize()} flow loss')
    setup(axes[2], f'{split.capitalize()} action MSE')
    for feature in features:
        name = f'm4-{feature}-s0'; color = COLORS[feature]; label = LABELS[feature]
        m = records(name, 'metrics.jsonl'); e = records(name, 'eval.jsonl')
        curve(axes[0], e, 'success', color, label)
        for ax, metric in zip(axes[1:], ['flow_loss', 'action_mse']):
            curve(ax, m, f'{split}/{metric}', color, label, True)
    return finish(fig, axes, title,
                  'Block-double · fresh streaming demonstrations · seed 0 · partial source runs; curves stop at observed data',
                  '500 episodes per success point; no seed uncertainty band. Losses: faint raw records + 50K-update means; first 10K omitted. Restarts shown separately.',
                  filename)


def continuations():
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    setup(axes[0], 'Robot success', True)
    setup(axes[1], 'Replacement-set validation flow loss')
    setup(axes[2], 'Replacement-set validation action MSE')
    count = 0
    bounds = []
    for feature in COLORS:
        name = f'm4-{feature}-s0-cont1'
        m = records(name, 'metrics.jsonl'); e = records(name, 'eval.jsonl')
        if not m:
            continue
        count += 1
        config = json.loads((OUT/'runs'/name/'config.json').read_text())
        label = LABELS[feature]; color = COLORS[feature]
        bounds.extend([config['resume_step'], m[-1]['step']])
        curve(axes[0], e, 'success', color, label)
        for ax, key in zip(axes[1:], ['validation/flow_loss', 'validation/action_mse']):
            curve(ax, m, key, color, label, True)
            ax.axvline(config['resume_step'], color=color, alpha=.4, lw=1, ls=':')
    if bounds:
        for ax in axes[1:]:
            ax.set_xlim(min(bounds) - 20_000, max(bounds) + 20_000)
            ax.set_xticks(np.linspace(min(bounds) - 20_000, max(bounds) + 20_000, 4))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1e6:.3f}M'))
    if not axes[0].lines:
        axes[0].text(.5, .48, 'No post-restart rollout\nevaluations recorded yet',
                     transform=axes[0].transAxes, ha='center', va='center', color='#64748B')
    return finish(fig, axes, 'Restarted segments · Keep the validation boundary visible',
                  f'{count} segments with training records · weights, optimizer, agent RNG and preprocessing restored',
                  'Fresh data and replacement validation sets. Dotted lines mark saved updates. No lines join old and new segments; these are not extra experiments.',
                  'continuations')


def plot():
    OUT.mkdir(exist_ok=True)
    figures = [mystery1(),
               mystery4(['handcrafted', 'standardized'], 'mystery4_baseline',
                        'Mystery 4 · Handcrafted versus standardized inputs'),
               mystery4(['handcrafted', 'standardized', 'xyz0.1', 'xyz10'], 'mystery4_scaling',
                        'Mystery 4 · Position scaling changes robot success'),
               mystery4(['handcrafted', 'standardized', 'xyz0.1', 'xyz10'], 'mystery4_validation',
                        'Mystery 4 · Held-out losses do not reliably rank control', 'validation'),
               continuations()]
    # Preserve the existing overview URL while making the primary figure readable.
    figures[2].savefig(OUT / 'curves.png', dpi=190)
    for fig in figures:
        plt.close(fig)
    print('Saved five PNG/SVG comparisons and the curves.png overview.')


if __name__ == '__main__':
    plot()
