"""Plot full reproduction curves; pilots are excluded from scientific figures."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def plot():
    data = defaultdict(lambda: defaultdict(dict))
    with (ROOT / 'results/curves.csv').open() as file:
        for row in csv.DictReader(file):
            if row['run'].startswith('pilot-'):
                continue
            for key, value in row.items():
                if key not in ('run', 'step') and value:
                    data[row['run']][key][int(row['step'])] = float(value)
    if not data:
        return
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row_idx, mystery in enumerate(['m1', 'm4']):
        groups = defaultdict(list)
        for run in data:
            if run.startswith(mystery + '-'):
                # A fresh validation set makes continuation segments separate curves.
                group = run if '-cont' in run else run.rsplit('-s', 1)[0]
                groups[group].append(run)
        for col, (metric, title) in enumerate([
            ('evaluation/success', 'Rollout success'),
            ('validation/flow_loss', 'Validation flow loss'),
            ('validation/action_mse', 'Validation action MSE'),
        ]):
            ax = axes[row_idx, col]
            for group, runs in sorted(groups.items()):
                by_step = defaultdict(list)
                for run in runs:
                    for step, value in data[run].get(metric, {}).items():
                        by_step[step].append(value)
                if not by_step:
                    continue
                steps = np.array(sorted(by_step))
                mean = np.array([np.mean(by_step[step]) for step in steps])
                std = np.array([np.std(by_step[step], ddof=1) if len(by_step[step]) > 1 else 0
                                for step in steps])
                line, = ax.plot(steps, mean, label=group)
                ax.fill_between(steps, mean - std, mean + std, color=line.get_color(), alpha=.15)
            ax.set(title=f'{mystery.upper()}: {title}', xlabel='Gradient updates')
            ax.grid(alpha=.2)
            if ax.lines:
                ax.legend(fontsize=7)
            else:
                ax.text(.5, .5, 'Awaiting metrics', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
    figure.suptitle('Partial OCBench reproduction curves (mean ± sample SD across available seeds)')
    figure.savefig(ROOT / 'results/curves.png', dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    plot()
