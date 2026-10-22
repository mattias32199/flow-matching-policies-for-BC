"""Recreate the five-run final figures using only the published NPZ files."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import plot as style
from plot_data import load, unified


def recreate(directory, output):
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    style.OUT = output
    features = ['handcrafted', 'standardized', 'xyz0.1', 'xyz10']
    archives = {f: load(directory / f'm4-{f}-s0.npz') for f in features}
    figures = []
    meta, runs = load(directory / 'm1-double-n10000-s0.npz')
    source = meta['experiment']; run = runs[source]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    style.setup(axes[0], 'Robot success', True)
    style.curve(axes[0], run['streams']['eval.jsonl'], 'success', '#2878B5', 'Policy success')
    for ax, metric, title in zip(axes[1:], ['flow_loss', 'action_mse'], ['Flow loss', 'Action MSE']):
        style.setup(ax, title)
        for split, color, line in [('training', '#64748B', '-'), ('validation', '#D55E00', '--')]:
            style.curve(ax, run['streams']['metrics.jsonl'], f'{split}/{metric}', color, split.capitalize(), True, line)
    figures.append(style.finish(fig, axes, 'Mystery 1 · Completed 2M-update run',
        'Block-double · fixed nominal 10K demonstrations · seed 0',
        '500 episodes per success point. Losses: raw traces + within-run 50K means; first 10K loss updates omitted. One seed.', 'mystery1'))
    for selected_features, filename, title, split in [
        (features[:2], 'mystery4_baseline', 'Mystery 4 · Handcrafted versus standardized', 'training'),
        (features, 'mystery4_scaling', 'Mystery 4 · Position scaling through 2M updates', 'training'),
        (features, 'mystery4_validation', 'Mystery 4 · Validation scores through 2M updates', 'validation')]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
        style.setup(axes[0], 'Robot success', True)
        style.setup(axes[1], f'{split.capitalize()} flow loss')
        style.setup(axes[2], f'{split.capitalize()} action MSE')
        for f in selected_features:
            meta, runs = archives[f]; color=style.COLORS[f]; label=style.LABELS[f]
            for ax, stream, key, loss in [(axes[0], 'eval.jsonl', 'success', False),
                    (axes[1], 'metrics.jsonl', f'{split}/flow_loss', True),
                    (axes[2], 'metrics.jsonl', f'{split}/action_mse', True)]:
                style.curve(ax, unified(meta, runs, stream), key, color, label, loss)
        figures.append(style.finish(fig, axes, title,
            'Block-double · streaming data · seed 0 · complete measured training history',
            'Raw traces + 50K-update means; first 10K loss updates omitted. Validation data changed during training; see report.', filename))
    figures[2].savefig(output/'curves.png', dpi=190)
    for fig in figures:
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('results/plot-data'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    recreate(args.data, args.output)
