"""Recreate the five-run final figures using only the published NPZ files."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import plot as style
from plot_data import load, branches


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
        (features, 'mystery4_validation', 'Mystery 4 · Validation scores across an explicit restart', 'validation')]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
        style.setup(axes[0], 'Robot success', True)
        style.setup(axes[1], f'{split.capitalize()} flow loss')
        style.setup(axes[2], f'{split.capitalize()} action MSE')
        for f in selected_features:
            meta, runs = archives[f]; color=style.COLORS[f];label=style.LABELS[f]
            for ax, stream, key, loss in [(axes[0], 'eval.jsonl', 'success', False),
                    (axes[1], 'metrics.jsonl', f'{split}/flow_loss', True),
                    (axes[2], 'metrics.jsonl', f'{split}/action_mse', True)]:
                for i, (_, rows) in enumerate(branches(meta, runs, stream)):
                    style.curve(ax, rows, key, color, label if i == 0 else '_nolegend_', loss,
                                '-' if i == 0 else '--')
                for segment, _ in branches(meta, runs, stream):
                    chosen = runs[segment]['config']
                    if 'resume_step' in chosen:
                        ax.axvline(chosen['resume_step'], color=color, alpha=.22, lw=.8, ls=':')
        figures.append(style.finish(fig, axes, title,
            'Block-double · streaming data · seed 0 · solid: original prefix; dashed: selected continuation',
            'Dotted lines mark restarts with new validation sets; no lines join segments. Raw traces + within-segment 50K means; first 10K loss updates omitted.', filename))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))
    style.setup(axes[0], 'Post-restart robot success', True)
    style.setup(axes[1], 'Replacement-set validation flow loss')
    style.setup(axes[2], 'Replacement-set validation action MSE')
    for f, (meta, runs) in archives.items():
        run=runs[meta['selected_segment']]; color=style.COLORS[f];label=style.LABELS[f]
        for ax, stream, key, loss in [(axes[0], 'eval.jsonl', 'success', False),
                (axes[1], 'metrics.jsonl', 'validation/flow_loss', True),
                (axes[2], 'metrics.jsonl', 'validation/action_mse', True)]:
            style.curve(ax, run['streams'][stream], key, color, label, loss)
    figures.append(style.finish(fig, axes, 'Selected continuation segments · Through 2M updates',
        'Original model/optimizer restored; streaming data and validation sets restarted',
        'These four segments complete four existing experiments. Original and superseded branches remain in the NPZ archives.', 'continuations'))
    figures[2].savefig(output/'curves.png', dpi=190)
    for fig in figures:
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('results/plot-data'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    recreate(args.data, args.output)
