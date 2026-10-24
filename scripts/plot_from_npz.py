"""Recreate the five-run final figures using only the published NPZ files."""
import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import plot as style
from plot_data import load, unified


def recreate_fixed_validation(directory, output):
    features = ['handcrafted', 'standardized', 'xyz0.1', 'xyz10']
    series = {}
    for f in features:
        with np.load(Path(directory) / f'm4-{f}-s0.npz', allow_pickle=False) as archive:
            if 'fixed_validation__step' not in archive:return
            series[f] = {k: archive['fixed_validation__'+k].copy() for k in
                         ['step', 'flow_loss', 'action_mse', 'flow_loss_batch_se', 'action_mse_batch_se']}
    style.OUT = Path(output)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, title in zip(axes, ['Fixed-set validation flow loss', 'Fixed-set validation action MSE']):
        style.setup(ax, title); ax.set_yscale('linear'); ax.set_ylabel('Loss')
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        ax.set_xlim(1_150_000, 2_050_000)
        ax.set_xticks([1_200_000, 1_400_000, 1_600_000, 1_800_000, 2_000_000])
    for f in features:
        selected = series[f]
        for ax, metric in zip(axes, ['flow_loss', 'action_mse']):
            ax.errorbar(selected['step'], selected[metric],
                yerr=selected[metric+'_batch_se'], color=style.COLORS[f],
                label=style.LABELS[f], marker='o', capsize=3, lw=1.8)
    style.finish(fig, axes, 'Mystery 4 · Reevaluated on one fixed validation set',
        'Retained checkpoints only · original-seed data (index 0, parallel 2500) · no training updates',
        '32 × 1024 sampled transitions per point; bars: one batch standard error. Regenerated simulator data, not historical raw arrays.', 'mystery4_validation_fixed')
    for extension in ['png', 'svg']:
        (Path(output)/('mystery4_validation.'+extension)).write_bytes((Path(output)/('mystery4_validation_fixed.'+extension)).read_bytes())
    plt.close(fig)


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
        (features, 'mystery4_scaling', 'Mystery 4 · Position scaling through 2M updates', 'training')]:
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
    recreate_fixed_validation(directory, output)
    recreate_extra_m1(directory, output)
    recreate_fourier(directory, output)


def recreate_extra_m1(directory, output):
    style.OUT = Path(output)
    for task, filename, title in [('double','mystery1_double_seeds','Mystery 1 · Double-task seeds'),
                                  ('single','mystery1_single','Mystery 1 · Completed single-task experiments')]:
        paths=sorted(Path(directory).glob(f'm1-{task}-*.npz'))
        if not paths:continue
        fig,axes=plt.subplots(1,3,figsize=(14,5.2))
        style.setup(axes[0],'Robot success',True)
        style.setup(axes[1],'Training flow loss')
        style.setup(axes[2],'Validation flow loss')
        for path,color in zip(paths,['#2878B5','#D55E00','#009E73']):
            meta,runs=load(path);cfg=runs[meta['selected_segment']]['config']
            label=f"{cfg['episodes']//1000}K demos · seed {cfg['seed']}"
            for ax,stream,key,loss in [(axes[0],'eval.jsonl','success',False),
                (axes[1],'metrics.jsonl','training/flow_loss',True),
                (axes[2],'metrics.jsonl','validation/flow_loss',True)]:
                style.curve(ax,unified(meta,runs,stream),key,color,label,loss)
        style.finish(fig,axes,title,'Fixed demonstrations · complete measured 2M-update histories',
            '500 episodes per success point. Losses: raw traces + 50K means; first 10K omitted. Dataset seeds differ across double-task seeds.',filename)
        plt.close(fig)


def recreate_fourier(directory, output):
    paths=sorted(Path(directory).glob('m4-fourier*-s0.npz'))
    if not paths:return
    style.OUT=Path(output)
    fig,axes=plt.subplots(1,3,figsize=(14,5.2))
    style.setup(axes[0],'Robot success',True)
    style.setup(axes[1],'Training flow loss')
    style.setup(axes[2],'Training action MSE')
    paths=[Path(directory)/'m4-standardized-s0.npz']+paths
    for path in paths:
        meta,runs=load(path);cfg=runs[meta['selected_segment']]['config'];f=cfg['features']
        for ax,stream,key,loss in [(axes[0],'eval.jsonl','success',False),
            (axes[1],'metrics.jsonl','training/flow_loss',True),
            (axes[2],'metrics.jsonl','training/action_mse',True)]:
            style.curve(ax,unified(meta,runs,stream),key,style.COLORS[f],style.LABELS[f],loss)
    style.finish(fig,axes,'Mystery 4 · Completed Fourier experiments',
        'Streaming block-double · seed 0 · completed 2M-update histories',
        '500 episodes per success point. Training losses: raw traces + 50K means; first 10K omitted. Validation datasets differ across restarts.', 'mystery4_fourier')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('results/plot-data'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    recreate(args.data, args.output)
