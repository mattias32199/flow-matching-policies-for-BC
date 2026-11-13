"""Figures + summary tables for the visual-BC study (visual-block-single) and the block-double
data-regime / transfer study. Reads runs/*/eval.jsonl and results/diagnostics/*; writes results/vision/.

  .venv/bin/python scripts/report_vision.py
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'vision'
DIAG = ROOT / 'results' / 'diagnostics'
# Categorical slots in fixed order (validated default palette, light mode).
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID = '#1f1f1e', '#6b6a63', '#e6e5e0'

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'savefig.dpi': 160, 'savefig.bbox': 'tight'})


def evals(run):
    p = ROOT / 'runs' / run / 'eval.jsonl'
    if not p.exists():
        return []
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return [(r['step'], r.get('success', 0.0)) for r in rows]


def final(run, n=3):
    """Mean of the last n evaluations (checkpoint noise is about +-0.05)."""
    e = evals(run)
    return dict(run=run, final=float(np.mean([s for _, s in e[-n:]])) if e else None,
                best=max((s for _, s in e), default=None), last_step=e[-1][0] if e else None,
                done=(ROOT / 'runs' / run / 'DONE').exists())


# ----------------------------------------------------------------------------- single
SINGLE = [  # (group, label, run)
    ('State (privileged)', 'state: handcrafted', 'v-state-single-n10000-int25-s0'),
    ('State (privileged)', 'state: standardized', 'v-state-single-standardized-n10000-int25-s0'),
    ('State (privileged)', 'state: xyz x0.1', 'v-state-single-xyz0.1-n10000-int25-s0'),
    ('State (privileged)', 'state: xyz x10', 'v-state-single-xyz10-n10000-int25-s0'),
    ('State (privileged)', 'state: Fourier', 'v-state-single-fourier4-n10000-int25-s0'),
    ('Frozen generic encoder (CLS)', 'random ViT-S', 'v-B-random_vit_s-s0'),
    ('Frozen generic encoder (CLS)', 'SimCLR R50', 'v-B-simclr_r50-s0'),
    ('Frozen generic encoder (CLS)', 'DINOv2-S', 'v-B-dinov2_s-s0'),
    ('Frozen generic encoder (CLS)', 'ResNet-50 (IN cls)', 'v-B-resnet50_ce-s0'),
    ('Frozen generic encoder (CLS)', 'DINOv2-B', 'v-B-dinov2_b-s0'),
    ('Frozen generic encoder (CLS)', 'MAE ViT-B', 'v-B-mae_b16-s0'),
    ('Frozen generic encoder (CLS)', 'CLIP B/16', 'v-B-clip_b16-s0'),
    ('Frozen, spatial pooling', 'DINOv2-S patches 2x2', 'v-C-dinov2_s_p2-s0'),
    ('Frozen, spatial pooling', 'DINOv2-S patches 4x4', 'v-C-dinov2_s_p4-s0'),
    ('End-to-end', 'Impala (scratch)', 'v-D-impala-scratch-s0'),
    ('End-to-end', 'ResNet-18 fine-tune', 'v-D-resnet18-ft-s0'),
    ('End-to-end', 'DINOv2-S fine-tune', 'v-D-dinov2_s-ft-s0'),
    ('Frozen, sim-pretrained (ours)', 'ResNet-18, sim DR', 'v-F-bc-enc-dr-resnet18-s0'),
    ('Frozen, sim-pretrained (ours)', 'DINOv2-S, sim DR', 'v-F-bc-enc-dr-dinov2s-s0'),
    ('Frozen, sim-pretrained (ours)', 'DINOv2-S, sim no-DR', 'v-F-bc-enc-nodr-dinov2s-s0'),
]
GROUP_COLOR = {'State (privileged)': MUTED, 'Frozen generic encoder (CLS)': C[0], 'Frozen, spatial pooling': C[2],
               'End-to-end': C[3], 'Frozen, sim-pretrained (ours)': C[1]}


def fig_single_headline(rows):
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.barh(yi, r['final'], height=0.62, color=GROUP_COLOR[r['group']], edgecolor='white', linewidth=2)
        ax.text(r['final'] + 0.012, yi, f"{r['final']:.2f}", va='center', fontsize=8.5, color=INK)
    ax.set_yticks(y, [r['label'] for r in rows], fontsize=9)
    state = [r['final'] for r in rows if r['group'].startswith('State')]
    ax.axvspan(min(state), max(state), color=MUTED, alpha=0.12, lw=0)
    ax.axvline(max(state), color=MUTED, lw=1, ls='--')
    ax.text(max(state) + 0.008, y[0] + 0.9, 'best privileged-state policy', color=MUTED, fontsize=8, va='bottom')
    ax.set_xlim(0, 1.08)
    ax.set_xlabel('success rate (mean of last 3 evaluations)')
    ax.grid(axis='y', visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLOR.values()]
    ax.set_ylim(y[-1] - 0.7, y[0] + 1.6)
    ax.legend(handles, list(GROUP_COLOR), loc='upper center', bbox_to_anchor=(0.42, -0.08), ncol=3, fontsize=8, frameon=False)
    ax.set_title('visual-block-single: same 10K demos, visual vs. ground-truth-state policies', loc='left', fontsize=11, color=INK)
    fig.savefig(OUT / 'single_headline.png')
    plt.close(fig)


def probe_rows():
    rows = {}
    p = DIAG / 'probe' / 'visual-block-single-task1-v0.jsonl'
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if 'r2' in r and 'encoder' in r:
            rows[r['encoder']] = r
    return rows


def fig_probe(single):
    probe = probe_rows()
    enc_run = {'random_vit_s': 'v-B-random_vit_s-s0', 'simclr_r50': 'v-B-simclr_r50-s0', 'dinov2_s': 'v-B-dinov2_s-s0',
               'resnet50_ce': 'v-B-resnet50_ce-s0', 'dinov2_b': 'v-B-dinov2_b-s0', 'mae_b16': 'v-B-mae_b16-s0',
               'clip_b16': 'v-B-clip_b16-s0', 'dinov2_s_p2': 'v-C-dinov2_s_p2-s0', 'dinov2_s_p4': 'v-C-dinov2_s_p4-s0',
               'pre:v-F-enc-nodr-dinov2s': 'v-F-bc-enc-nodr-dinov2s-s0', 'pre:v-F-enc-dr-dinov2s': 'v-F-bc-enc-dr-dinov2s-s0',
               'pre:v-F-enc-dr-resnet18': 'v-F-bc-enc-dr-resnet18-s0'}
    label = {r['run']: r['label'] for r in single}
    group = {r['run']: r['group'] for r in single}
    pts = []
    for enc, run in enc_run.items():
        if enc in probe:
            pts.append(dict(encoder=enc, run=run, label=label[run], group=group[run],
                            cube_r2=probe[enc]['r2']['cube1_xyz'], mean_r2=probe[enc]['mean_r2'], success=final(run)['final']))
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for g, col in GROUP_COLOR.items():
        sel = [q for q in pts if q['group'] == g]
        if sel:
            ax.scatter([1 - q['cube_r2'] for q in sel], [q['success'] for q in sel], s=64, color=col, label=g,
                       edgecolor='white', linewidth=2, zorder=3)
    for q in pts:
        if q['label'] in ('random ViT-S', 'SimCLR R50', 'DINOv2-S', 'DINOv2-S patches 4x4', 'DINOv2-S, sim no-DR', 'CLIP B/16'):
            ax.annotate(q['label'], (1 - q['cube_r2'], q['success']), xytext=(6, -3), textcoords='offset points', fontsize=8, color=INK)
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set_xlabel('cube-position probe error, 1 - R² (ridge on frozen features, log scale)')
    ax.set_ylabel('BC success rate')
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    ax.set_title('Success tracks the last few % of cube-localization accuracy', loc='left', fontsize=11, color=INK)
    fig.savefig(OUT / 'single_probe_vs_success.png')
    plt.close(fig)
    return pts


def fig_curves(runs, path, title, labels, colors, xmax=None):
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for run, lab, col in zip(runs, labels, colors):
        e = evals(run)
        if e:
            ax.plot([s / 1000 for s, _ in e], [v for _, v in e], color=col, lw=2, marker='o', ms=4, label=lab)
    ax.set_xlabel('updates (thousands)')
    ax.set_ylabel('success rate')
    ax.set_ylim(0, 1)
    if xmax:
        ax.set_xlim(0, xmax)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(title, loc='left', fontsize=11, color=INK)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------- double
def window(run, lo, hi):
    v = [s for st, s in evals(run) if lo <= st <= hi]
    return float(np.mean(v)) if v else None


def fig_double_scaling():
    pts = {}
    for k, suf in ((1, ''), (5, '-int5'), (25, '-int25')):
        for n in (1, 3, 10, 30, 100):
            run = f'r-fixed-n{n}k{suf}'
            if evals(run):
                pts.setdefault(k, []).append((n * 1000, window(run, 850000, 1000000), run))
    pts.setdefault(25, []).append((10000, window('v-state-double-n10000-int25-s0', 850000, 1000000), 'v-state-double-n10000-int25-s0'))
    pts[25].sort()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for (k, lab), col in zip(((1, 'state, chunk start every step'), (5, 'state, every 5th step'), (25, 'state, every 25th step')), C[:3]):
        q = [p for p in pts.get(k, []) if p[1] is not None]
        ax.plot([p[0] for p in q], [p[1] for p in q], color=col, lw=2, marker='o', ms=6, label=lab)
    rep = window('r-replay-b10k-r10k', 850000, 1000000)
    ax.scatter([110000], [rep], marker='D', s=70, color=C[4], edgecolor='white', linewidth=2, zorder=4,
               label='state, 10K buffer + 1K fresh / 10K updates (110K seen)')
    m4 = window('m4-handcrafted-s0', 850000, 1000000)
    ax.axhline(m4, color=MUTED, ls='--', lw=1)
    ax.text(1100, m4 + 0.015, f'one-use streaming (~800K episodes) at 1M updates: {m4:.2f}', color=MUTED, fontsize=8)
    vis = final('v-dbl-dinov2_s_p2-s0')['final']
    ax.scatter([10000], [vis], marker='*', s=220, color=C[1], edgecolor='white', linewidth=1.5, zorder=5,
               label='VISUAL frozen DINOv2-S 2x2, every 25th step (300K upd.)')
    ax.set_xscale('log')
    ax.set_xlabel('expert episodes collected (fixed dataset)')
    ax.set_ylabel('success (mean of evals at 850K-1M updates)')
    ax.set_ylim(0, 0.75)
    ax.legend(fontsize=7.5, frameon=False, loc='upper left')
    ax.set_title('block-double: episodes matter, frame density barely does', loc='left', fontsize=11, color=INK)
    fig.savefig(OUT / 'double_data_scaling.png')
    plt.close(fig)
    return pts


def fig_takeover():
    rows = [json.loads(l) for l in (DIAG / 'takeover' / 'block-double-task1-v0.jsonl').read_text().splitlines()]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    names = [('m4-handcrafted-s0-cont2', 'student: streaming-trained (alone 0.62)'),
             ('m1-double-n10000-s0', 'student: 10K demos (alone 0.27)'),
             ('v-state-double-n10000-int25-s0', 'student: 10K demos, every 25th (alone 0.15)')]
    for (s, lab), col in zip(names, C):
        q = sorted((r['k'], r['final_success']) for r in rows if r['student'] == s)
        ax.plot([k for k, _ in q], [v for _, v in q], color=col, lw=2, marker='o', ms=5, label=lab)
    for r in rows:
        if r['student'] == 'none':
            ax.axhline(r['final_success'], color=MUTED, ls='--' if r['expert_mistake_prob'] > 0 else ':', lw=1)
            ax.text(2520, r['final_success'] + 0.012, f"expert alone ({'10% mistakes' if r['expert_mistake_prob'] > 0 else 'no mistakes'})",
                    color=MUTED, fontsize=7.5, va='bottom')
    ax.set_xlabel('takeover step k (student acts for k steps, then the scripted expert re-plans)')
    ax.set_ylabel('final success')
    ax.set_xlim(0, 3500)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, frameon=False, loc='lower left')
    ax.set_title('block-double: expert takeover after k student steps (500 episodes each)', loc='left', fontsize=11, color=INK)
    fig.savefig(OUT / 'double_takeover.png')
    plt.close(fig)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    single = [dict(group=g, label=l, **final(r)) for g, l, r in SINGLE]
    fig_single_headline(single)
    probe = fig_probe(single)
    fig_curves(['v-state-single-n10000-int25-s0', 'v-B-dinov2_s-s0', 'v-C-dinov2_s_p4-s0', 'v-D-resnet18-ft-s0',
                'v-F-bc-enc-nodr-dinov2s-s0'], OUT / 'single_curves.png', 'visual-block-single: learning curves',
               ['state (handcrafted)', 'frozen DINOv2-S CLS', 'frozen DINOv2-S 4x4 patches', 'ResNet-18 fine-tune',
                'frozen DINOv2-S, sim-pretrained'], [MUTED, C[0], C[2], C[3], C[1]], xmax=520)
    scaling = fig_double_scaling()
    take = fig_takeover()
    dbl_vis = [dict(label=l, **final(r)) for l, r in [
        ('state, 10K demos, every 25th step (1M upd.)', 'v-state-double-n10000-int25-s0'),
        ('frozen DINOv2-S CLS', 'v-dbl-dinov2_s-s0'), ('frozen DINOv2-S 2x2 patches', 'v-dbl-dinov2_s_p2-s0'),
        ('frozen DINOv2-S 4x4 patches', 'v-dbl-dinov2_s_p4-s0'), ('frozen DINOv2-B CLS', 'v-dbl-dinov2_b-s0'),
        ('frozen CLIP B/16', 'v-dbl-clip_b16-s0'), ('frozen ResNet-50', 'v-dbl-resnet50_ce-s0'),
        ('frozen random ViT-S', 'v-dbl-random_vit_s-s0'),
        ('frozen DINOv2-S, sim-pretrained (no DR)', 'v-dbl-F-bc-enc-nodr-dinov2s-s0')]]
    regime = []
    for run in sorted(p.name for p in (ROOT / 'runs').glob('r-*')):
        cfg = json.loads((ROOT / 'runs' / run / 'config.json').read_text())
        data = [json.loads(l) for l in (ROOT / 'runs' / run / 'data.jsonl').read_text().splitlines()]
        regime.append(dict(base=cfg['base'], start_interval=cfg['start_interval'], refresh_every=cfg['refresh_every'],
                           episodes_seen_k=data[-1]['collections_seen'], chunk_starts=data[-1]['allowed_starts'],
                           s450_600k=window(run, 450000, 600000), s850k_1m=window(run, 850000, 1000000), **final(run)))
    for run, sub in (('v-F-bc-enc-nodr-dinov2s-s0', 'v-F-bc-enc-nodr-dinov2s-s0'), ('v-B-dinov2_b-s0', 'v-B-dinov2_b-s0')):
        src = DIAG / 'saliency' / f'{sub}.png'
        if src.exists():
            shutil.copy(src, OUT / f'saliency_{run}.png')
    sal = {}
    for p in sorted((DIAG / 'saliency').glob('*.json')):
        r = json.loads(p.read_text())
        sal[r['run']] = {c: round(v['cube_ratio'], 2) if v['cube_ratio'] else None for c, v in r['cameras'].items()}
    cost = [json.loads(l) for l in (DIAG / 'vis_cost.jsonl').read_text().splitlines()]
    summary = dict(single=single, probe=probe, double_visual=dbl_vis, double_regime=regime, takeover=take,
                   saliency_cube_ratio=sal, vis_cost=cost)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(dict(single={r['label']: round(r['final'], 3) for r in single},
                          double_visual={r['label']: (r['final'] and round(r['final'], 3), r['last_step']) for r in dbl_vis},
                          regime={r['run']: (r['s850k_1m'] and round(r['s850k_1m'], 3), r['last_step']) for r in regime}), indent=1))


if __name__ == '__main__':
    main()
