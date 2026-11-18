"""Block-single scaling grid (follow-up group S/L): fit failure(P, E) and plot it.

Reads runs/{s,l}-sgl-bc-p{P}k-e{E}-s{seed}/final_eval.jsonl (1000 rollouts; falls back to the last eval.jsonl
entry, marked unfinished). Model, fitted on P > 0 cells:  fail = c + (a + b (P/1e4)^-beta) (E/1000)^-alpha.
Writes experiments/vision-bc/figures/scaling_single.png and scaling_single.json.
  .venv/bin/python scripts/report_scaling.py
"""
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments' / 'vision-bc' / 'figures'


def load():
    rows = []
    for d in sorted(glob.glob(str(ROOT / 'runs' / '[sl]-sgl-bc-p*k-e*-s*'))):
        m = re.search(r'/([sl])-sgl-bc-p(\d+)k-e(\d+)-s(\d)$', d)
        f = Path(d) / 'final_eval.jsonl'
        final = f.exists()
        f = f if final else Path(d) / 'eval.jsonl'
        if not f.exists():
            continue
        r = json.loads(f.read_text().splitlines()[-1])
        rows.append(dict(group=m[1], P=int(m[2]) * 1000, E=int(m[3]), seed=int(m[4]), success=r['success'],
                         episodes=r['episodes'], step=r['step'], final=final))
    return rows


def model(t, P, E):
    return t[0] + (t[1] + t[3] * (P / 1e4) ** -t[4]) * (E / 1000) ** -t[2]


def main():
    rows = load()
    s = [r for r in rows if r['group'] == 's']
    P, E = (np.array([r[k] for r in s], float) for k in ('P', 'E'))
    F = 1 - np.array([r['success'] for r in s])
    m = P > 0
    fit = least_squares(lambda t: model(t, P[m], E[m]) - F[m], [0.03, 0.1, 1, 0.05, 0.5], bounds=(0, np.inf))
    g = least_squares(lambda t: t[0] + t[1] * (E[~m] / 1000) ** -t[2] - F[~m], [0.2, 0.6, 0.5], bounds=(0, np.inf))
    summary = dict(model='fail = c + (a + b (P/1e4)^-beta) (E/1000)^-alpha, P>0 cells',
                   params=dict(zip(('c', 'a', 'alpha', 'b', 'beta'), map(float, fit.x))),
                   rmse=float(np.sqrt((fit.fun ** 2).mean())), n=int(m.sum()),
                   generic_cls=dict(model='fail = c + a (E/1000)^-alpha', params=dict(zip(('c', 'a', 'alpha'), map(float, g.x))),
                                    rmse=float(np.sqrt((g.fun ** 2).mean()))), rows=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'scaling_single.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1))

    blues = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']
    fig, ax = plt.subplots(figsize=(7.5, 5))
    Es = np.geomspace(250, 8818, 100)
    for Pk, col, lab in [(0, '#8a8984', 'generic DINOv2-S CLS (P = 0)')] + [
            (p, c, f'P = {p // 1000}K pairs') for p, c in zip((2000, 6000, 20000, 60000, 190000), blues)]:
        sel = [r for r in s if r['P'] == Pk]
        ax.scatter([r['E'] for r in sel], [1 - r['success'] for r in sel], s=36, color=col, zorder=3,
                   edgecolor='white', linewidth=1.5, marker='o' if all(r['final'] for r in sel) else 'o')
        y = (g.x[0] + g.x[1] * (Es / 1000) ** -g.x[2]) if Pk == 0 else model(fit.x, Pk, Es)
        ax.plot(Es, y, color=col, lw=2, ls='--' if Pk == 0 else '-', label=lab)
    lr = [r for r in rows if r['group'] == 'l']
    ax.scatter([r['E'] for r in lr], [1 - r['success'] for r in lr], marker='x', s=60, color='#222', zorder=4,
               label='leak control: pairs only from the E demos (20K / 190K)')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks([250, 500, 1000, 2000, 4000, 8818]); ax.set_xticklabels(['250', '500', '1K', '2K', '4K', '8.8K'])
    ax.set_yticks([0.03, 0.05, 0.1, 0.2, 0.5, 1]); ax.set_yticklabels(['0.03', '0.05', '0.1', '0.2', '0.5', '1'])
    ax.minorticks_off()
    ax.set_xlabel('demonstration episodes E'); ax.set_ylabel('failure rate (1 − success, 1000 rollouts)')
    t = fit.x
    ax.set_title(f'block-single, frozen CLS visual BC, 300K updates\nfit: fail = {t[0]:.3f} + ({t[1]:.3f} + '
                 f'{t[3]:.3f}·(P/10K)^-{t[4]:.2f})·(E/1K)^-{t[2]:.2f}', fontsize=10)
    ax.grid(color='#e6e5e1', lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='lower left')
    fig.tight_layout(); fig.savefig(OUT / 'scaling_single.png', dpi=150)


if __name__ == '__main__':
    main()
