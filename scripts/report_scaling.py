"""Block-single scaling grid (follow-up group S/L): fit failure(P, E) and plot it.

Reads runs/s-sgl-bc-p{P}[-r{rep}]-e{E}-s{seed}/ (rep = pair-file/encoder replicate) final_eval.jsonl (1000 rollouts; falls back to the last eval.jsonl
entry, marked unfinished). Model, fitted on P > 0 cells: a logit law (see model()); the earlier power law is also fitted on P >= 2K.
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
    for d in sorted(glob.glob(str(ROOT / 'runs' / 's-sgl-bc-p*-e*-s*'))):
        m = re.search(r'/(s)-sgl-bc-p(\d+)(k?)(?:-r(\d))?-e(\d+)-s(\d)$', d)
        f = Path(d) / 'final_eval.jsonl'
        final = f.exists()
        f = f if final else Path(d) / 'eval.jsonl'
        if not f.exists():
            continue
        r = json.loads(f.read_text().splitlines()[-1])
        rows.append(dict(group=m[1], P=int(m[2]) * (1000 if m[3] else 1), rep=int(m[4] or 0), E=int(m[5]), seed=int(m[6]), success=r['success'],
                         episodes=r['episodes'], step=r['step'], final=final))
    return rows


FLOOR = 0.02  # failure floor (the best cells sit at 0.03-0.05)


def model(t, P, E):
    """Logit law: logit(success) = k0 + kE ln(E/1K) + kP ln(P/10K) + kPP ln(P/10K)^2, failure >= FLOOR."""
    lp, le = np.log(P / 1e4), np.log(E / 1000)
    return FLOOR + (1 - FLOOR) / (1 + np.exp(t[0] + t[1] * le + t[2] * lp + t[3] * lp ** 2))


def product(t, P, E):
    """Power law that fits P >= 2K well but not low P: fail = c + (a + b (P/1e4)^-beta) (E/1000)^-alpha."""
    return t[0] + (t[1] + t[3] * (P / 1e4) ** -t[4]) * (E / 1000) ** -t[2]


def main():
    rows = load()
    s = [r for r in rows if r['group'] == 's']
    P, E = (np.array([r[k] for r in s], float) for k in ('P', 'E'))
    F = 1 - np.array([r['success'] for r in s])
    m = P > 0
    fit = least_squares(lambda t: model(t, P[m], E[m]) - F[m], [2, 1, 0.3, 0])
    hi = P >= 2000
    pw = least_squares(lambda t: product(t, P[hi], E[hi]) - F[hi], [0.03, 0.1, 1, 0.05, 0.5], bounds=(0, np.inf))
    g = least_squares(lambda t: t[0] + t[1] * (E[~m] / 1000) ** -t[2] - F[~m], [0.2, 0.6, 0.5], bounds=(0, np.inf))
    rm = lambda r: float(np.sqrt((r ** 2).mean()))
    pw_all = product(pw.x, P[m], E[m]) - F[m]
    summary = dict(model=f'fail = {FLOOR} + (1 - {FLOOR}) / (1 + exp(k0 + kE ln(E/1K) + kP ln(P/10K) + kPP ln(P/10K)^2)), P>0',
                   params=dict(zip(('k0', 'kE', 'kP', 'kPP'), map(float, fit.x))),
                   rmse=rm(fit.fun), max_abs=float(np.abs(fit.fun).max()), n=int(m.sum()),
                   power_law_P_ge_2K=dict(model='fail = c + (a + b (P/1e4)^-beta) (E/1000)^-alpha',
                                          params=dict(zip(('c', 'a', 'alpha', 'b', 'beta'), map(float, pw.x))),
                                          rmse=rm(pw.fun), rmse_extrapolated_to_all_P=rm(pw_all)),
                   generic_cls=dict(model='fail = c + a (E/1000)^-alpha', params=dict(zip(('c', 'a', 'alpha'), map(float, g.x))),
                                    rmse=rm(g.fun)), rows=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'scaling_single.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}, indent=1))

    PS = sorted({int(p) for p in P if p > 0})
    blues = ['#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#256abf', '#184f95', '#0d366b']
    fig, ax = plt.subplots(figsize=(7.5, 5))
    Es = np.geomspace(250, 8818, 100)
    for Pk, col, lab in [(0, '#8a8984', 'generic DINOv2-S CLS (P = 0)')] + [
            (p, c, f'P = {p // 1000}K pairs' if p >= 1000 else f'P = {p} pairs') for p, c in zip(PS, blues[-len(PS):])]:
        sel = [r for r in s if r['P'] == Pk]
        ax.scatter([r['E'] for r in sel], [1 - r['success'] for r in sel], s=36, color=col, zorder=3,
                   edgecolor='white', linewidth=1.5, marker='o' if all(r['final'] for r in sel) else 'o')
        y = (g.x[0] + g.x[1] * (Es / 1000) ** -g.x[2]) if Pk == 0 else model(fit.x, Pk, Es)
        ax.plot(Es, y, color=col, lw=2, ls='--' if Pk == 0 else '-', label=lab)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks([250, 500, 1000, 2000, 4000, 8818]); ax.set_xticklabels(['250', '500', '1K', '2K', '4K', '8.8K'])
    ax.set_yticks([0.03, 0.05, 0.1, 0.2, 0.5, 1]); ax.set_yticklabels(['0.03', '0.05', '0.1', '0.2', '0.5', '1']); ax.set_ylim(0.02, 1.1)
    ax.minorticks_off()
    ax.set_xlabel('demonstration episodes E'); ax.set_ylabel('failure rate (1 − success, 1000 rollouts)')
    k = fit.x
    ax.set_title(f'block-single, frozen CLS visual BC, 300K updates; lines: logit fit\nlogit(success) = {k[0]:.2f} + '
                 f'{k[1]:.2f} ln(E/1K) + {k[2]:.2f} ln(P/10K) {k[3]:+.3f} ln(P/10K)$^2$ (failure floor {FLOOR})', fontsize=9)
    ax.grid(color='#e6e5e1', lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc='lower left')
    fig.tight_layout(); fig.savefig(OUT / 'scaling_single.png', dpi=150)


if __name__ == '__main__':
    main()
