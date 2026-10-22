"""Publish the requested five-run report only after all five finish at 2M."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import time
import numpy as np
from collection_cache import locked, write_json
from plot_data import ROOT, export, check_complete, load
from prepare_migration import leaves, sha256


def command(args, **kwargs):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, **kwargs)


def progress(message):
    path = ROOT / 'PROGRESS.md'
    path.read_text()
    with path.open('a') as file:
        file.write(f'\n- {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}: Final publication: {message}\n')


def validate_checkpoints(plan):
    results = {}
    for name in [plan['m1'], *plan['m4'].values()]:
        path = ROOT/'runs'/name/'checkpoints/params_2000000.pkl'
        with path.open('rb') as file:
            state = pickle.load(file)['agent']
        if int(state['network']['step']) != 2_000_001:
            raise ValueError(f'Final checkpoint step mismatch: {name}')
        for array in leaves(state):
            if array.dtype.kind in 'biufc' and not np.isfinite(array).all():
                raise ValueError(f'Nonfinite final checkpoint: {name}')
        del state
        results[name] = {'step': 2_000_000, 'bytes': path.stat().st_size,
                         'sha256': sha256(path), 'all_numeric_values_finite': True}
    path = ROOT/'results/validation/final-five-checkpoints.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(results, indent=2)+'\n')


def write_report(plan, index):
    m1meta, m1runs = load(ROOT/'results/plot-data'/f'{plan["m1"]}.npz')
    m1 = m1runs[plan['m1']]; records=m1['streams']['metrics.jsonl']; evaluations=m1['streams']['eval.jsonl']
    low=min(records, key=lambda row: row['validation/flow_loss']); final=records[-1]
    lines=['# Mystery 1 and 4: completed five-run results', '',
        'The requested **M1 double-10K seed-0 run and four main M4 conditions have all completed 2,000,000 updates**, including final 500-episode rollout evaluations. Fourier variants and other M1 runs are separate experiments; see the live catalog below.', '',
        '## Mystery 1: loss and control can disagree', '',
        '![Completed Mystery 1](results/mystery1.png)', '',
        f"Final robot success: **{evaluations[-1]['success']:.1%}**. Validation flow loss reached its minimum **{low['validation/flow_loss']:.5f} at {low['step']:,} updates**, then rose to **{final['validation/flow_loss']:.5f} at 2M** ({final['validation/flow_loss']/low['validation/flow_loss']:.2f}×). Final training flow loss was {final['training/flow_loss']:.5f}; validation action MSE was {final['validation/action_mse']:.5f}.", '',
        'This completed run supports the observation that worsening validation flow loss need not mean worsening robot control. It does not prove that overfitting causes better control or establish robustness across training seeds. The single-task dataset-size comparison is a separate question.', '',
        '## Mystery 4: scaling affects behavior much more than the training scores suggest', '',
        '![Completed Mystery 4 baseline](results/mystery4_baseline.png)', '',
        '![Completed Mystery 4 scaling](results/mystery4_scaling.png)', '',
        'All four rows below are evaluated at **the same 2M optimizer updates**, using 500 robot attempts each. These remain single-seed results; checkpoint evaluations are not independent training seeds.', '',
        '| Input | Final success | Train flow | Val flow | Val action MSE |',
        '| --- | ---: | ---: | ---: | ---: |']
    scores={}
    for source, selected in plan['m4'].items():
        meta,runs=load(ROOT/'results/plot-data'/f'{source}.npz');run=runs[selected]
        m=run['streams']['metrics.jsonl'][-1];e=run['streams']['eval.jsonl'][-1];feature=run['config']['features'];scores[feature]=e['success']
        lines.append(f"| [{feature}](results/runs/{selected}/README.md) | {e['success']:.1%} | {m['training/flow_loss']:.5f} | {m['validation/flow_loss']:.5f} | {m['validation/action_mse']:.5f} |")
    best=max(scores,key=scores.get);worst=min(scores,key=scores.get)
    lines+=['',f"The observed final success spread is **{100*(max(scores.values())-min(scores.values())):.1f} percentage points**: {best} is highest and {worst} lowest in this seed. Use the measured losses in the table rather than assuming they are identical. This comparison describes the observed effect; it does not prove a distribution-shift mechanism or a universal ranking.", '',
        '### Complete training curves', '',
        '![Complete validation history](results/mystery4_validation.png)', '',
        'Each condition is shown as one continuous measured history through 2M updates. Overlapping records are resolved using the checkpoint ancestry; discarded training tails are not mixed into the final trajectory. Weights, optimizer state and agent RNG were restored during recovery; training streams, sampler and validation sets restarted, so validation scores before and after recovery use different held-out sets.', '',
        'Loss panels show faint raw records plus nonoverlapping 50K-update means, with logarithmic loss axes and the first 10K loss updates omitted. Success points are measured evaluations, not smoothed or extrapolated. No multi-seed uncertainty band is implied.', '',
        '## Download the complete original plot measurements', '',
        'Each NPZ is **compressed, lossless and pickle-free**: numeric arrays, every unsmoothed scalar record, configuration, and explicit segment/branch metadata. These are the full measurements used for plotting, not simulated robot demonstrations or model checkpoints. Each file provides one ready-to-plot history from the original early steps to 2M; detailed provenance is retained as additional metadata.', '',
        '| Experiment | Complete NPZ | Compressed size |', '| --- | --- | ---: |']
    for source, item in index['runs'].items():
        lines.append(f"| {source} | [Download](results/plot-data/{item['file']}) | {item['bytes']/1024:.1f} KiB |")
    lines+=['',f"Combined size: **{sum(v['bytes'] for v in index['runs'].values())/1024:.1f} KiB**. [SHA-256 hashes and file manifest](results/plot-data/manifest.json).", '',
        'Recreate the figures from only those five NPZ files (from the repository root):', '',
        '```bash', 'source scripts/env.sh', '.venv/bin/python scripts/plot_from_npz.py --data results/plot-data --output .local/recreated-plots', '```', '',
        'For direct NumPy access, use `d = np.load(filename, allow_pickle=False)`. Plot `d["history__metrics__step"]` against `d["history__metrics__training__flow_loss"]` (or `history__metrics__validation__flow_loss`, `history__metrics__training__action_mse`, `history__metrics__validation__action_mse`). Robot success is `history__eval__success` against `history__eval__step`. These arrays already combine the complete selected history in increasing step order; no manual segment stitching is needed. `history__metrics__jsonl` and `history__eval__jsonl` preserve the full unsmoothed records. [Archive reader/exporter](scripts/plot_data.py) documents the schema.', '',
        '## Reproduction limits and validation', '',
        'These are state-based, finite-budget experiments, not image policies or a literal infinite-data limit. Our frozen z-score/scaling, one-use streaming policy and restarted validation sets differ from an exact numerical blog reproduction. See the [plain-language guide](EXPERIMENT_GUIDE.md), [native and wrapper commands](OCBENCH_COMMANDS.md), [original blog](https://seohong.me/blog/behavioral-cloning-mystery/), and [final checkpoint integrity checks](results/validation/final-five-checkpoints.json). Checkpoint files, robot data and private scheduler paths remain local.', '',
        '[Full run catalog](results/runs/README.md) · [W&B](https://forge.coreweave.com/wandb/yzchhr/ocbench-bc-mysteries-1-4) · [Earlier performance experiments](results/profiling/README.md)', '']
    text=(ROOT/'RESULTS.md').read_text()
    begin='<!-- LIVE_RESULTS_START -->';end='<!-- LIVE_RESULTS_END -->'
    if begin in text:
        lines += [text[text.index(begin):text.index(end)+len(end)]]
    (ROOT/'RESULTS.md').write_text('\n'.join(lines).rstrip()+'\n')


def finalize(plan, publish=False):
    state_path=ROOT/'.local/final-publication-state.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    env=os.environ.copy();env['GIT_SSH_COMMAND']='ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes'
    if state.get('published') or (state.get('commit') and not publish):
        return state
    if not state.get('commit'):
        check_complete(ROOT,plan)
        if publish:
            command(['git','diff','--cached','--quiet'])
            branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
            if branch != plan['branch']:
                raise RuntimeError('Publication branch changed; refusing automatic commit.')
        from summarize import summarize
        summarize()
        validate_checkpoints(plan)
        index=export(plan=plan,require_complete=True)
        from plot_from_npz import recreate
        recreate(ROOT/'results/plot-data',ROOT/'results')
        write_report(plan,index)
        paths=['RESULTS.md','results/plot-data','results/validation/final-five-checkpoints.json',
               'results/catalog.json','results/curves.csv','results/runs/README.md','results/curves.png']
        for filename in ['mystery1','mystery4_baseline','mystery4_scaling','mystery4_validation','continuations']:
            paths += [f'results/{filename}.png',f'results/{filename}.svg']
        for source in [plan['m1'],*plan['m4']]:
            paths += [str(p.relative_to(ROOT)) for p in (ROOT/'results/runs').iterdir()
                      if p.name==source or p.name.startswith(source+'-cont')]
        if not publish:
            return {'prepared':True,'bytes':sum(v['bytes'] for v in index['runs'].values())}
        command(['git','add','--',*paths]);command(['git','diff','--cached','--check'])
        command(['git','-c','user.name=Haoran Huan','-c','user.email=yzhhr@outlook.com','commit','-m',
                 'Complete main Mystery 1 and 4 results with lossless per-run plot data'])
        state['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        write_json(state_path,state)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=state['commit']:
        raise RuntimeError('HEAD moved after publication commit; manual reconciliation needed.')
    command(['git','push','origin','HEAD:'+plan['branch']],env=env)
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+plan['branch']],cwd=ROOT,text=True,env=env).split()[0]
    if remote!=state['commit']:
        raise RuntimeError('Remote publication verification failed.')
    state['published']=True;state['published_at']=time.time();write_json(state_path,state)
    progress('Published and verified completed report plus all five per-run NPZ archives: '+state['commit'])
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    plan=json.loads((ROOT/'.local/final-results-plan.json').read_text())
    with locked(ROOT/'.local/final-publication.lock'):
        try:
            print(json.dumps(finalize(plan,args.publish)),flush=True)
        except Exception as error:
            progress(f'Failed ({type(error).__name__}: {error}); no incomplete result is published.')
            raise
