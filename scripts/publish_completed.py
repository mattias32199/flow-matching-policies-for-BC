"""Publish only DONE experiments with 2M training and final rollout measurements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from collection_cache import locked, write_json
from individual_jobs import ROOT, resolve_run, log
from plot_data import check_complete, export, load, unified

BASES = ['m1-double-n10000-s0', 'm1-double-n10000-s1', 'm1-single-n10000-s0',
         'm1-single-n50000-s0', 'm4-handcrafted-s0', 'm4-standardized-s0',
         'm4-xyz0.1-s0', 'm4-xyz10-s0', 'm4-fourier1-s0', 'm4-fourier4-s0']


def completed_plan():
    main = json.loads((ROOT/'.local/final-results-plan.json').read_text())
    selected = {main['m1']: main['m1'], **main['m4']}
    for base in BASES:
        name = selected.get(base, resolve_run(base, ROOT))
        # check_complete also expects the m1 source itself, so inspect selection directly.
        from plot_data import load_run
        if not (ROOT/'runs'/name/'config.json').exists(): continue
        run = load_run(ROOT, name)
        m = run['streams']['metrics.jsonl']; e = run['streams']['eval.jsonl']
        if run['completed'] and m and e and m[-1]['step']==e[-1]['step']==2_000_000:
            selected[base] = name
    return {'m1':main['m1'], 'm4':{k:v for k,v in selected.items() if k!=main['m1']}}


def write_completed_section(index):
    lines = ['<!-- COMPLETED_EXTRAS_START -->', '### Additional completed M1 experiments', '',
        '![Double-task training seeds](results/mystery1_double_seeds.png)', '',
        '![Single-task completed experiments](results/mystery1_single.png)', '',
        '| Task | Nominal demonstrations | Seed | Final success | Train flow | Val flow | Val action MSE |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for source, item in index['runs'].items():
        if not source.startswith('m1-'): continue
        meta,runs=load(ROOT/'results/plot-data'/item['file'])
        cfg=runs[meta['selected_segment']]['config']; m=unified(meta,runs,'metrics.jsonl')[-1];e=unified(meta,runs,'eval.jsonl')[-1]
        task='Double' if 'double' in source else 'Single'
        lines.append(f"| {task} | {cfg['episodes']:,} | {cfg['seed']} | {e['success']:.1%} | {m['training/flow_loss']:.5f} | {m['validation/flow_loss']:.5f} | {m['validation/action_mse']:.5f} |")
    lines += ['', 'Every row has completed 2M updates and a final 500-episode rollout. Double-task seeds 0 and 1 use different demonstration seeds as well as training seeds; they are not a controlled training-seed-only comparison. Single-task and double-task success are different tasks. A 10K-versus-50K comparison is reported only after both single-task runs finish. Continuations follow checkpoint ancestry and exclude discarded post-checkpoint updates.', '', '<!-- COMPLETED_EXTRAS_END -->']
    fourier=[source for source in index['runs'] if 'fourier' in source]
    if fourier:
        lines[-1:-1]=['### Completed Fourier experiments', '',
            '![Completed Fourier experiments](results/mystery4_fourier.png)', '',
            '| Input | Final success | Train flow | Train action MSE |',
            '| --- | ---: | ---: | ---: |']
        for source in ['m4-standardized-s0', *fourier]:
            meta,runs=load(ROOT/'results/plot-data'/index['runs'][source]['file'])
            m=unified(meta,runs,'metrics.jsonl')[-1];e=unified(meta,runs,'eval.jsonl')[-1]
            lines.insert(-1,f"| {source} | {e['success']:.1%} | {m['training/flow_loss']:.5f} | {m['training/action_mse']:.5f} |")
        lines.insert(-1,'')
    path=ROOT/'RESULTS.md'; text=path.read_text(); block='\n'.join(lines)
    begin='<!-- COMPLETED_EXTRAS_START -->'; end='<!-- COMPLETED_EXTRAS_END -->'
    if begin in text:
        text=text[:text.index(begin)]+block+text[text.index(end)+len(end):]
    else:
        text=text.replace('## Mystery 4:',block+'\n\n## Mystery 4:',1)
    begin='## Download the complete original plot measurements'; end='## Reproduction limits and validation'
    download=[begin,'',
        'Each completed experiment has a compressed, pickle-free NPZ with full unsmoothed training/success histories, configuration, and checkpoint ancestry. Historical measurements remain lossless.', '',
        '| Experiment | Complete NPZ | Compressed size |','| --- | --- | ---: |']
    for source,item in index['runs'].items():
        download.append(f"| {source} | [Download](results/plot-data/{item['file']}) | {item['bytes']/1024:.1f} KiB |")
    download += ['',f"Combined size: **{sum(v['bytes'] for v in index['runs'].values())/1024:.1f} KiB**. [SHA-256 manifest](results/plot-data/manifest.json).", '',
        'Recreate the figures using only the published NPZ files:', '', '```bash', 'source scripts/env.sh',
        '.venv/bin/python scripts/plot_from_npz.py --data results/plot-data --output .local/recreated-plots','```','',
        '**For the four main M4 conditions, use `fixed_validation__*`, which is exactly what the corrected validation figure plots.** Every point uses one shared regenerated held-out dataset (index 0, parallel 2500); only retained checkpoints were reevaluated. These are measured points, with no interpolation or rescaling. The dense historical validation cannot be repaired exactly because the old held-out arrays and intermediate checkpoints were not retained.', '',
        '```python', 'import numpy as np', 'with np.load("results/plot-data/m4-handcrafted-s0.npz", allow_pickle=False) as d:',
        '    validation_steps = d["fixed_validation__step"]', '    validation_flow = d["fixed_validation__flow_loss"]',
        '    validation_mse = d["fixed_validation__action_mse"]', '    validation_se = d["fixed_validation__flow_loss_batch_se"]',
        '    training_steps = d["history__metrics__step"]', '    training_flow = d["history__metrics__training__flow_loss"]',
        '    success_steps = d["history__eval__step"]', '    success = d["history__eval__success"]','```','',
        '`history__metrics__validation__*` preserves the original logged scores, including the held-out-set change at resume; it is not the corrected validation figure. `audit__*` retains all three diagnostic datasets. M1 validation uses its original `history__metrics__validation__*` arrays. [Validation audit and limitations](results/validation/M4_VALIDATION_AUDIT.md) · [Archive schema](scripts/plot_data.py).','',]
    text=text[:text.index(begin)]+'\n'.join(download)+'\n'+text[text.index(end):]
    count=len(index['runs'])
    text=text.replace('# Mystery 1 and 4: completed five-run results', '# Mystery 1 and 4: completed experiments')
    intro_start=text.index('\n\n')+2; intro_end=text.index('\n\n',intro_start)
    text=text[:intro_start]+f'**{count} of 10 experiments have completed 2,000,000 updates and final 500-episode evaluations.** The four main M4 conditions are complete. Additional completed M1 results, plots and per-experiment NPZ files are included below; unfinished experiments are excluded from final comparisons.'+text[intro_end:]
    text=text.replace('All five runs use **state-based `block-double-task1-v0`**, seed 0, and **2M optimizer updates**.', 'The original five-run comparison uses **state-based `block-double-task1-v0`**, seed 0, and **2M optimizer updates**. Additional M1 runs use the same model/optimizer settings; their task, demonstration count and seed are listed below.')
    text=text.replace('Each NPZ includes both full logged histories (`history__*`) and fixed-set validation measurements (`audit__*`).', 'Each M4 NPZ includes full logged histories (`history__*`), ready-to-plot fixed-set validation arrays (`fixed_validation__*`), and the complete audit (`audit__*`). The plotted fixed-set series has no validation-set switch at resume.')
    path.write_text(text.rstrip()+'\n')


def publish(publish=False):
    with locked(ROOT/'.local/completed-publication.lock'):
        plan=completed_plan(); check_complete(ROOT,plan)
        signature=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()+
            Path(__file__).read_bytes()+(ROOT/'scripts/plot_data.py').read_bytes()+
            (ROOT/'scripts/plot_from_npz.py').read_bytes()).hexdigest()
        state_path=ROOT/'.local/completed-publication-state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('signature')==signature and state.get('published'):return state
        if publish and state.get('signature')==signature and state.get('commit'):
            return finish_push(state, state_path)
        from summarize import summarize
        from plot_from_npz import recreate
        summarize();index=export(plan=plan,require_complete=True)
        recreate(ROOT/'results/plot-data',ROOT/'results');write_completed_section(index)
        if not publish:return {'completed':len(index['runs']), 'signature':signature}
        def git(*args, **kw):
            env=os.environ.copy();env['GIT_SSH_COMMAND']='ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes'
            return subprocess.run(['git',*args],cwd=ROOT,env=env,text=True,check=True,**kw)
        branch=git('branch','--show-current',capture_output=True).stdout.strip()
        if branch!='reproduce-ocbench-mysteries-1-4':raise RuntimeError('Unexpected publication branch')
        git('diff','--cached','--quiet')
        paths=['RESULTS.md','results/plot-data','results/runs','results/catalog.json','results/curves.csv','results/curves.png']
        for filename in ['mystery1','mystery1_double_seeds','mystery1_single','mystery4_baseline','mystery4_scaling','mystery4_validation','mystery4_validation_fixed']:
            paths += [f'results/{filename}.png',f'results/{filename}.svg']
        for extension in ['png','svg']:
            extra='results/mystery4_fourier.'+extension
            if (ROOT/extra).exists():paths.append(extra)
        git('add','--',*paths);git('diff','--cached','--check')
        if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
            git('-c','user.name=Haoran Huan','-c','user.email=yzhhr@outlook.com','commit','-m',f'Publish {len(index["runs"])} completed experiments and consistent validation arrays')
        commit=git('rev-parse','HEAD',capture_output=True).stdout.strip()
        state={'signature':signature,'published':False,'commit':commit,'completed':len(index['runs'])}
        write_json(state_path,state)
        return finish_push(state,state_path)


def finish_push(state,state_path):
    env=os.environ.copy();env['GIT_SSH_COMMAND']='ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes'
    ref='refs/heads/reproduce-ocbench-mysteries-1-4'
    subprocess.run(['git','push','origin',state['commit']+':'+ref],cwd=ROOT,env=env,check=True)
    remote=subprocess.check_output(['git','ls-remote','origin',ref],cwd=ROOT,env=env,text=True).split()[0]
    if remote!=state['commit']:raise RuntimeError('Remote publication SHA mismatch')
    state.update(published=True,published_at=time.time());write_json(state_path,state)
    log(f'Published completed-experiment update: {state["completed"]} experiments, remote verified {state["commit"]}.')
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    print(json.dumps(publish(args.publish)))
