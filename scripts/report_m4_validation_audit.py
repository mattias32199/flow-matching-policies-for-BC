"""Summarize read-only M4 reevaluation; keep historical and corrected measurements distinct."""
import json
from pathlib import Path
from train import ROOT
from plot_data import export
from demo_identity import generator_identity


def report():
    path=ROOT/'results/validation/m4-validation-audit.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    plan=json.loads((ROOT/'.local/final-results-plan.json').read_text())
    features=['handcrafted','standardized','xyz0.1','xyz10']
    # Refuse partial diagnostics: all retained source/final checkpoints on all three datasets.
    for source,selected in plan['m4'].items():
        checkpoints=list((ROOT/'runs'/source/'checkpoints').glob('params_*.pkl'))+list((ROOT/'runs'/selected/'checkpoints').glob('params_*.pkl'))
        for p in checkpoints:
            step=int(p.stem.split('_')[-1])
            for label in ['original512','original2500','resumed2500']:
                assert any(r['experiment']==source and r['checkpoint_step']==step and r['validation_set']==label for r in rows),(source,step,label)
    private=ROOT/'.local/m4-validation-audit'
    metadata={'generator':generator_identity(),'batch_seed':20261022,'batches':32,'batch_size':1024,
      'protocol':'Same sampled transition indices and flow/action noise keys across checkpoints for each dataset; no parameter updates.',
      'limitation':'Seed-regenerated simulator arrays, not byte-identical copies of historical validation data that was not retained. Batch SE measures Monte Carlo diagnostic variation, not training-seed uncertainty.',
      'datasets':{p.stem:json.loads(p.read_text()) for p in private.glob('*.json')}}
    (ROOT/'results/validation/m4-validation-audit-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    def row(source,step,label):return next(r for r in rows if r['experiment']==source and r['checkpoint_step']==step and r['validation_set']==label)
    lines=['# M4 validation discontinuity audit','',
      'Streaming recovery regenerated validation data: collection index 0 became 3072. The first three conditions resumed at 1.4M; XYZ×10 resumed at 1.3M. Validation scores are diagnostics only: they do not update weights, choose checkpoints, or control the constant Adam learning rate.', '',
      'All saved model/Adam/agent-RNG tensors survive deserialization exactly. Boundary checkpoint hashes match the recorded resume configuration, step counters match, and observation/action preprocessing is unchanged. These checks support correct checkpoint restoration; they do not claim an identical uninterrupted training-data/sampler trajectory.', '',
      'The validation-set change is confirmed by the executed code and run configuration. Same-checkpoint measurements below demonstrate dataset sensitivity; they cannot assign an exact fraction of the historical jump because the original raw validation arrays were not retained. No checkpoint-loading defect has been found, and a validation-only change does not justify rerunning training.', '',
      '## Controlled same-checkpoint comparison', '',
      'Each row below evaluates the SAME boundary model on seed-regenerated index-0 and index-3072 validation sets at the SAME parallelism (2500), with 32 batches of 1024 transitions and fixed sampling/noise keys. The final column reevaluates the 2M model on the SAME index-0 set (parallel 2500). No training is rerun.', '',
      '| Condition | Boundary | Boundary: index 0 flow | Boundary: index 3072 flow | Set-only change | 2M: index 0 flow |',
      '| --- | ---: | ---: | ---: | ---: | ---: |']
    for source,selected in plan['m4'].items():
        config=json.loads((ROOT/'runs'/source/'config.json').read_text());resumed=json.loads((ROOT/'runs'/selected/'config.json').read_text());step=resumed['resume_step']
        original='original2500'
        a=row(source,step,original);b=row(source,step,'resumed2500');c=row(source,2_000_000,original)
        lines.append(f"| {config['features']} | {step:,} | {a['flow_loss']:.5f} | {b['flow_loss']:.5f} | {100*(b['flow_loss']/a['flow_loss']-1):+.1f}% | {c['flow_loss']:.5f} |")
    lines += ['', '## Fixed-set checkpoint comparison', '',
      '![Fixed validation checkpoint audit](../mystery4_validation_fixed.png)', '',
      'This figure evaluates every retained checkpoint on one shared original-seed validation dataset (index 0, parallel 2500). Points are actual reevaluations; lines only connect retained checkpoints. Error bars are one Monte Carlo batch standard error, not uncertainty across training seeds.', '',
      '## What can be repaired', '',
      'Reevaluation makes the retained checkpoint comparison use a consistent validation set. It does not require retraining merely because validation changed. The dense historical validation curve cannot be corrected exactly: the runner retained only the latest two checkpoints per segment, plus migration snapshots, and the original raw validation arrays were not pinned. Do not interpolate reevaluated points and present them as historical measurements. Robot rollout success is a separate measurement and has not been replaced.', '',
      'The first three runs also changed simulator collection parallelism from 512 to 2500; XYZ×10 already used 2500. The audit includes index-0 datasets at both settings to separate this implementation change from the seed/split change. Regenerated data can differ from historical data because of simulator/GPU numerical variation; this audit does not claim historical byte identity.', '',
      '## Prevention', '',
      'New training processes atomically save the actual raw validation arrays as validation.npz. Migration bundles copy those arrays with a SHA-256 checksum, and resumed training reuses them rather than regenerating a held-out set. Existing running processes cannot retroactively recover an already lost dataset.', '',
      '## Files and reproduction', '',
      '[All measurements](m4-validation-audit.jsonl) · [Dataset/protocol metadata](m4-validation-audit-metadata.json). Each published per-condition NPZ also contains these measurements under `audit__*`, separate from the original `history__*` arrays.', '',
      'Teammates can recreate every figure, including the fixed-set audit, from the five published NPZ files with `.venv/bin/python scripts/plot_from_npz.py --data results/plot-data --output .local/recreated-plots`. Filter `audit__validation_set == "original2500"` and plot `audit__flow_loss` against `audit__checkpoint_step` for direct NumPy access.', '',
      'Run `source scripts/env.sh` and `.venv/bin/python scripts/audit_m4_validation.py` inside a GPU allocation; then `.venv/bin/python scripts/report_m4_validation_audit.py`. Raw regenerated validation datasets stay local; scalar diagnostics are public.','']
    (ROOT/'results/validation/M4_VALIDATION_AUDIT.md').write_text('\n'.join(lines))
    export(plan=plan,require_complete=True)
    from plot_from_npz import recreate_fixed_validation
    recreate_fixed_validation(ROOT/'results/plot-data',ROOT/'results')
    report_path=ROOT/'RESULTS.md';text=report_path.read_text();start='<!-- M4_VALIDATION_AUDIT_START -->';end='<!-- M4_VALIDATION_AUDIT_END -->'
    section='\n'.join([start,'### Validation discontinuity: controlled reevaluation','',
      '**The historical validation curve changes held-out datasets at recovery. Connecting the plotted line does not make those scores directly comparable.** We audited restoration and reevaluated saved checkpoints on fixed, seed-regenerated validation sets without training updates. See the [controlled measurements and limitations](results/validation/M4_VALIDATION_AUDIT.md).','',
      'The final-score table retains the original logged validation scores. The fixed-set figure uses one consistent regenerated dataset and averages 32 batches per checkpoint; its numbers are separate from the single-batch historical logs.','',
      '[Historical logged validation curve — held-out data changes during recovery](results/mystery4_validation.png)','',end,''])
    if start in text:text=text[:text.index(start)]+section+text[text.index(end)+len(end):]
    else:text=text.replace('## Download the complete original plot measurements',section+'\n## Download the complete original plot measurements')
    # Update compressed sizes after adding the audit arrays.
    index=json.loads((ROOT/'results/plot-data/manifest.json').read_text())
    import re
    for source,item in index['runs'].items():
        text=re.sub(r'(\| '+re.escape(source)+r' \| \[Download\]\([^\n]+?\) \| )[^|]+(\|)',lambda m:m[1]+f"{item['bytes']/1024:.1f} KiB "+m[2],text)
    text=re.sub(r'Combined size: \*\*[^*]+\*\*',f"Combined size: **{sum(v['bytes'] for v in index['runs'].values())/1024:.1f} KiB**",text)
    text=text.replace('Each condition is shown as one continuous measured history through 2M updates.', 'Training and robot-success curves show one continuous measured history through 2M updates. The validation figure above compares retained checkpoints on one fixed regenerated set; original logged validation remains in the archives.')
    text=text.replace('![Complete validation history](results/mystery4_validation.png)', '![Fixed-set validation reevaluation](results/mystery4_validation_fixed.png)')
    report_path.write_text(text)


if __name__=='__main__':report()
