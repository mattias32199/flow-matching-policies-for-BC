"""Compute plain-language findings and a portable scheduler snapshot; no private paths exported."""
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from matrix import matrix
from catalog import read_records

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'


def at(rows, step):
    return next(row for row in rows if row['step'] == step)


def report():
    def records(run, stream):
        return read_records(OUT / 'runs' / run / stream)
    m1 = 'm1-double-n10000-s0'
    metrics = records(m1, 'metrics.jsonl'); evaluations = records(m1, 'eval.jsonl')
    minimum = min(metrics, key=lambda row: row['validation/flow_loss'])
    features = ['handcrafted', 'standardized', 'xyz0.1', 'xyz10']
    common = set.intersection(*[{row['step'] for row in records(f'm4-{f}-s0', 'eval.jsonl')} for f in features])
    common_step = max(common)
    comparisons = {feature: dict(evaluation=at(records(f'm4-{feature}-s0', 'eval.jsonl'), common_step),
                                metrics=at(records(f'm4-{feature}-s0', 'metrics.jsonl'), common_step))
                   for feature in features}
    ledger_path = ROOT / '.local/individual-job-status.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    jobs = []
    for planned in matrix():
        name = planned['run']
        segment = name + '-cont1' if name.startswith('m4-') and name + '-cont1' in ledger else name
        entry = ledger.get(segment, {})
        state = 'completed' if (ROOT / 'runs' / segment / 'DONE').exists() else entry.get('state', 'not_submitted')
        constraint, hardware = entry.get('gpu_constraint', 'A6000'), None
        if entry.get('job_id'):
            # Whitelist fields; scheduler workdirs, scripts and node names are never copied.
            output = subprocess.run(['scontrol', 'show', 'job', '-o', entry['job_id']],
                                    check=True, capture_output=True, text=True).stdout
            fields = dict(token.split('=', 1) for token in output.split() if '=' in token)
            state = fields.get('JobState', state).lower()
            constraint = fields.get('Features', constraint)
            node = fields.get('NodeList', '(null)')
            if state == 'running' and node != '(null)':
                info = subprocess.run(['sinfo', '-h', '-N', '-n', node, '-o', '%f'], check=True,
                                      capture_output=True, text=True).stdout.strip().splitlines()
                hardware = info[0] if info else None
        rows = records(segment, 'metrics.jsonl'); ev = records(segment, 'eval.jsonl')
        jobs.append(dict(experiment=name, segment=segment, status=state,
                         latest_update=rows[-1]['step'] if rows else None,
                         latest_evaluation_update=ev[-1]['step'] if ev else None,
                         gpu_constraint=constraint if entry else None, gpu_type=hardware))
    result = dict(snapshot_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  mystery1=dict(run=m1, minimum_validation_flow=minimum, final=metrics[-1],
                                nearby_early_evaluation=at(evaluations, 200_000),
                                final_evaluation=evaluations[-1],
                                final_to_minimum_validation_flow_ratio=metrics[-1]['validation/flow_loss']/minimum['validation/flow_loss']),
                  mystery4=dict(common_evaluation_update=common_step, conditions=comparisons), jobs=jobs)
    (OUT / 'findings.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    result = report()
    print('Findings refreshed; common M4 checkpoint:', result['mystery4']['common_evaluation_update'])
