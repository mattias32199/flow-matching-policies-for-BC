"""Export portable scalar metrics only; never copy logs, configs, or checkpoints."""
import csv
import json
from pathlib import Path

from catalog import export_catalog

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path):
    records = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Writer may be in the middle of its final record.
    return records


def summarize():
    rows = []
    status = []
    catalog = export_catalog()
    for config_file in sorted((ROOT / 'runs').glob('*/config.json')):
        config = json.loads(config_file.read_text())
        if config['run'].startswith('smoke'):
            continue
        run = config_file.parent
        records = read_jsonl(run / 'metrics.jsonl')
        evaluations = read_jsonl(run / 'eval.jsonl')
        latest = records[-1] if records else {}
        evaluation = evaluations[-1] if evaluations else {}
        status.append(dict(run=config['run'], step=latest.get('step', 0),
                           train=latest.get('training/flow_loss'), val=latest.get('validation/flow_loss'),
                           mse=latest.get('validation/action_mse'), success=evaluation.get('success'),
                           eval_step=evaluation.get('step'), done=(run / 'DONE').exists()))
        for record in records:
            rows.append(dict(run=config['run'], **record))
        for record in evaluations:
            rows.append(dict(run=config['run'], **{f'evaluation/{k}' if k != 'step' else k: v
                                                   for k, v in record.items()}))
    output = ROOT / 'results' / 'curves.csv'
    keys = ['run', 'step'] + sorted({key for row in rows for key in row} - {'run', 'step'})
    with output.open('w', newline='') as file:
        writer = csv.DictWriter(file, keys, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    def number(value):
        return '—' if value is None else f'{value:.4f}'
    tracking = json.loads((ROOT / 'results' / 'tracking.json').read_text())
    tracking_link = (f"[W&B project]({tracking['project_url']}) · [W&B run catalog]({tracking['catalog_url']})"
                     if tracking['status'] == 'connected' else 'W&B: authenticated upload pending.')
    report = ['<!-- LIVE_RESULTS_START -->', '## Live experiment status', '', tracking_link, '',
              'Pilot runs validate execution only. Reproduction runs target 2,000,000 updates; '
              'partial curves do not establish either mystery.', '',
              '| Run records | W&B | Updates | Train flow | Val flow | Val action MSE | Success | Eval update | Finished |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for row in status:
        url = catalog[row['run']]['wandb_url']
        remote = f'[Charts]({url})' if url else 'pending upload'
        report.append(f"| [{row['run']}](results/runs/{row['run']}/README.md) | {remote} | {row['step']} | {number(row['train'])} | {number(row['val'])} | "
                      f"{number(row['mse'])} | {number(row['success'])} | {row['eval_step'] or '—'} | "
                      f"{'yes' if row['done'] else 'no'} |")
    report += ['', 'Blank evaluation cells mean no recorded evaluation yet. Undefined duration '
               'aggregates at zero success are stored as null; success itself remains zero.',
               '<!-- LIVE_RESULTS_END -->']
    target = ROOT / 'RESULTS.md'
    text = target.read_text()
    begin = '<!-- LIVE_RESULTS_START -->'
    end = '<!-- LIVE_RESULTS_END -->'
    block = '\n'.join(report)
    if begin in text:
        before, rest = text.split(begin, 1)
        _, after = rest.split(end, 1)
        text = before + block + after
    else:
        text += '\n' + block + '\n'
    target.write_text(text)
    return len(rows)


if __name__ == '__main__':
    print(f'Exported {summarize()} metric rows.')
