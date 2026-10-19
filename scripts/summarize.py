"""Export portable scalar metrics only; never copy logs, configs, or checkpoints."""
import csv
import json
from pathlib import Path

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
    for config_file in sorted((ROOT / 'runs').glob('*/config.json')):
        config = json.loads(config_file.read_text())
        if config['run'].startswith('smoke'):
            continue
        run = config_file.parent
        records = read_jsonl(run / 'metrics.jsonl')
        evaluations = read_jsonl(run / 'eval.jsonl')
        for record in records:
            rows.append(dict(run=config['run'], **record))
        for record in evaluations:
            rows.append(dict(run=config['run'], **{f'evaluation/{k}' if k != 'step' else k: v
                                                   for k, v in record.items()}))
    output = ROOT / 'results' / 'curves.csv'
    keys = ['run', 'step'] + sorted({key for row in rows for key in row} - {'run', 'step'})
    with output.open('w', newline='') as file:
        writer = csv.DictWriter(file, keys)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == '__main__':
    print(f'Exported {summarize()} metric rows.')
