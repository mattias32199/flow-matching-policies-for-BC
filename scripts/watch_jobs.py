"""Wake the existing Codex thread on run completion, failure, or a stalled service."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def progress(root, message):
    path = root / 'PROGRESS.md'
    if path.exists():
        path.read_text()
    with path.open('a') as file:
        file.write(f'- {timestamp()}: Completion watcher: {message}\n')


def observe(root, now=None, training_stall=1200, initial_collection_stall=21600):
    now = time.time() if now is None else now
    queue_file = root / '.local' / 'queue-status.json'
    queue = json.loads(queue_file.read_text())
    completed = sorted(p.parent.name for p in (root / 'runs').glob('*/DONE'))
    alerts = {}
    active = [name for name in queue.get('active', {}).values() if name]
    if now - queue_file.stat().st_mtime > 180:
        alerts['supervisor-stale'] = 'Supervisor heartbeat is older than 3 minutes.'
    for name in active:
        directory = root / 'runs' / name
        metrics = directory / 'metrics.jsonl'
        anchor = metrics if metrics.exists() else directory / 'config.json'
        if not anchor.exists():
            continue  # Supervisor may have just spawned the process.
        threshold = training_stall if metrics.exists() else initial_collection_stall
        age = now - anchor.stat().st_mtime
        if age > threshold:
            alerts[f'stalled:{name}'] = f'{name} has no new training record for {int(age)} seconds.'
    mirror = root / '.local' / 'wandb-mirror.log'
    if active and mirror.exists() and now - mirror.stat().st_mtime > 600:
        alerts['wandb-mirror-stale'] = 'W&B mirror has not updated its log for over 10 minutes.'
    return dict(completed=completed, failures=queue.get('failures', []), alerts=alerts,
                active=active, pending=queue.get('pending', 0))


def collect_events(observation, state):
    # Existing diagnostic completions/failures establish a baseline, not new alerts.
    if not state:
        state.update(completed_seen=observation['completed'], failures_seen=observation['failures'],
                     alerts_active=[], pending=[], delivered=0)
    events = []
    for name in sorted(set(observation['completed']) - set(state['completed_seen'])):
        events.append(dict(kind='completed', run=name, detail=f'{name}: DONE marker exists.'))
    for name in sorted(set(observation['failures']) - set(state['failures_seen'])):
        events.append(dict(kind='failed', run=name, detail=f'{name}: supervisor reports failure.'))
    for key, detail in observation['alerts'].items():
        if key not in state['alerts_active']:
            events.append(dict(kind='alert', key=key, detail=detail))
    state['completed_seen'] = observation['completed']
    state['failures_seen'] = observation['failures']
    state['alerts_active'] = list(observation['alerts'])
    state['pending'].extend(events)
    return events


def notification(events):
    details = '; '.join(event['detail'] for event in events)
    return (
        'OCBench watcher event: ' + details + '\n'
        'Continue this existing reproduction task. Inspect the queue and exact run logs. '
        'For completed experiments, analyze evaluations/checkpoints, update RESULTS and run evidence, '
        'and commit/push completed results using Haoran Huan <yzhhr@outlook.com>. '
        'For failures/stalls, diagnose and recover within the existing authorized scope. '
        'Preserve unaffected GPU jobs and PROGRESS history. Do not create timed metric commits '
        'or another watcher. Report material findings to the user.'
    )


def deliver(root, thread, events, queue_command='codex'):
    prompt = notification(events)
    result = subprocess.run([queue_command, 'queue', '--thread', thread, '--message', prompt],
                            cwd=root, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError(f'Codex queue returned exit {result.returncode}: {result.stderr[-500:]}')
    return result.stdout.strip()


def poll(root, thread, state_path, queue_command='codex', sender=deliver):
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    observation = observe(root)
    events = collect_events(observation, state)
    # Persist pending events before delivery; failed queue calls are retried next poll.
    write_json(state_path, state)
    if state['pending']:
        response = sender(root, thread, state['pending'], queue_command)
        sent = list(state['pending'])
        state['pending'] = []
        state['delivered'] += len(sent)
        write_json(state_path, state)
        with (root / '.local' / 'watch-events.jsonl').open('a') as file:
            file.write(json.dumps(dict(time=timestamp(), events=sent, acknowledgement=response)) + '\n')
        progress(root, f'Queued follow-up for {len(sent)} event(s) to the existing thread.')
    write_json(root / '.local' / 'watcher-status.json', dict(
        checked_at=timestamp(), active=observation['active'], pending_jobs=observation['pending'],
        delivered_events=state['delivered'], pending_notifications=len(state['pending']),
        notification_method='codex queue to existing thread',
    ))
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thread', default=os.environ.get('CODEX_THREAD_ID'))
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--queue-command', default='codex')
    args = parser.parse_args()
    if not args.thread:
        parser.error('Specify --thread or set CODEX_THREAD_ID.')
    state_path = ROOT / '.local' / 'watcher-state.json'
    last_error = None
    progress(ROOT, 'Started persistent event watcher; completion/failure/stall events queue this thread.')
    while True:
        try:
            events = poll(ROOT, args.thread, state_path, args.queue_command)
            if last_error:
                progress(ROOT, 'Watcher recovered; pending notifications retried successfully.')
            last_error = None
            print(json.dumps(dict(checked_at=timestamp(), new_events=len(events))), flush=True)
        except Exception as error:
            message = f'{type(error).__name__}: {error}'
            if message != last_error:
                progress(ROOT, f'Watcher error, will retry: {message}')
            last_error = message
            print(message, flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
