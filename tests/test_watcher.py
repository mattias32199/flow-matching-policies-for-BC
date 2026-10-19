import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from watch_jobs import collect_events, observe, poll


class WatcherTest(unittest.TestCase):
    def setup_root(self, directory):
        root=Path(directory)
        (root/'.local').mkdir()
        (root/'runs').mkdir()
        (root/'.local/queue-status.json').write_text(json.dumps(dict(active={},pending=1,failures=[])))
        return root

    def test_baseline_diagnostics_ignored_new_completion_delivered_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.setup_root(directory)
            old=root/'runs/smoke-v3';old.mkdir();(old/'DONE').touch()
            state=root/'.local/watcher-state.json'
            calls=[]
            def sender(root,thread,events,command):calls.append(events.copy());return 'queued'
            poll(root,'same-thread',state,sender=sender)
            self.assertEqual(calls,[])
            run=root/'runs/m1-full';run.mkdir();(run/'DONE').touch()
            poll(root,'same-thread',state,sender=sender)
            poll(root,'same-thread',state,sender=sender)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][0]['run'],'m1-full')

    def test_failed_delivery_survives_restart_and_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.setup_root(directory);path=root/'.local/watcher-state.json'
            poll(root,'thread',path)
            (root/'.local/queue-status.json').write_text(json.dumps(dict(active={},pending=1,failures=['m4'])))
            def fail(*args):raise RuntimeError('daemon unavailable')
            with self.assertRaises(RuntimeError):poll(root,'thread',path,sender=fail,completion_only=False)
            self.assertEqual(len(json.loads(path.read_text())['pending']),1)
            delivered=[]
            def success(root,thread,events,command):delivered.extend(events);return 'queued'
            poll(root,'thread',path,sender=success,completion_only=False)
            self.assertEqual(delivered[0]['kind'],'failed')
            self.assertEqual(json.loads(path.read_text())['pending'],[])

    def test_default_does_not_wake_on_failure_or_stall(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.setup_root(directory);path=root/'.local/watcher-state.json'
            poll(root,'thread',path)
            (root/'.local/queue-status.json').write_text(json.dumps(dict(active={},pending=1,failures=['m4'])))
            def unexpected(*args):self.fail('Non-completion must not invoke Codex')
            poll(root,'thread',path,sender=unexpected)
            self.assertEqual(json.loads(path.read_text())['pending'],[])

    def test_stall_alert_can_recur_after_recovery(self):
        state={}
        base=dict(completed=[],failures=[],alerts={})
        collect_events(base,state)
        self.assertEqual(len(collect_events(dict(base,alerts={'stalled:a':'stalled'}),state)),1)
        self.assertEqual(collect_events(dict(base,alerts={'stalled:a':'stalled'}),state),[])
        collect_events(base,state)
        self.assertEqual(len(collect_events(dict(base,alerts={'stalled:a':'stalled'}),state)),1)

    def test_initial_collection_is_not_flagged_as_training_stall(self):
        with tempfile.TemporaryDirectory() as directory:
            root=self.setup_root(directory)
            (root/'.local/queue-status.json').write_text(json.dumps(dict(active={'0':'large'},pending=0,failures=[])))
            run=root/'runs/large';run.mkdir();(run/'config.json').write_text('{}')
            now=(run/'config.json').stat().st_mtime+1800
            os.utime(root/'.local/queue-status.json',(now,now))
            result=observe(root,now=now)
            self.assertNotIn('stalled:large',result['alerts'])
            (run/'metrics.jsonl').write_text('{}\n')
            os.utime(run/'metrics.jsonl',(now-1800,now-1800))
            self.assertIn('stalled:large',observe(root,now=now)['alerts'])


if __name__=='__main__':unittest.main()
