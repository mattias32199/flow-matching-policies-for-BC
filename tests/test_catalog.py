import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from catalog import export_catalog, run_id, read_records
from wandb_mirror import payload, publish_records


class CatalogTest(unittest.TestCase):
    def test_portable_records_and_exact_eval_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = root / 'runs' / 'smoke-test'
            local.mkdir(parents=True)
            config = dict(run='smoke-test', seed=0, eval_episodes=4, api_key='must-not-export',
                          hostname='private-machine', working_directory='/private/path')
            (local / 'config.json').write_text(json.dumps(config))
            (local / 'eval.jsonl').write_text(json.dumps(dict(step=2, success=0., secret='must-not-export'))+'\n')
            (local / 'DONE').write_text('done')
            (root / '.local').mkdir()
            (root / '.local/wandb-links.json').write_text(json.dumps({'smoke-test': {
                'id':run_id(config),'url':'https://example.test/run'}}))
            entries = export_catalog(root)
            public = root / 'results/runs/smoke-test'
            content = '\n'.join(p.read_text() for p in public.iterdir())
            self.assertNotIn('must-not-export', content)
            self.assertNotIn('private-machine', content)
            self.assertNotIn('/private/path', content)
            self.assertIn('eval.jsonl#L1', content)
            self.assertIn('https://example.test/run', content)
            self.assertEqual(entries['smoke-test']['status'], 'completed')
            self.assertEqual(len([v for v in entries.values() if v['status']=='queued']),10)

    def test_stale_identity_is_not_linked(self):
        self.assertNotEqual(run_id(dict(run='m4', steps=100)),run_id(dict(run='m4', steps=2000000)))
        self.assertEqual(run_id(dict(run='m4', api_key='one')),run_id(dict(run='m4', api_key='two')))

    def test_late_evaluation_keeps_gradient_axis_and_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'metrics.jsonl').write_text(json.dumps({'step':1000,'training/flow_loss':.1})+'\n')
            class FakeRun:
                def __init__(self): self.rows=[]
                def log(self, row): self.rows.append(row)
            run=FakeRun()
            cursors={}
            self.assertEqual(publish_records(run,root,cursors),1)
            (root/'eval.jsonl').write_text(json.dumps({'step':500,'success':0.,'mean_success_steps':None})+'\n')
            self.assertEqual(publish_records(run,root,cursors),1)
            self.assertEqual(run.rows[-1]['gradient_step'],500)
            self.assertEqual(run.rows[-1]['evaluation/success'],0.)
            self.assertEqual(publish_records(run,root,cursors),0)
            (root/'eval.jsonl').write_text('')
            with self.assertRaises(RuntimeError): publish_records(run,root,cursors)

    def test_corrupt_history_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'log.jsonl'
            path.write_text('{"step": 1}\n{"step":')
            self.assertEqual(read_records(path), [{'step': 1}])
            path.write_text('{"step": 1}\ninvalid\n')
            with self.assertRaises(json.JSONDecodeError): read_records(path)

    def test_payload_ids_include_step_and_drop_extra_fields(self):
        first=payload('metrics.jsonl',{'step':1,'training/flow_loss':.1,'hostname':'private'})
        second=payload('metrics.jsonl',{'step':2,'training/flow_loss':.1})
        self.assertNotIn('hostname',first)
        self.assertNotEqual(first['record_uid'],second['record_uid'])


if __name__ == '__main__': unittest.main()
