import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prefetch_control as module


class PrefetchTests(unittest.TestCase):
    def prepare(self, directory):
        root = Path(directory)
        (root / '.local').mkdir()
        (root / '.local/demo-producer-config.json').write_text(json.dumps(dict(enabled=True, seeds=[1], partition='test', workers=2)))
        return root, SimpleNamespace(data_seed=1, mode='streaming', collection_cache=True)

    def test_repeated_consumers_submit_only_two_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root, args = self.prepare(directory)
            with patch.object(module, 'ROOT', root), patch.object(module.subprocess, 'run', return_value=SimpleNamespace(stdout='')) as query, patch.object(module, 'submit', side_effect=['998', '999']) as submit:
                for _ in range(4):
                    module._last_check = 0
                    module.ensure_producer(args, needed_index=36)
                self.assertEqual(query.call_count, 1)
                self.assertEqual(submit.call_count, 2)
                jobs = json.loads((root / '.local/demo-producer-job.json').read_text())['jobs']
                self.assertEqual({j['shard'] for j in jobs}, {0, 1})

    def test_running_pool_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root, args = self.prepare(directory)
            jobs = [dict(job_id=str(998+s), seed=1, shard=s) for s in range(2)]
            (root / '.local/demo-producer-job.json').write_text(json.dumps(dict(jobs=jobs, checked_at=0)))
            with patch.object(module, 'ROOT', root), patch.object(module.subprocess, 'run', return_value=SimpleNamespace(stdout='998 RUNNING\n999 RUNNING\n')) as query, patch.object(module, 'submit') as submit:
                module._last_check = 0
                module.ensure_producer(args)
                self.assertEqual(query.call_count, 1)
                submit.assert_not_called()
