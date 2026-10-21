import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import individual_jobs


class IndividualJobsTests(unittest.TestCase):
    def test_prepare_is_private_resource_specific_and_does_not_submit(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            parent.chmod(0o700)
            root = parent / 'project'
            (root / '.local').mkdir(parents=True)
            root.chmod(0o700)
            (root / 'PROGRESS.md').write_text('# TODO\n')
            plan = root / '.local/plan.json'
            plan.write_text(json.dumps([dict(run=name, command=['python', 'train.py'])
                for name in ['m4-fourier1-s0', 'm1-single-n10000-s0', 'm1-single-n50000-s0']]))
            with patch.object(individual_jobs, 'ROOT', root), \
                 patch.object(individual_jobs, 'PLAN', plan), \
                 patch.object(individual_jobs, 'PREPARED', root / '.local/prepared.json'):
                jobs = individual_jobs.prepare()
                self.assertEqual([j['memory'] for j in jobs], ['48G', '64G', '96G'])
                self.assertEqual(individual_jobs.prepare(), jobs)
            for job in jobs:
                path = Path(job['launcher'])
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)
                content = path.read_text()
                self.assertTrue(content.startswith('#!/bin/bash\n'))
                self.assertIn('#SBATCH --time=2-00:00:00\n', content)
                self.assertIn('#SBATCH --partition=deepaklong\n', content)
                self.assertEqual(path.name, job['run'] + '.sbatch')
                self.assertIn('#SBATCH --constraint=A6000|A5000', content)

    def test_active_includes_pending_and_excludes_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / '.local').mkdir()
            (root / '.local/individual-job-status.json').write_text(json.dumps({
                'a': {'state': 'submitted'}, 'b': {'state': 'running'},
                'c': {'state': 'failed'}}))
            self.assertEqual(individual_jobs.active_runs(root), {'a', 'b'})
