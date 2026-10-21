import json
import stat
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import private_slurm


class PrivateSubmissionTest(unittest.TestCase):
    def test_opaque_private_launcher_and_workdir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'project'
            (root / '.local').mkdir(parents=True)
            root.chmod(0o700)
            with patch.object(private_slurm, 'ROOT', root), patch.object(private_slurm.subprocess, 'run', return_value=SimpleNamespace(stdout='123\n')) as run:
                self.assertEqual(private_slurm.submit(['python', 'private_task.py'], 'test'), '123')
            arguments = run.call_args.args[0]
            launcher = Path(arguments[-1])
            self.assertEqual(run.call_args.kwargs['cwd'], launcher.parent)
            self.assertEqual(stat.S_IMODE(launcher.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(launcher.parent.stat().st_mode), 0o700)
            self.assertNotIn('private_task', str(launcher))
            self.assertNotIn('project', next(x for x in arguments if x.startswith('--chdir=')))
            self.assertIn('private_task.py', launcher.read_text())
