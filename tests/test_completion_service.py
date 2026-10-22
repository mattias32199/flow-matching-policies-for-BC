from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import completion_service as service


class RecoveryPolicyTests(unittest.TestCase):
    def test_intentional_cancellation_and_unrelated_failures_are_not_restarted(self):
        for state in ['CANCELLED','COMPLETED','RUNNING','FAILED']:
            with patch.object(service,'job_fields',return_value={'JobState':state}),patch.object(service,'prepare') as prepare:
                self.assertFalse(service.recover('test',{'job_id':'1','state':'running'}))
                prepare.assert_not_called()

    def test_timeout_creates_a_distinct_checkpoint_segment_and_requeues_same_job(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'.local').mkdir();run=root/'runs/case/checkpoints';run.mkdir(parents=True)
            (run/'params_100.pkl').write_bytes(b'fixture')
            (root/'.local/individual-prepared-jobs.json').write_text(json.dumps([
                {'run':'case','command':['python','train.py','--run','case']}]))
            (root/'.local/final-results-plan.json').write_text(json.dumps({'m1':'done','m4':{}}))
            bundle=root/'.local/bundle';bundle.mkdir()
            with patch.object(service,'ROOT',root),patch.object(service,'job_fields',return_value={'JobState':'TIMEOUT'}), \
                 patch.object(service,'prepare',return_value=(bundle,{'step':100})), \
                 patch.object(service,'record') as record,patch.object(service,'log'),patch.object(service.subprocess,'run') as run_command:
                self.assertTrue(service.recover('case',{'job_id':'123','state':'running'}))
            aliases=json.loads((root/'.local/individual-replacements.json').read_text())
            self.assertEqual(aliases,{'case':'case-cont1'})
            prepared=json.loads((root/'.local/individual-prepared-jobs.json').read_text())
            self.assertIn('--resume-bundle',prepared[-1]['command'])
            run_command.assert_any_call(['scontrol','requeue','123'],check=True)
            record.assert_any_call('case',state='superseded',replacement_run='case-cont1')
