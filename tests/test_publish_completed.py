import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import publish_completed as publisher
from plot_data import export


class CompletedPublicationTests(unittest.TestCase):
    def test_missing_final_evaluation_is_excluded_and_report_can_be_updated_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'.local').mkdir();(root/'runs').mkdir()
            (root/'.local/final-results-plan.json').write_text(json.dumps({'m1':'m1-double-n10000-s0','m4':{}}))
            names=['m1-double-n10000-s0','m1-double-n10000-s1','m1-single-n10000-s0']
            for name in names:
                p=root/'runs'/name;p.mkdir()
                (p/'config.json').write_text(json.dumps({'run':name,'features':'handcrafted','episodes':10000,'seed':int(name[-1])}))
                (p/'DONE').touch()
                (p/'metrics.jsonl').write_text(json.dumps({'step':2_000_000,'training/flow_loss':.1,'validation/flow_loss':.2,'validation/action_mse':.3})+'\n')
                step=1_950_000 if 'single' in name else 2_000_000
                (p/'eval.jsonl').write_text(json.dumps({'step':step,'success':.5})+'\n')
            report=root/'RESULTS.md'
            report.write_text('# Mystery 1 and 4: completed five-run results\n\nThe requested **runs** completed.\n\n## Mystery 4: Results\n\n## Download the complete original plot measurements\n\nOld files.\n\n## Reproduction limits and validation\n')
            with patch.object(publisher,'ROOT',root),patch.object(publisher,'BASES',names):
                plan=publisher.completed_plan()
                self.assertEqual(set(plan['m4']),{'m1-double-n10000-s1'})
                index=export(root,plan,root/'results/plot-data',True)
                publisher.write_completed_section(index)
                first=report.read_text()
                publisher.write_completed_section(index)
                self.assertEqual(first,report.read_text())
                self.assertEqual(first.count('<!-- COMPLETED_EXTRAS_START -->'),1)
                self.assertIn('**2 of 10 experiments',first)
