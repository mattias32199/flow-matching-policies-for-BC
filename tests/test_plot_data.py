import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from plot_data import export, load, branches, check_complete, unified


class PlotDataTests(unittest.TestCase):
    def setup_root(self, directory):
        root=Path(directory);(root/'runs').mkdir()
        plan={'m1':'m1-test','m4':{'m4-test':'m4-test-cont2'}}
        for name in ['m1-test','m4-test','m4-test-cont1','m4-test-cont2']:
            p=root/'runs'/name;p.mkdir()
            cfg={'run':name,'features':'handcrafted','seed':0,'private_path':'must-not-export'}
            if name.endswith('cont2'):cfg.update(resume_step=1_400_000,resume_source_run='m4-test',validation_reset=True)
            (p/'config.json').write_text(json.dumps(cfg))
            rows=[{'step':1_400_000,'training/flow_loss':.2,'validation/flow_loss':.3,'validation/action_mse':.4},
                  {'step':2_000_000,'training/flow_loss':.1,'validation/flow_loss':.2,'validation/action_mse':.3}]
            (p/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
            (p/'eval.jsonl').write_text(json.dumps({'step':2_000_000,'success':.5,'mean_success_steps':None})+'\n')
            if name in ['m1-test','m4-test-cont2']:(p/'DONE').touch()
        return root,plan

    def test_archive_is_lossless_pickle_free_and_keeps_superseded_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            root,plan=self.setup_root(directory)
            index=export(root,plan,root/'out',True)
            meta,runs=load(root/'out/m4-test.npz')
            self.assertEqual(set(runs),{'m4-test','m4-test-cont1','m4-test-cont2'})
            self.assertIsNone(runs['m4-test-cont2']['streams']['eval.jsonl'][0]['mean_success_steps'])
            self.assertNotIn('private_path',str(meta))
            segments=branches(meta,runs,'metrics.jsonl')
            self.assertEqual([x['step'] for x in segments[0][1]],[1_400_000])
            self.assertEqual([x['step'] for x in segments[1][1]],[2_000_000])
            with np.load(root/'out/m4-test.npz',allow_pickle=False) as f:
                self.assertTrue(all(f[k].dtype.kind!='O' for k in f.files))
                np.testing.assert_array_equal(f['history__metrics__step'], [1_400_000, 2_000_000])
                self.assertEqual([r['step'] for r in unified(meta,runs,'metrics.jsonl')], [1_400_000, 2_000_000])
            self.assertEqual(index['runs']['m4-test']['bytes'],(root/'out/m4-test.npz').stat().st_size)

    def test_missing_final_rollout_or_done_blocks_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root,plan=self.setup_root(directory)
            (root/'runs/m4-test-cont2/eval.jsonl').write_text(json.dumps({'step':1_950_000,'success':.5})+'\n')
            with self.assertRaises(RuntimeError):check_complete(root,plan)
            self.assertFalse((root/'out').exists())


class AncestryTests(unittest.TestCase):
    def test_multiple_restarts_keep_every_selected_prefix_and_no_overlap(self):
        runs={
          'source':{'config':{},'streams':{'metrics.jsonl':[{'step':1},{'step':4},{'step':6}]}},
          'middle':{'config':{'resume_source_run':'source','resume_step':4},
                    'streams':{'metrics.jsonl':[{'step':5},{'step':7},{'step':9}]}},
          'last':{'config':{'resume_source_run':'middle','resume_step':7},
                  'streams':{'metrics.jsonl':[{'step':8},{'step':10}]}}}
        result=branches({'experiment':'source','selected_segment':'last'},runs,'metrics.jsonl')
        self.assertEqual([r['step'] for n,rows in result for r in rows],[1,4,5,7,8,10])
