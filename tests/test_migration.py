import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import jax
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train import FBCAgent, get_config, Features, ActionNormalizer, saved_preprocessing
from utils.flax_utils import save_agent
from migration import read_bundle, restore_state
from prepare_migration import prepare


class MigrationTests(unittest.TestCase):
    def test_pinned_checkpoint_preserves_next_update_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'runs/source'
            (run / 'checkpoints').mkdir(parents=True)
            config = dict(env='block-double-task1-v0', mode='streaming', features='standardized',
                          seed=0, data_seed=0, episodes=1000, val_episodes=128,
                          batch_size=4, width=16, depth=2)
            (run / 'config.json').write_text(json.dumps(config))
            (run / 'collections.jsonl').write_text('{"index": 10}\n')
            observations = np.random.default_rng(2).normal(size=(4, 37)).astype(np.float32)
            actions = np.random.default_rng(3).normal(size=(4, 175)).astype(np.float32)
            features = Features(observations, 'standardized')
            normalizer = ActionNormalizer.fit(actions[:, :7], 1., 5.)
            np.savez(run / 'preprocessing.npz', center=features.center, scale=features.scale,
                     q_low=normalizer.q_low, q_high=normalizer.q_high, clip=normalizer.clip,
                     gripper_dims=np.flatnonzero(normalizer.gripper_dims))
            agent_config = get_config()
            agent_config.actor_hidden_dims = (16, 16)
            agent_config.batch_size = 4
            batch = dict(observations=features(observations), actions=actions)
            agent = FBCAgent.create(0, batch['observations'], actions, agent_config)
            for _ in range(3):
                agent, _ = agent.update(batch)
            save_agent(agent, str(run / 'checkpoints'), 3)
            # An interrupted newer write must not hide the last complete checkpoint.
            (run / 'checkpoints/params_4.pkl').write_bytes(b'\x80')
            bundle, manifest = prepare('source', root)
            self.assertEqual(manifest['step'], 3)
            self.assertEqual(len(manifest['rejected_candidates']), 1)
            args = SimpleNamespace(**config, steps=10)
            self.assertEqual(read_bundle(bundle, args)['step'], 3)
            restored = restore_state(agent, bundle, manifest)
            uninterrupted, _ = agent.update(batch)
            continued, _ = restored.update(batch)
            for expected, actual in zip(jax.tree_util.tree_leaves(uninterrupted),
                                        jax.tree_util.tree_leaves(continued)):
                np.testing.assert_array_equal(expected, actual)
            f, n = saved_preprocessing(bundle / 'preprocessing.npz', 'standardized')
            np.testing.assert_array_equal(f(observations + 2), features(observations + 2))
            np.testing.assert_array_equal(n.normalize(actions[:, :7]), normalizer.normalize(actions[:, :7]))
            (run / 'checkpoints/params_3.pkl').unlink()
            self.assertTrue((bundle / manifest['checkpoint']).exists())
            args.features = 'handcrafted'
            with self.assertRaisesRegex(ValueError, 'features'):
                read_bundle(bundle, args)


class FixedMigrationTests(unittest.TestCase):
    def test_fixed_checkpoint_keeps_validation_identity_and_restores_agent(self):
        import pickle
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'runs/fixed';(run/'checkpoints').mkdir(parents=True)
            config=dict(env='block-single-task1-v0',mode='fixed',features='handcrafted',seed=0,
                        data_seed=0,episodes=10000,val_episodes=128,batch_size=4,width=16,depth=2)
            (run/'config.json').write_text(json.dumps(config))
            (run/'collections.jsonl').write_text('{"index":0}\n')
            with (run/'checkpoints/params_3.pkl').open('wb') as file:
                pickle.dump({'agent':{'network':{'step':np.asarray(4)},'weights':np.asarray([1.,2.])}},file)
            bundle,manifest=prepare('fixed',root)
            self.assertFalse(manifest['validation_reset'])
            self.assertEqual(manifest['fresh_stream_index'],0)
            self.assertIn('fixed_dataset',manifest['continuation_kind'])
            self.assertEqual(read_bundle(bundle,SimpleNamespace(**config,steps=10))['step'],3)
