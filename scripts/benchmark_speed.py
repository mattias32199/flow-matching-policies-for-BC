"""Finite block-double throughput tests; run with scripts/env.sh and a chosen CUDA_VISIBLE_DEVICES.

Results include contention if other processes use the selected GPU. Synthetic loss
checks validate execution, not task-learning equivalence between compute dtypes.
"""
import argparse,gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'ocbench/impls')]
parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','sim']);args=parser.parse_args()
def emit(row):
 print(json.dumps(row),flush=True)
 with (ROOT/'.local'/f'speed-{args.mode}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
if args.mode=='train':
 import jax,numpy as np
 from agents.fbc import FBCAgent,get_config
 batch={key:jax.device_put(value) for key,value in dict(observations=np.random.default_rng(1).normal(size=(1024,37)).astype(np.float32),actions=np.random.default_rng(2).normal(size=(1024,175)).astype(np.float32)).items()}
 for dtype in ['float32','bfloat16']:
  config=get_config();config.compute_dtype=dtype
  agent=FBCAgent.create(0,np.zeros((1,37),np.float32),np.zeros((1,175),np.float32),config)
  for _ in range(10):agent,info=agent.update(batch)
  jax.block_until_ready(info)
  start=time.monotonic()
  for _ in range(100):agent,info=agent.update(batch)
  jax.block_until_ready(info)
  elapsed=time.monotonic()-start
  emit(dict(mode='train',compute_dtype=dtype,updates=100,seconds=elapsed,updates_per_second=100/elapsed,loss=float(info['actor/actor_loss']),parameter_dtypes=sorted({str(x.dtype) for x in jax.tree_util.tree_leaves(agent.network.params)})))
  del agent;gc.collect()
else:
 from simulation import configure_iteration_warnings
 configure_iteration_warnings()
 from envs.streaming_data import generate_mjwarp_datasets
 for episodes,parallel in [(1000,512),(1000,1024),(2500,2500)]:
  emit(dict(mode='sim',event='start',episodes=episodes,parallel=parallel))
  start=time.monotonic()
  train,val,metrics=generate_mjwarp_datasets('block-double-task1-v0',episodes,0,98765,0,max_parallel_episodes=parallel)
  elapsed=time.monotonic()-start
  emit(dict(mode='sim',episodes=episodes,parallel=parallel,seconds=elapsed,episodes_per_second=episodes/elapsed,transitions=int(len(train['actions'])),success_rate=metrics['train/success_rate']))
  del train,val;gc.collect()
