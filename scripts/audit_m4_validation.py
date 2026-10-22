"""Evaluate saved M4 checkpoints on fixed, seed-reconstructed validation datasets.

Never updates an agent; preserves original training records. Private raw arrays stay local.
"""
import gc
import json
from pathlib import Path
import pickle

import flax.serialization
import jax
import numpy as np
from train import ROOT, FBCAgent, get_config, saved_preprocessing, dataset, generate_mjwarp_datasets
from simulation import configure_iteration_warnings
from prepare_migration import sha256


def emit(path, value):
    with path.open('a') as f:f.write(json.dumps(value, allow_nan=False)+'\n')
    print(json.dumps(value),flush=True)


def main():
    configure_iteration_warnings()
    if not any(d.platform=='gpu' for d in jax.devices()):raise RuntimeError('GPU required')
    private=ROOT/'.local/m4-validation-audit';private.mkdir(exist_ok=True)
    out=ROOT/'results/validation/m4-validation-audit.jsonl'
    out.parent.mkdir(exist_ok=True)
    plan=json.loads((ROOT/'.local/final-results-plan.json').read_text())
    sets={}
    descriptions=[('original512',0,512),('original2500',0,2500),('resumed2500',3072,2500)]
    for label,index,parallel in descriptions:
        path=private/(label+'.npz')
        if not path.exists():
            train,val,metrics=generate_mjwarp_datasets('block-double-task1-v0',1000,128,0,index,max_parallel_episodes=parallel)
            del train
            np.savez_compressed(path,**val);del val
            (private/(label+'.json')).write_text(json.dumps(metrics))
            gc.collect()
        sets[label]=path
    completed=set()
    if out.exists():
        completed={(r['experiment'],r['checkpoint_step'],r['validation_set']) for r in map(json.loads,out.read_text().splitlines())}
    for source,selected in plan['m4'].items():
        config=json.loads((ROOT/'runs'/source/'config.json').read_text())
        resumed=json.loads((ROOT/'runs'/selected/'config.json').read_text())
        f,n=saved_preprocessing(ROOT/'runs'/source/'preprocessing.npz',config['features'])
        other_f,other_n=saved_preprocessing(ROOT/'runs'/selected/'preprocessing.npz',config['features'])
        for a,b in [(f.center,other_f.center),(f.scale,other_f.scale),(n.q_low,other_n.q_low),(n.q_high,other_n.q_high)]:
            np.testing.assert_array_equal(a,b)
        cfg=get_config();cfg.actor_hidden_dims=(config['width'],)*config['depth'];cfg.batch_size=config['batch_size']
        checkpoints=list((ROOT/'runs'/source/'checkpoints').glob('params_*.pkl'))+list((ROOT/'runs'/selected/'checkpoints').glob('params_*.pkl'))
        checkpoints.sort(key=lambda p:int(p.stem.split('_')[-1]))
        template=None
        for checkpoint in checkpoints:
            step=int(checkpoint.stem.split('_')[-1])
            if all((source,step,label) in completed for label in sets):continue
            with checkpoint.open('rb') as file:state=pickle.load(file)['agent']
            if step==resumed['resume_step']:
                assert sha256(checkpoint)==resumed['resume_checkpoint_sha256']
            if template is None:
                template=FBCAgent.create(config['seed'],np.zeros((1,37),np.float32),np.zeros((1,175),np.float32),cfg)
            agent=flax.serialization.from_state_dict(template,state)
            assert int(agent.network.step)==step+1
            # Every saved tensor including Adam moments/count and agent RNG must survive restore exactly.
            restored=flax.serialization.to_state_dict(agent)
            a=jax.tree_util.tree_leaves(state);b=jax.tree_util.tree_leaves(restored)
            assert len(a)==len(b)
            for x,y in zip(a,b):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
            del state,restored,a,b
            gc.collect()
            for label,path in sets.items():
                if (source,step,label) in completed:continue
                with np.load(path,allow_pickle=False) as raw:val=dataset(dict(raw),n)
                rng=np.random.default_rng(20261022);losses=[];mses=[]
                for batch_index in range(32):
                    ids=rng.integers(val.size,size=1024)
                    batch=val.sample(1024,idxs=ids,keys=('observations','actions'))
                    batch['observations']=f(batch['observations'])
                    key=jax.random.PRNGKey(batch_index+20261022)
                    _,info=agent.total_loss(batch,grad_params=None,rng=key)
                    metrics=agent.compute_metrics(batch,rng=key)
                    losses.append(float(info['actor/actor_loss']));mses.append(float(metrics['actor/mse']))
                row=dict(experiment=source,checkpoint_step=step,validation_set=label,
                    dataset_sha256=sha256(path),transitions=val.size,successful_episodes=int(val['terminals'].sum()),
                    batches=32,batch_size=1024,flow_loss=float(np.mean(losses)),
                    flow_loss_batch_se=float(np.std(losses,ddof=1)/np.sqrt(len(losses))),
                    action_mse=float(np.mean(mses)),action_mse_batch_se=float(np.std(mses,ddof=1)/np.sqrt(len(mses))),
                    restored_state_exact=True,preprocessing_exact=True,
                    checkpoint_sha256=sha256(checkpoint))
                emit(out,row);del val
            del agent;gc.collect()
    print('AUDIT_DONE',flush=True)


if __name__=='__main__':main()
