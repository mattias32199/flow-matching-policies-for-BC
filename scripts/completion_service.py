"""Durable Slurm service: recover timed-out segments and publish the completed five-run report."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collection_cache import locked, write_json
from individual_jobs import ROOT, resolve_run, record, log, submit_replacement
from prepare_migration import prepare


def job_fields(job_id):
    result = subprocess.run(['scontrol','show','job',job_id,'-o'],capture_output=True,text=True)
    if result.returncode == 0:
        return dict(token.split('=',1) for token in result.stdout.split() if '=' in token) | {'scheduler_live': True}
    result = subprocess.run(['sacct','-n','-X','-j',job_id,'-o','State','-P'],capture_output=True,text=True,check=True)
    state=result.stdout.strip().splitlines()
    return {'JobState':state[0].split('|')[0] if state else 'UNKNOWN', 'scheduler_live': False}


def recover(name, item):
    fields=job_fields(item['job_id']);state=fields.get('JobState','UNKNOWN')
    # An intentional user cancellation is respected; never automatically undo it.
    restart = state in ('TIMEOUT','PREEMPTED','NODE_FAIL') or (
        state=='FAILED' and item.get('state')=='checkpoint_continuation_required')
    if not restart:
        return False
    directory=ROOT/'runs'/name
    has_checkpoint=bool(list((directory/'checkpoints').glob('params_*.pkl')))
    bundle,manifest=prepare(name) if has_checkpoint else (None, {'step': 0})
    if manifest['step']>=2_000_000:
        log(f'{name} saved2M but lacks completion gate; requires final evaluation audit.')
        return False
    base=name.split('-cont')[0]
    prepared_path=ROOT/'.local/individual-prepared-jobs.json'
    prepared=json.loads(prepared_path.read_text())
    number=1
    while any(job['run']==f'{base}-cont{number}' for job in prepared):number+=1
    new=f'{base}-cont{number}'
    old=next(job for job in prepared if job['run']==name)
    job=dict(old);job.update(run=new,job_name=new)
    cmd=list(old['command']);cmd[cmd.index('--run')+1]=new
    target=int(cmd[cmd.index('--steps')+1]) if '--steps' in cmd else 2_000_000
    if target != 2_000_000:
        raise ValueError('Scientific jobs must retain the absolute 2M-update target')
    # With no new checkpoint, retain the original resume bundle (or start at zero).
    if bundle is not None and '--resume-bundle' in cmd:cmd[cmd.index('--resume-bundle')+1]=str(bundle.relative_to(ROOT))
    elif bundle is not None:cmd+=['--resume-bundle',str(bundle.relative_to(ROOT))]
    elif '--resume-bundle' in cmd:
        original=Path(cmd[cmd.index('--resume-bundle')+1])
        if not original.is_absolute():original=ROOT/original
        manifest=json.loads((original/'manifest.json').read_text())
    job['command']=cmd;prepared.append(job);write_json(prepared_path,prepared)
    aliases_path=ROOT/'.local/individual-replacements.json'
    aliases=json.loads(aliases_path.read_text()) if aliases_path.exists() else {}
    aliases[name]=new;write_json(aliases_path,aliases)
    record(name,state='superseded',replacement_run=new)
    record(new,state='requeue_pending',job_id=item['job_id'],resume_step=manifest['step'],replaces=name)
    plan_path=ROOT/'.local/final-results-plan.json';plan=json.loads(plan_path.read_text())
    for source,selected in plan['m4'].items():
        if selected==name:plan['m4'][source]=new
    write_json(plan_path,plan)
    if fields.get('scheduler_live', True):
        subprocess.run(['scontrol','update','JobId='+item['job_id'],'JobName='+new],capture_output=True,text=True)
    requeue_pending(new, item)
    # Requeue has a short automatic delay; this eligibility update is best-effort.
    subprocess.run(['scontrol','update','JobId='+item['job_id'],'StartTime=now'],capture_output=True,text=True)
    log(f'Automatic continuation {name}→{new} at{manifest["step"]}; scheduler allocation recorded in individual ledger.')
    return True


def requeue_pending(name, item):
    """Retry a prepared restart without interrupting an already running replacement."""
    fields=job_fields(item['job_id'])
    state=fields.get('JobState','UNKNOWN')
    if state in ('RUNNING','PENDING','CONFIGURING','SUSPENDED'):
        record(name,state='running' if state=='RUNNING' else 'submitted')
        return
    if state not in ('TIMEOUT','PREEMPTED','NODE_FAIL','FAILED'):
        raise RuntimeError(f'Recovery requires scheduler reconciliation: {state}')
    if not fields.get('scheduler_live', True):
        submit_replacement(name)
        return
    result = subprocess.run(['scontrol','requeue',item['job_id']],capture_output=True,text=True)
    if result.returncode:
        if 'Invalid job id' in result.stderr:
            submit_replacement(name)
            return
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
    record(name,state='submitted')


def notify_once(message, marker):
    if marker.exists():return
    thread=json.loads((ROOT/'.local/completion-thread.json').read_text())['thread']
    result=subprocess.run(['codex','queue','--thread',thread,'--message',message],
                          cwd=ROOT,capture_output=True,text=True,timeout=45)
    if result.returncode==0:marker.write_text(str(time.time()))
    else:log('Completion notification pending: '+result.stderr[-300:])


def tick():
    from finalize_results import finalize
    from plot_data import check_complete
    path=ROOT/'.local/individual-job-status.json'
    data=json.loads(path.read_text())
    # Main comparisons are considered first; ordinary Slurm fair-share priority remains unchanged.
    for name,item in sorted(data.items(),key=lambda x:(not x[0].startswith('m4-') or 'fourier' in x[0],x[0])):
        if item.get('state')=='superseded' or (ROOT/'runs'/name/'DONE').exists():continue
        try:
            if item.get('state')=='requeue_pending':
                requeue_pending(name,item)
            else:
                recover(name,item)
        except Exception as error:
            log(f'Recovery attention for {name}: {type(error).__name__}: {error}')
    plan=json.loads((ROOT/'.local/final-results-plan.json').read_text())
    publication=ROOT/'.local/final-publication-state.json'
    published=json.loads(publication.read_text()).get('published',False) if publication.exists() else False
    if not published:
        try:check_complete(ROOT,plan)
        except RuntimeError:pass
        else:
            try:
                with locked(ROOT/'.local/final-publication.lock'):
                    result=finalize(plan,publish=True)
            except Exception as error:
                notify_once('All requested five experiments completed2M, but final report/NPZ publication needs attention: '+type(error).__name__+': '+str(error)+'. Inspect .local/final-publication-state.json and completion-service logs; finish the authorized GitHub publication, preserving all data.',ROOT/'.local/final-publication-needs-attention')
                raise
            notify_once('The completed M1 and four main M4 experiments have reached2M. RESULTS.md and five per-run NPZ files were committed/pushed and verified: '+result['commit']+'. Inspect and report final findings; preserve other jobs.',ROOT/'.local/final-publication-notified')
    if publication.exists():
        published_state=json.loads(publication.read_text())
        if published_state.get('published'):
            notify_once('The requested five-run report and per-run NPZ files were published: '+published_state['commit']+'. Report the completed findings; preserve other jobs.', ROOT/'.local/final-publication-notified')
    from publish_completed import publish
    publish(publish=True)
    # Keep recovering the other scientific jobs after the main report is published.
    selected={resolve_run(job['run']) for job in json.loads((ROOT/'.local/individual-launch-plan.json').read_text())}
    return all((ROOT/'runs'/name/'DONE').exists() for name in selected)


def reserve_successor():
    """A pending afterany job survives even if this service is killed unexpectedly."""
    path=ROOT/'.local/completion-service-job.json'
    info=json.loads(path.read_text())
    current=os.environ['SLURM_JOB_ID']
    if info.get('predecessor_job_id')==current and info.get('job_id')!=current:
        return info['job_id']
    result=subprocess.run(['sbatch','--parsable','--dependency=afterany:'+current,info['launcher']],
                          capture_output=True,text=True,check=True,timeout=45)
    successor=result.stdout.strip().split(';')[0]
    if not successor.isdigit():raise ValueError('Unrecognized service successor job ID')
    info.update(job_id=successor,predecessor_job_id=current)
    write_json(path,info)
    log(f'Completion service {current} reserved independent afterany successor {successor}.')
    return successor


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    os.environ['JAX_PLATFORMS']='cpu'
    launched=time.monotonic();last_error=None
    successor=reserve_successor() if not args.once else None
    while True:
        try:
            with locked(ROOT/'.local/completion-service.lock'):
                complete=tick()
            last_error=None
            write_json(ROOT/'.local/completion-service-status.json',{'checked_at':time.time(),'all_remaining_complete':complete})
            print('Completion service checked; all remaining complete='+str(complete),flush=True)
            if complete or args.once:
                if complete and successor:
                    subprocess.run(['scancel',successor],check=True)
                break
        except Exception as error:
            message=f'{type(error).__name__}: {error}'
            if message!=last_error:log('Completion service retry: '+message)
            last_error=message
            if args.once:raise
        if time.monotonic()-launched>46*3600:
            log('Service yielding to reserved successor '+str(successor))
            break
        time.sleep(60)
