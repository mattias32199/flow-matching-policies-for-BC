"""Durable Slurm service: recover timed-out segments and publish the completed five-run report."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from collection_cache import locked, write_json
from individual_jobs import ROOT, resolve_run, record, log
from prepare_migration import prepare


def job_fields(job_id):
    result = subprocess.run(['scontrol','show','job',job_id,'-o'],capture_output=True,text=True)
    if result.returncode == 0:
        return dict(token.split('=',1) for token in result.stdout.split() if '=' in token)
    result = subprocess.run(['sacct','-n','-X','-j',job_id,'-o','State','-P'],capture_output=True,text=True,check=True)
    state=result.stdout.strip().splitlines()
    return {'JobState':state[0].split('|')[0] if state else 'UNKNOWN'}


def recover(name, item):
    fields=job_fields(item['job_id']);state=fields.get('JobState','UNKNOWN')
    # An intentional user cancellation is respected; never automatically undo it.
    restart = state in ('TIMEOUT','PREEMPTED','NODE_FAIL') or (
        state=='FAILED' and item.get('state')=='checkpoint_continuation_required')
    if not restart:
        return False
    directory=ROOT/'runs'/name
    if not (directory/'checkpoints').exists() or not list((directory/'checkpoints').glob('params_*.pkl')):
        log(f'Automatic recovery blocked for {name}: no saved checkpoint; preserving artifacts.')
        return False
    bundle,manifest=prepare(name)
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
    if '--resume-bundle' in cmd:cmd[cmd.index('--resume-bundle')+1]=str(bundle.relative_to(ROOT))
    else:cmd+=['--resume-bundle',str(bundle.relative_to(ROOT))]
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
    subprocess.run(['scontrol','update','JobId='+item['job_id'],'JobName='+new],check=True)
    subprocess.run(['scontrol','requeue',item['job_id']],check=True)
    record(new,state='submitted')
    # Requeue has a short automatic delay; this eligibility update is best-effort.
    subprocess.run(['scontrol','update','JobId='+item['job_id'],'StartTime=now'],capture_output=True,text=True)
    log(f'Automatic timeout/preemption continuation {name}→{new} at{manifest["step"]}; same job{item["job_id"]}.')
    return True


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
        if item.get('state')=='requeue_pending':
            subprocess.run(['scontrol','requeue',item['job_id']],check=True)
            record(name,state='submitted')
        else:
            recover(name,item)
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
    # Keep recovering the other scientific jobs after the main report is published.
    selected={resolve_run(job['run']) for job in json.loads((ROOT/'.local/individual-launch-plan.json').read_text())}
    return all((ROOT/'runs'/name/'DONE').exists() for name in selected)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
    os.environ['JAX_PLATFORMS']='cpu'
    launched=time.monotonic();last_error=None
    while True:
        try:
            with locked(ROOT/'.local/completion-service.lock'):
                complete=tick()
            last_error=None
            write_json(ROOT/'.local/completion-service-status.json',{'checked_at':time.time(),'all_remaining_complete':complete})
            print('Completion service checked; all remaining complete='+str(complete),flush=True)
            if complete or args.once:break
        except Exception as error:
            message=f'{type(error).__name__}: {error}'
            if message!=last_error:log('Completion service retry: '+message)
            last_error=message
            if args.once:raise
        if time.monotonic()-launched>46*3600:
            info=json.loads((ROOT/'.local/completion-service-job.json').read_text())
            result=subprocess.run(['sbatch','--parsable','--dependency=afterany:'+os.environ['SLURM_JOB_ID'],info['launcher']],
                                  capture_output=True,text=True,check=True)
            info['job_id']=result.stdout.strip().split(';')[0];write_json(ROOT/'.local/completion-service-job.json',info)
            log('Handed durable completion service to successor job'+info['job_id'])
            break
        time.sleep(60)
