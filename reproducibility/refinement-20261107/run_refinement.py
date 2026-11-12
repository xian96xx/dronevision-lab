"""Bounded sequential GPU experiments; retain the original completed runs unchanged."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'reports/refinement-queue.json'

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2),encoding='utf-8');os.replace(temp,path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    lock=ROOT/'work/refinement.lock'
    if lock.exists():
        pid=int(lock.read_text())
        if psutil.pid_exists(pid):raise RuntimeError(f'Refinement queue already has live PID {pid}')
        raise RuntimeError('Inspect stale refinement lock before restarting')
    jobs=[]
    for seed in [43,44]:
        for variant in ['clean','boundary','wrong']:
            name=f'seg-noise-{variant}-seed{seed}'
            jobs.append({'name':name,'status':'pending','command':[sys.executable,'-u','scripts/segmentation.py','train',
                '--name',name,'--variant',variant,'--subset','40','--epochs','10','--seed',str(seed)]})
    name='seg-full-weighted-20'
    jobs.append({'name':name,'status':'pending','command':[sys.executable,'-u','scripts/segmentation.py','train',
        '--name',name,'--epochs','20','--seed','42','--class-weighting','inverse-sqrt']})
    if not args.resume and any((ROOT/'runs'/j['name']).exists() for j in jobs):
        raise RuntimeError('A refinement run already exists; inspect it before resuming')
    with lock.open('x') as f:f.write(str(os.getpid()))
    state={'status':'running','started_utc':now(),'pid':os.getpid(),'jobs':jobs,
           'scope':'6 additional segmentation-noise runs plus 1 full class-weighted run; public UAVid only',
           'original_runs_unchanged':True,'test_used_for_tuning':False}
    snapshot=ROOT/'work/refinement-20261107-source';snapshot.mkdir(exist_ok=True)
    for filename in ['segmentation.py','run_refinement.py']:
        source=ROOT/'scripts'/filename;shutil.copy2(source,snapshot/filename)
    state['source_sha256']={p.name:sha(p) for p in snapshot.glob('*.py')}
    def save():write(STATE,state)
    save()
    try:
        for job in jobs:
            record_path=ROOT/'reports'/f'{job["name"]}-training.json'
            if args.resume and record_path.exists():
                record=json.loads(record_path.read_text())
                if record.get('status')=='completed' and sha(ROOT/'runs'/job['name']/'best/model.safetensors')==record['best_weights_sha256']:
                    job.update(status='completed',reused_verified=True);save();continue
                if (ROOT/'runs'/job['name']/'last.pt').exists():job['command'].append('--resume')
            job.update(status='running',started_utc=now());save()
            log=ROOT/'logs'/f'refinement-{job["name"]}.log'
            with log.open('a' if args.resume else 'w',encoding='utf-8') as f:
                child=subprocess.Popen(job['command'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                    env={**os.environ,'PYTHONUTF8':'1'})
                job['pid']=child.pid;job['log']=str(log.relative_to(ROOT));save()
                code=child.wait()
            job.update(status='completed' if code==0 else 'failed',exit_code=code,ended_utc=now())
            if code==0:
                record=json.loads(record_path.read_text())
                if not (record['status']=='completed' and record['encoder_max_abs_delta']>0 and
                        sha(ROOT/'runs'/job['name']/'best/model.safetensors')==record['best_weights_sha256']):
                    job.update(status='failed',error='Completion evidence verification failed')
            save()
            summarize=ROOT/'scripts/summarize_refinement.py'
            if summarize.exists():
                with (ROOT/'logs/refinement-summary.log').open('a',encoding='utf-8') as f:
                    subprocess.run([sys.executable,str(summarize)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        state.update(status='completed' if all(j['status']=='completed' for j in jobs) else 'completed_with_failures',ended_utc=now());save()
    except Exception:
        import traceback
        state.update(status='failed',error=traceback.format_exc(),ended_utc=now());save();raise
    finally:
        lock.unlink(missing_ok=True)
    print(json.dumps(state,indent=2),flush=True)

if __name__=='__main__':main()
