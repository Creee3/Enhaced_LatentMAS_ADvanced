"""Versioned management-only GPU0/1 expansion; scientific worker unchanged."""
import fcntl,os,subprocess,sys,time
from pathlib import Path
RUN=Path(__file__).resolve().parents[1];sys.path.insert(0,str(RUN))
from cloud_common import ROOT,load,save,verify,identity,gpu_rows,qlock,sha
from manager import owned,stable
def eligible(fresh,live_on_card):
 if not fresh:return False
 return fresh['free_mib']>=21*1024 if live_on_card else fresh['utilization']<=30 and fresh['free_mib']/fresh['total_mib']>=.70
def main():
 guard=(RUN/'manager.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 source=verify();workers=load(RUN/'workers.json',[])
 for n,e in load(RUN/'run_lock.json')['snapshots'].items():assert sha(RUN/n)==e['sha256'],n
 assert all(owned(w) for w in workers),'Takeover requires all original workers alive'
 save(RUN/'manager.json',dict(identity=identity(os.getpid()),source=source,entry='management/pool_manager.py',started=time.time()))
 try:
  while True:
   policy=load(RUN/'management/gpu01_resource_policy.json');assert policy['max_workers'] in [6,8] and policy['per_gpu_max']=={'0':policy['max_workers']-4,'1':4}
   live=[w for w in workers if owned(w)]
   pending=sum(t['state']=='pending' for t in load(RUN/'queue.json')['tasks'])
   if ((RUN/'STOP').exists() or not pending) and not live:break
   if pending and len(live)<policy['max_workers'] and not (RUN/'STOP').exists():
    for gpu in [0,1]:
     attempts=sum(w['gpu']==gpu for w in workers);card_live=[w for w in live if w['gpu']==gpu]
     failed=any((Path(w['out'])/'failure.json').exists() for w in workers if w['gpu']==gpu)
     if attempts>=policy['per_gpu_max'][str(gpu)] or failed:continue
     if not all(load(Path(w['out'])/'model_ready.json') for w in card_live):continue
     rows=gpu_rows([gpu]);fresh=rows[0] if rows else None
     if not eligible(fresh,card_live):continue
     # Re-read pending work before each spawn; existing workers use same qlock.
     if not any(t['state']=='pending' for t in load(RUN/'queue.json')['tasks']):break
     out=RUN/'attempts'/('gpu'+str(gpu)+'_slot'+str(attempts+1)+'_'+str(time.time_ns()));log=out.with_suffix('.log')
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),LATENTMAS_SHARED_LEDGER=str(ROOT/'cumulative_budget.json'),PYTHONUNBUFFERED='1',PYTHONHASHSEED='42',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
     argv=[sys.executable,str(RUN/'cloud_worker.py'),'--out',str(out)]
     with log.open('ab') as f:p=subprocess.Popen(argv,cwd=RUN,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
     ident=stable(p.pid,gpu,argv);workers.append(dict(pid=p.pid,gpu=gpu,slot=attempts+1,out=str(out),log=str(log),admission=fresh,initial_identity=ident,manager_version='gpu01_v2'))
     save(RUN/'workers.json',workers);break
   with qlock():
    queue=load(RUN/'queue.json')
    for task in queue['tasks']:
     if task['state']=='running' and not any(w['pid']==task['owner_pid'] and owned(w) for w in workers):task.update(state='failed_held',failure='owner exited; preserve scene, no automatic retry')
    save(RUN/'queue.json',queue)
   live=[w for w in workers if owned(w)]
   save(RUN/'manager_status.json',dict(unix=time.time(),identity=identity(os.getpid()),entry='gpu01_v2',live_pids=[w['pid'] for w in live],attempts=len(workers),tasks={s:sum(t['state']==s for t in queue['tasks']) for s in ['pending','running','complete','failed_held']}))
   if not live and pending and all(sum(w['gpu']==gpu for w in workers)>=policy['per_gpu_max'][str(gpu)] or any((Path(w['out'])/'failure.json').exists() for w in workers if w['gpu']==gpu) for gpu in [0,1]):break
   time.sleep(10)
  save(RUN/'manager_exit.json',dict(completed=all(t['state']=='complete' for t in load(RUN/'queue.json')['tasks']),user_stop=(RUN/'STOP').exists()))
 except BaseException as exc:
  save(RUN/'manager_failure.json',dict(error=repr(exc),unix=time.time()));raise
if __name__=='__main__':main()
