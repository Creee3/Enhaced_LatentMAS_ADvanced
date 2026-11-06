"""Explicitly authorized GPU1 pair, max2 attempts, no automatic retries."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
from cloud_common import RUN,ROOT,load,save,verify,identity,gpu_rows,qlock
from manager import owned,stable
def main():
 guard=(RUN/'manager.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 source=verify();workers=[];attempts=0
 save(RUN/'manager.json',dict(identity=identity(os.getpid()),source=source,entry='pair_manager.py'))
 while True:
  queue=load(RUN/'queue.json');live=[w for w in workers if owned(w)];pending=sum(t['state']=='pending' for t in queue['tasks'])
  if (RUN/'STOP').exists() and not live:break
  if not pending and not live:break
  if pending and attempts<2 and len(live)<2 and not (RUN/'STOP').exists():
   first_ready=not workers or load(Path(workers[0]['out'])/'model_ready.json')
   rows=gpu_rows([1]);fresh=rows[0] if rows else None
   # For second same-card worker, user explicitly permits own-model utilization.
   eligible=fresh and fresh['free_mib']/fresh['total_mib']>=.70 and (live or fresh['utilization']<=30)
   if first_ready and eligible:
    attempts+=1;out=RUN/'attempts'/('gpu1_slot'+str(attempts)+'_'+str(time.time_ns()));log=out.with_suffix('.log')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',LATENTMAS_SHARED_LEDGER=str(ROOT/'cumulative_budget.json'),PYTHONUNBUFFERED='1',PYTHONHASHSEED='42',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
    argv=[sys.executable,str(RUN/'cloud_worker.py'),'--out',str(out)]
    with log.open('ab') as stream:p=subprocess.Popen(argv,cwd=RUN,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    ident=stable(p.pid,1,argv);workers.append(dict(pid=p.pid,gpu=1,out=str(out),log=str(log),admission=fresh,initial_identity=ident,slot=attempts))
    save(RUN/'workers.json',workers)
  with qlock():
   queue=load(RUN/'queue.json')
   for task in queue['tasks']:
    if task['state']=='running' and not any(w['pid']==task['owner_pid'] and owned(w) for w in workers):task.update(state='failed_held',failure='Owner exited, no automatic retry')
   save(RUN/'queue.json',queue)
  save(RUN/'manager_status.json',dict(unix=time.time(),live_pids=[w['pid'] for w in workers if owned(w)],attempts=attempts,tasks={s:sum(t['state']==s for t in queue['tasks']) for s in ['pending','running','complete','failed_held']}))
  if not any(owned(w) for w in workers) and attempts==2 and pending:break
  time.sleep(10)
 save(RUN/'manager_exit.json',dict(completed=all(t['state']=='complete' for t in load(RUN/'queue.json')['tasks']),user_stop=(RUN/'STOP').exists()))
if __name__=='__main__':main()
