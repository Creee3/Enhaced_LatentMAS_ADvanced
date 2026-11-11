"""Six resident models, GPU1 first, one engineering gate without reloading."""
import os,subprocess,sys,time,traceback
from pathlib import Path
from common import HERE,ROOT,load,save,append,verify,identity,locked,choose
from resource_policy import admit,cpu_set
def own_pipe(pipe):
 found=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  x=identity(int(p.name))
  if not x or x['uid']!=os.getuid():continue
  try:env=dict(z.split(b'=',1) for z in (p/'environ').read_bytes().split(b'\0') if b'=' in z)
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
  if env.get(b'CUDA_MPS_PIPE_DIRECTORY',b'').decode()==str(pipe):found.append(x)
 return found

def foreign_mps(gpu,owned_pipes):
 conflicts=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   argv=(p/'cmdline').read_bytes().split(b'\0');name=Path(argv[0].decode()).name if argv and argv[0] else ''
   if name not in ['nvidia-cuda-mps-server','nvidia-cuda-mps-control']:continue
   env=dict(z.split(b'=',1) for z in (p/'environ').read_bytes().split(b'\0') if b'=' in z)
   if env.get(b'CUDA_MPS_PIPE_DIRECTORY',b'').decode() in owned_pipes:continue
   visible=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode().split(',')
   if visible==[''] or str(gpu['index']) in visible or gpu['uuid'] in visible:conflicts.append(int(p.name))
  except (PermissionError,FileNotFoundError,ProcessLookupError):continue
 return conflicts


def main(execute=False):
 source=verify();plan=load(HERE/'plan.json');assert plan['max_workers']==6 and plan['max_per_gpu']==4
 budget=load(ROOT/'cumulative_budget.json');assert load(HERE/'preflight.json')['passed'] and sum(budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens'])+load(HERE/'preflight.json')['upper_positions']<=115000000
 if not execute:print(__import__('json').dumps(dict(validated=True,launched=False)));return
 import fcntl
 guard=(HERE/'manager.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 scope=(ROOT/'role_intervention120_scope.lock').open('a');fcntl.flock(scope,fcntl.LOCK_EX|fcntl.LOCK_NB)
 folder=HERE/'management';folder.mkdir(exist_ok=True);assert not (folder/'identity.json').exists();save(folder/'identity.json',dict(identity=identity(os.getpid()),source=source,unix=time.time()))
 procs=[];services={};attempted={str(g['index']):0 for g in plan['gpus']};blocked=set();last_affinity=0
 def live():return [w for w in procs if w['process'].poll() is None]
 def register():save(folder/'workers.json',[{k:v for k,v in w.items() if k!='process'} for w in procs])
 def owner_live(w):
  x=identity(w['pid']);return x is not None and all(x[k]==w['identity'][k] for k in ['uid','start_ticks','argv','CUDA_VISIBLE_DEVICES'])
 def stop_loading():
  if not (HERE/'STOP').exists():return
  for w in live():
   if (Path(w['out'])/'model_ready.json').exists() or not owner_live(w):continue
   b=load(ROOT/'cumulative_budget.json');assert not any(v['pid']==w['pid'] for v in b['active_reservations'].values());save(Path(w['out'])/'intentional_loading_stop.json',dict(identity=identity(w['pid']),zero_RIDs=True));w['process'].terminate()
 def redistribute():
  nonlocal last_affinity
  if time.monotonic()-last_affinity<60:return
  last_affinity=time.monotonic();allowed=cpu_set(Path('/sys/fs/cgroup/cpuset.cpus.effective').read_text())
  if not allowed:return
  for w in live():
   if not (Path(w['out'])/'model_ready.json').exists() or not owner_live(w):continue
   target={allowed[w['slot']%len(allowed)]}
   for thread in Path('/proc',str(w['pid']),'task').iterdir():
    try:os.sched_setaffinity(int(thread.name),target)
    except (PermissionError,ProcessLookupError):pass
 try:
  while True:
   stop_loading();redistribute()
   for w in procs:
    code=w['process'].poll()
    if code is None or w.get('reaped'):continue
    w.update(reaped=True,exit_code=code);register()
    unfinished=any(t['state']=='running' and t.get('owner',{}).get('pid')==w['pid'] for t in load(HERE/'queue.json')['tasks'])
    if (code or unfinished) and not (HERE/'STOP').exists():
     blocked.add(w['gpu_uuid']);save(Path(w['out'])/'manager_observed_exit.json',dict(returncode=code,unix=time.time()))
     with locked():
      q=load(HERE/'queue.json')
      for t in q['tasks']:
       if t['state']=='running' and t.get('owner',{}).get('pid')==w['pid']:t.update(state='held',error='worker exited; preserve scene')
      save(HERE/'queue.json',q)
   q=load(HERE/'queue.json');eng=[q['tasks'][i] for i in q['engineering_indices']];released=(HERE/'engineering_release.json').exists()
   if not released and any(t['state']=='held' for t in eng):(HERE/'MANAGEMENT_HOLD').touch()
   all_waiting=bool(live()) and all((Path(w['out'])/'current.json').exists() and load(Path(w['out'])/'current.json')['phase']=='waiting_engineering' for w in live())
   if all(t['state']=='complete' for t in eng) and all_waiting and not (HERE/'engineering_gate_ready.json').exists():
    save(HERE/'engineering_gate_ready.json',dict(keys=len(eng)*len(plan['methods']),waiting_for_local_archive=True,models_retained=True,unix=time.time()))
   if not live() and procs and (all(t['state'] in ['complete','held'] for t in q['tasks']) or (HERE/'STOP').exists() or (HERE/'MANAGEMENT_HOLD').exists()):break
   phase='formal' if released else 'engineering';pending=choose(q,phase) is not None
   if (HERE/'STOP').exists() or (HERE/'MANAGEMENT_HOLD').exists():
    if not live():break
    time.sleep(2);continue
   launched=False
   for gpu in plan['gpus']:
    card=str(gpu['index']);uuid=gpu['uuid'];oncard=[w for w in live() if w['gpu_uuid']==uuid]
    if attempted[card]>=gpu['max_workers'] or len(live())>=6 or not pending:continue
    fresh=subprocess.run(['nvidia-smi','-i',uuid,'--query-gpu=utilization.gpu,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip();util,free,total=map(int,fresh.split(','));conflicts=foreign_mps(gpu,{str(x['pipe']) for x in services.values()})
    decision=admit(live_total=len(live()),live_gpu=len(oncard),all_ready=all((Path(w['out'])/'model_ready.json').exists() for w in oncard),blocked=uuid in blocked or bool(conflicts),util=util,free=free,total=total,pending=pending)
    append(folder/'admission.jsonl',dict(gpu=gpu,util=util,free=free,total=total,admitted=decision,foreign_MPS_conflicts=conflicts,live_total=len(live()),live_gpu=len(oncard),unix=time.time()))
    if not decision:continue
    if uuid not in services:
     pipe=Path('/tmp')/f'geom{os.getpid()}_{card}';pipe.mkdir(mode=0o700);log=folder/f'mps_{card}';log.mkdir(mode=0o700)
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(log),LATENTMAS_SHARED_LEDGER=str(ROOT/'cumulative_budget.json'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
     start=subprocess.run(['/usr/bin/nvidia-cuda-mps-control','-d'],env=env,capture_output=True,text=True,timeout=30);assert start.returncode==0,start.stderr;services[uuid]=dict(env=env,pipe=pipe,card=card);save(folder/f'mps_{card}_start.json',dict(gpu_uuid=uuid,pipe=str(pipe),owned=own_pipe(pipe),unix=time.time()))
    env=services[uuid]['env'];slot=len(procs);out=HERE/'attempts'/f'resident_w{slot}_{time.time_ns()}';attempted[card]+=1
    with (folder/f'worker{slot}.log').open('ab') as log:p=subprocess.Popen([sys.executable,str(HERE/'worker.py'),'--slot',str(slot),'--out',str(out)],cwd=HERE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    time.sleep(.1);owner=identity(p.pid);assert owner is not None
    w=dict(process=p,pid=p.pid,identity=owner,gpu_uuid=uuid,gpu_index=gpu['index'],slot=slot,out=str(out));procs.append(w);register()
    while not (out/'model_ready.json').exists() and p.poll() is None:stop_loading();time.sleep(1)
    launched=True;break
   if not launched:time.sleep(5)
  q=load(HERE/'queue.json');save(folder/'finished.json',dict(all_complete=all(t['state']=='complete' for t in q['tasks']),counts={s:sum(t['state']==s for t in q['tasks']) for s in ['pending','running','held','complete']},keys=sum(len(t['committed_methods']) for t in q['tasks']),unix=time.time()))
 except BaseException as e:
  save(folder/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()));(HERE/'MANAGEMENT_HOLD').touch();raise
 finally:
  while live():stop_loading();time.sleep(2)
  for uuid,service in services.items():
   before=own_pipe(service['pipe']);stop=subprocess.run(['/usr/bin/nvidia-cuda-mps-control'],input='quit\n',env=service['env'],capture_output=True,text=True,timeout=30);time.sleep(1);after=own_pipe(service['pipe']);save(folder/f"mps_{service['card']}_stop.json",dict(returncode=stop.returncode,owned_before=before,owned_after=after,unix=time.time()))
if __name__=='__main__':main('--execute' in sys.argv)
