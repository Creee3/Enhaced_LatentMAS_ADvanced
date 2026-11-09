"""Explicit manager for this run only; private per-GPU MPS, at most six workers."""
import argparse,os,subprocess,sys,time,traceback
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

def main(phase,execute=False):
 source=verify();plan=load(HERE/'plan.json');assert plan['max_workers']==6 and plan['methods'] in [['H','L'],['Full','receiver','GGG','RRR','TTT']]
 q0=load(HERE/'queue.json');eligible0=set(q0['engineering_indices']) if phase=='engineering' else {t['index'] for t in q0['tasks']}
 required=sum(p['upper_positions'] for p in load(HERE/'preflight.json')['proofs'] if p['index'] in eligible0 and q0['tasks'][p['index']]['state']!='complete')
 budget=load(ROOT/'cumulative_budget.json');assert sum(budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens'])+required+plan.get('future_upper_positions',0)<=115000000
 if phase=='formal':
  audit=load(HERE/'engineering_audit_verified.json');archive=load(HERE/'engineering_archive_verified.json')
  assert audit['passed'] and archive['passed'] and audit['keys']==len(q0['engineering_indices'])*len(plan['methods']) and audit['run_lock_sha256']==source['run_lock_sha256']
 if not execute:print(__import__('json').dumps(dict(validated=True,phase=phase,max_workers=6,remaining_upper_positions=required,launched=False)));return
 import fcntl
 scope_guard=(ROOT/'HL705_extension129_scope.lock').open('a');fcntl.flock(scope_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 guard=(HERE/'manager.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 folder=HERE/'management'/phase;folder.mkdir(parents=True,exist_ok=False);save(folder/'identity.json',dict(identity=identity(os.getpid()),source=source,phase=phase,unix=time.time()))
 procs=[];services={};attempted={str(g['index']):0 for g in plan['gpus']};blocked=set();last_affinity=0
 def live():return [x for x in procs if x['process'].poll() is None]
 def pending():return choose(load(HERE/'queue.json'),phase) is not None
 def register():save(folder/'workers.json',[{k:v for k,v in x.items() if k!='process'} for x in procs])
 def verify_owner(w):
  x=identity(w['pid']);return x is not None and x['uid']==os.getuid() and x['start_ticks']==w['identity']['start_ticks'] and x['argv'][1]==str(HERE/'worker.py') and x['CUDA_VISIBLE_DEVICES']==w['gpu_uuid']
 def stop_loading_if_requested():
  if not (HERE/'STOP').exists():return
  b=load(ROOT/'cumulative_budget.json')
  for w in live():
   if not (Path(w['out'])/'model_ready.json').exists() and not any(e['pid']==w['pid'] for e in b['active_reservations'].values()) and verify_owner(w):
    save(Path(w['out'])/'intentional_loading_stop.json',dict(identity=identity(w['pid']),reserved=0,unix=time.time()));w['process'].terminate()
 def redistribute():
  nonlocal last_affinity
  if time.monotonic()-last_affinity<60:return
  allowed=cpu_set(Path('/sys/fs/cgroup/cpuset.cpus.effective').read_text());last_affinity=time.monotonic()
  if not allowed:return
  for w in live():
   if not (Path(w['out'])/'model_ready.json').exists() or not verify_owner(w):continue
   target={allowed[w['slot']%len(allowed)]};changed=[]
   for t in (Path('/proc')/str(w['pid'])/'task').iterdir():
    try:
     tid=int(t.name)
     if os.sched_getaffinity(tid)!=target:os.sched_setaffinity(tid,target);changed.append(tid)
    except (FileNotFoundError,ProcessLookupError,OSError):pass
   if changed:append(folder/'cpu_affinity.jsonl',dict(pid=w['pid'],start_ticks=w['identity']['start_ticks'],allowed=allowed,assigned=sorted(target),thread_count=len(changed),unix=time.time()))
 try:
  while True:
   stop_loading_if_requested();redistribute()
   for w in procs:
    code=w['process'].poll()
    if code is None or w.get('reaped'):continue
    w.update(reaped=True,exit_code=code)
    if code!=0 and not (HERE/'STOP').exists():
     blocked.add(w['gpu_uuid']);save(Path(w['out'])/'manager_observed_exit.json',dict(returncode=code,identity=identity(w['pid']),unix=time.time()))
     with locked():
      q=load(HERE/'queue.json')
      for task in q['tasks']:
       if task['state']=='running' and task.get('owner',{}).get('pid')==w['pid']:task.update(state='held',error='owned worker exited; preserve artifacts, no automatic recovery')
      save(HERE/'queue.json',q)
     register()
   if not pending() and not live():break
   if blocked and not live():break
   if (HERE/'STOP').exists():
    if not live():break
    time.sleep(1);continue
   launched=False
   for gpu in plan['gpus']:
    uuid=gpu['uuid'];card=str(gpu['index']);oncard=[w for w in live() if w['gpu_uuid']==uuid]
    if attempted[card]>=4 or not pending():continue
    fresh=subprocess.run(['nvidia-smi','-i',uuid,'--query-gpu=utilization.gpu,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip();util,free,total=map(int,fresh.split(','))
    conflicts=foreign_mps(gpu,{str(x['pipe']) for x in services.values()})
    decision=admit(live_total=len(live()),live_gpu=len(oncard),all_ready=all((Path(w['out'])/'model_ready.json').exists() for w in oncard),blocked=bool(blocked) or bool(conflicts) or len(oncard)>=3,util=util,free=free,total=total,pending=pending());append(folder/'admission.jsonl',dict(gpu=gpu,util=util,free=free,total=total,admitted=decision,live_total=len(live()),live_gpu=len(oncard),foreign_MPS_conflicts=conflicts,unix=time.time()))
    if not decision:continue
    if uuid not in services:
     pipe=Path('/tmp')/f'rw{os.getpid()}_{card}';pipe.mkdir(mode=0o700);log=folder/f'mps_{card}';log.mkdir(mode=0o700)
     env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(log),LATENTMAS_SHARED_LEDGER=str(ROOT/'cumulative_budget.json'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
     start=subprocess.run(['/usr/bin/nvidia-cuda-mps-control','-d'],env=env,capture_output=True,text=True,timeout=30);assert start.returncode==0,start.stderr
     services[uuid]=dict(env=env,pipe=pipe,card=card);save(folder/f'mps_{card}_start.json',dict(gpu_uuid=uuid,pipe=str(pipe),mode=oct(pipe.stat().st_mode&0o777),owned=own_pipe(pipe),unix=time.time()))
    env=services[uuid]['env'];slot=len(procs);out=HERE/'attempts'/f'{phase}_w{slot}_{time.time_ns()}';attempted[card]+=1
    with (folder/f'worker{slot}.log').open('ab') as log:p=subprocess.Popen([sys.executable,str(HERE/'worker.py'),'--phase',phase,'--slot',str(slot),'--out',str(out)],cwd=HERE,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    time.sleep(.1);owner=identity(p.pid);assert owner is not None
    w=dict(process=p,pid=p.pid,identity=owner,gpu_uuid=uuid,gpu_index=gpu['index'],slot=slot,out=str(out));procs.append(w);register();launched=True
    begin=time.monotonic()
    while not (out/'model_ready.json').exists() and p.poll() is None:
     stop_loading_if_requested()
     # Loading has no scientific deadline; STOP and hourly inspection remain active.
     time.sleep(1)
    save(folder/f'mps_{card}_scene.json',dict(owned=own_pipe(services[uuid]['pipe']),server_list=subprocess.run(['/usr/bin/nvidia-cuda-mps-control'],input='get_server_list\n',env=env,capture_output=True,text=True,timeout=30).stdout,unix=time.time()))
    break
   if not launched:time.sleep(10)
  q=load(HERE/'queue.json');eligible=[t for t in q['tasks'] if phase=='formal' or t['index'] in q['engineering_indices']]
  complete=all(t['state']=='complete' for t in eligible)
  save(folder/'finished.json',dict(all_eligible_complete=complete,phase=phase,counts={s:sum(t['state']==s for t in eligible) for s in ['pending','running','held','complete']},keys=sum(len(t['committed_methods']) for t in eligible),user_STOP=(HERE/'STOP').exists(),unix=time.time()))
 except BaseException as e:save(folder/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),identity=identity(os.getpid()),unix=time.time()));raise
 finally:
  while live():stop_loading_if_requested();time.sleep(1)
  for uuid,service in services.items():
   before=own_pipe(service['pipe']);stop=subprocess.run(['/usr/bin/nvidia-cuda-mps-control'],input='quit\n',env=service['env'],capture_output=True,text=True,timeout=30);time.sleep(2);after=own_pipe(service['pipe']);save(folder/f"mps_{service['card']}_stop.json",dict(returncode=stop.returncode,owned_before=before,owned_after=after,unix=time.time()))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['engineering','formal'],required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();main(a.phase,a.execute)
