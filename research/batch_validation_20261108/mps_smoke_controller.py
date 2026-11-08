"""Fresh private MPS, exact prefix rebuild gate, then two native long requests."""
import json,os,subprocess,sys,time,signal,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];OLD=ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z';sys.path.insert(0,str(OLD))
from cloud_common import identity,load,save,verify
def main():
 import torch
 a=load(OUT/'launch.json');parent=Path(a['parent']);rebuilt=Path(a['rebuilt']);children=[];daemon=False
 pipe=Path('/tmp')/('lsmk_'+OUT.name[-12:]);pipe.mkdir(mode=0o700);(OUT/'mps_logs').mkdir(mode=0o700)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=a['gpu_uuid'],CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(OUT/'mps_logs'),LATENTMAS_SHARED_LEDGER=str(ROOT/'cumulative_budget.json'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 save(OUT/'namespace.json',dict(pipe=str(pipe),logs=str(OUT/'mps_logs'),uid=os.getuid()))
 control=lambda text:subprocess.run(['/usr/bin/nvidia-cuda-mps-control'],input=text+'\n',text=True,env=env,capture_output=True,timeout=30)
 def scene():
  found=[]
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   x=identity(int(p.name))
   if not x:continue
   try:values=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\0') if b'=' in v)
   except (PermissionError,FileNotFoundError,ProcessLookupError):continue
   if values.get(b'CUDA_MPS_PIPE_DIRECTORY',b'').decode()==str(pipe):found.append(x)
  return found
 def launch(argv,name):
  with (OUT/(name+'.log')).open('ab') as log:p=subprocess.Popen([sys.executable]+argv,env=env,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
  children.append(p);return p
 def wait_for(test,timeout=1500):
  start=time.monotonic()
  while not test():
   failures=list(OUT.glob('failure_*.json'))+list(rebuilt.glob('failure_*.json'))
   if failures:raise RuntimeError('worker failed: '+str(failures[0]))
   if time.monotonic()-start>timeout:raise RuntimeError('bounded smoke timeout')
   time.sleep(1)
 def admission(first):
  start=time.monotonic()
  while True:
   raw=subprocess.run(['nvidia-smi','-i','0','--query-gpu=utilization.gpu,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout;u,f,t=map(int,raw.strip().split(','))
   if ((u<=30 and f/t>=.70) if first else f>=21*1024):return
   if time.monotonic()-start>120:raise RuntimeError('fresh GPU admission unavailable')
   time.sleep(1)
 try:
  started=subprocess.run(['/usr/bin/nvidia-cuda-mps-control','-d'],env=env,capture_output=True,text=True,timeout=30);assert started.returncode==0;daemon=True
  p=launch([str(rebuilt/'worker.py'),'prepare'],'prefix');wait_for(lambda:p.poll() is not None);assert p.returncode==0 and load(rebuilt/'prepared.json')['passed']
  matches=[]
  for i in range(4):
   old=torch.load(parent/f'packet{i}.pt',weights_only=True);new=torch.load(rebuilt/f'packet{i}.pt',weights_only=True)
   same=old['absolute']==new['absolute'] and all(torch.equal(old[n],new[n]) for n in ['ids','rng','position']) and all(torch.equal(t,v) for xs,ys in zip(old['kv'],new['kv']) for t,v in zip(xs,ys));matches.append(same)
  save(OUT/'prefix_gate.json',dict(passed=all(matches),matches=matches,source=verify()))
  assert all(matches),'Prefix changed under MPS; do not generate long answers'
  tr=[json.loads(b) for b in (rebuilt/'prefix/trace.jsonl').read_bytes().split(b'\n') if b];prefixcost=sum(x['actual_positions'] for x in tr if x['event']=='request_settled');bound=prefixcost+sum(torch.load(rebuilt/f'packet{i}.pt',weights_only=True)['ids'].shape[-1]+8192 for i in range(2));assert bound<=50000
  save(OUT/'preflight.json',dict(prefix_cost=prefixcost,total_upper_bound=bound,cap=8192))
  models=[]
  for i in range(2):
   admission(not i);p=launch([str(OUT/'hf_smoke.py'),'--index',str(i)],'hf'+str(i));models.append(p);wait_for(lambda i=i:(OUT/f'hf{i}/ready.json').exists());save(OUT/f'hf_identity{i}.json',identity(p.pid))
  cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=2
  for i,p in enumerate(models):
   assert identity(p.pid)==load(OUT/f'hf_identity{i}.json')
   for task in Path('/proc',str(p.pid),'task').iterdir():os.sched_setaffinity(int(task.name),{cpus[i]})
  save(OUT/'mps_scene.json',dict(owned_processes=scene(),server_list=control('get_server_list').stdout));(OUT/'go').touch()
  wait_for(lambda:all(p.poll() is not None for p in models));assert all(p.returncode==0 for p in models)
  result=dict(passed=True,engineering_only=True,prefix_all_four_exact=True,hf=[{k:v for k,v in load(OUT/f'hf{i}/result.json').items() if k!='tokens'} for i in range(2)],source=verify());save(OUT/'result.json',result)
 finally:
  for p in children:
   if p.poll() is None:
    budget=load(ROOT/'cumulative_budget.json');assert not any(v['pid']==p.pid for v in budget['active_reservations'].values()),'Do not interrupt healthy in-flight reservation'
    actual=identity(p.pid);assert actual and actual['uid']==os.getuid() and actual['argv'][1].startswith(str(OUT));os.kill(p.pid,signal.SIGTERM);p.wait(timeout=30)
  if daemon:
   before=scene();stop=control('quit');time.sleep(2);save(OUT/'mps_stop.json',dict(returncode=stop.returncode,owned_before=before,owned_after=scene()))
if __name__=='__main__':
 try:main()
 except BaseException as e:save(OUT/'failure_controller.json',dict(error=repr(e),unix=time.time()));raise
