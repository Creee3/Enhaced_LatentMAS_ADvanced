"""Bounded, private GPU0 MPS test; all source decode code remains B1."""
import json,os,sys,time,subprocess,signal,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];OLD=ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z';sys.path.insert(0,str(OLD))
from cloud_common import identity,load,save
def main():
 parent=Path(load(OUT/'launch.json')['parent']);children=[];daemon=False
 pipe=Path('/tmp')/('lmps_'+OUT.name[-12:]);pipe.mkdir(mode=0o700);(OUT/'mps_logs').mkdir(mode=0o700)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=load(OUT/'launch.json')['gpu_uuid'],CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(OUT/'mps_logs'),LATENTMAS_SHARED_LEDGER=str(ROOT/'cumulative_budget.json'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
 save(OUT/'mps_namespace.json',dict(pipe=str(pipe),logs=str(OUT/'mps_logs'),uid=os.getuid()))
 control=lambda command:subprocess.run(['/usr/bin/nvidia-cuda-mps-control'],input=command+'\n',text=True,env=env,capture_output=True,timeout=30)
 def scene():
  report=[]
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   x=identity(int(p.name))
   if not x:continue
   try:values=dict(s.split(b'=',1) for s in (p/'environ').read_bytes().split(b'\0') if b'=' in s)
   except (FileNotFoundError,PermissionError,ProcessLookupError):continue
   if values.get(b'CUDA_MPS_PIPE_DIRECTORY',b'').decode()==str(pipe):report.append(x)
  return report
 def wait_for(test):
  start=time.monotonic()
  while not test():
   failures=list(OUT.glob('failure_*.json'))
   if failures:raise RuntimeError('Worker failure '+failures[0].name)
   if time.monotonic()-start>360:raise RuntimeError('bounded stage timeout')
   time.sleep(1)
 try:
  p=subprocess.run(['/usr/bin/nvidia-cuda-mps-control','-d'],env=env,capture_output=True,text=True,timeout=30);save(OUT/'mps_start.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));assert p.returncode==0;daemon=True
  for i in range(4):
   fresh=subprocess.run(['nvidia-smi','-i','0','--query-gpu=utilization.gpu,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout;u,f,t=map(int,fresh.strip().split(','));assert (u<=30 and f/t>=.70) if not i else f>=21*1024
   save(OUT/f'admission{i}.json',dict(util=u,free=f,total=t,unix=time.time()))
   with (OUT/f'process{i}.log').open('ab') as log:p=subprocess.Popen([sys.executable,str(OUT/'worker.py'),'process','--index',str(i)],cwd=OUT,env=env,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
   children.append(p);wait_for(lambda i=i:(OUT/f'process{i}/ready.json').exists());save(OUT/f'identity{i}.json',identity(p.pid))
  save(OUT/'mps_scene.json',dict(owned_processes=scene(),server_list=control('get_server_list').stdout))
  cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=4
  for i,p in enumerate(children):
   assert identity(p.pid)==load(OUT/f'identity{i}.json')
   for task in Path('/proc',str(p.pid),'task').iterdir():os.sched_setaffinity(int(task.name),{cpus[i]})
  for rep in range(2):
   wait_for(lambda:all((OUT/f'process{i}/waiting{rep}').exists() for i in range(4)));(OUT/f'go{rep}').touch();wait_for(lambda:all((OUT/f'process{i}/result{rep}.json').exists() for i in range(4)))
  wait_for(lambda:all(p.poll() is not None for p in children));assert all(p.returncode==0 for p in children)
  result=dict(passed=True,engineering_only=True,processes=[load(OUT/f'process{i}/done.json') for i in range(4)],source_worker_sha=hashlib.sha256((OUT/'worker.py').read_bytes()).hexdigest())
  receipts=[]
  for p in OUT.glob('process*/trace.jsonl'):receipts += [x for b in p.read_bytes().split(b'\n') if b and (x:=json.loads(b))['event']=='request_settled']
  result['charged_positions']=sum(x['actual_positions'] for x in receipts);assert result['charged_positions']<=20000
  save(OUT/'result.json',result)
 finally:
  for p in children:
   if p.poll() is None:
    actual=identity(p.pid);budget=load(ROOT/'cumulative_budget.json');assert not any(x['pid']==p.pid for x in budget['active_reservations'].values()),'Do not interrupt in-flight request'
    assert actual and actual['uid']==os.getuid() and actual['argv'][1]==str(OUT/'worker.py');os.kill(p.pid,signal.SIGTERM);p.wait(timeout=30)
  if daemon:
   before=scene();response=control('quit');time.sleep(2);save(OUT/'mps_stop.json',dict(returncode=response.returncode,stdout=response.stdout,stderr=response.stderr,owned_before=before,owned_after=scene()))
if __name__=='__main__':
 try:main()
 except BaseException as e:save(OUT/'failure_controller.json',dict(error=repr(e),unix=time.time()));raise
