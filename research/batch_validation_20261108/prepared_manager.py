"""Guarded native TRG4/MPS launcher. Default is validation only, no GPU job."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025');SCIENCE=ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z'
def main(execute=False):
 sys.path.insert(0,str(SCIENCE));from cloud_common import load,save,verify,identity
 plan=load(OUT/'launch_plan.json');report=load(OUT/'validation_report.json')
 assert OUT.parent.resolve()==(ROOT/'evidence').resolve() and OUT.name.startswith('trg4_mps_smoke_')
 assert not (OUT/'STOP').exists() and not (OUT/'completed.json').exists()
 assert plan['status']=='VALIDATED_FOUR_CASE_SMOKE' and plan['case_indices']==[0,1,2,3] and plan['methods']==['T','R','G']
 assert (plan['seed'],plan['latent_steps'],plan['cap'],plan['context'],plan['precision'],plan['norm'])==(42,10,8192,32768,'BF16','norm_only')
 assert plan['max_workers']==4 and plan['gpu']==0 and plan['cumulative_position_limit']==115000000
 assert report['passed'] and all(report[k] for k in ['HF128_exact','MPS128_exact','MPS_prefix_four_exact','native_cap8192_EOS','all_owned_services_stopped'])
 assert hashlib.sha256((OUT/'validation_report.json').read_bytes()).hexdigest()==plan['validation_report_sha256']
 assert hashlib.sha256((OUT/'full_role_worker.py').read_bytes()).hexdigest()==plan['worker_sha256']
 assert verify()['run_lock_sha256']==plan['source_run_lock_sha256']
 budget=load(ROOT/'cumulative_budget.json');upper=4*9*32768
 assert budget['actual_tokens']+budget['reserved_tokens']+budget['unknown_cost_upper_bound_tokens']+upper<=115000000
 if not execute:print(json.dumps(dict(validated=True,launched=False,conservative_upper_positions=upper)));return
 guard=(OUT/'manager.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'workers.json').exists(),'Never restart completed or partial attempts blindly'
 pipe=Path('/tmp')/('ltrg_'+str(os.getpid()));pipe.mkdir(mode=0o700);(OUT/'mps_logs').mkdir(mode=0o700)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=plan['gpu_uuid'],CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(OUT/'mps_logs'),LATENTMAS_SHARED_LEDGER=str(ROOT/'cumulative_budget.json'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
 def owned_scene():
  found=[]
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   x=identity(int(p.name))
   if not x:continue
   try:e=dict(v.split(b'=',1) for v in (p/'environ').read_bytes().split(b'\0') if b'=' in v)
   except (FileNotFoundError,PermissionError,ProcessLookupError):continue
   if e.get(b'CUDA_MPS_PIPE_DIRECTORY',b'').decode()==str(pipe):found.append(x)
  return found
 workers=[];daemon=False
 try:
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    argv=(p/'cmdline').read_bytes().split(b'\0');name=Path(argv[0].decode()).name if argv and argv[0] else '';assert name not in ['nvidia-cuda-mps-control','nvidia-cuda-mps-server'],'Existing MPS server; leave it untouched'
   except (FileNotFoundError,PermissionError,ProcessLookupError):pass
  for i in range(4):
   if (OUT/'STOP').exists():break
   started=time.monotonic()
   while True:
    raw=subprocess.run(['nvidia-smi','-i',plan['gpu_uuid'],'--query-gpu=utilization.gpu,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout;u,f,t=map(int,raw.strip().split(','));live=any(p.poll() is None for p in workers)
    if ((u<=30 and f/t>=.70) if not live else f>=21*1024):break
    if time.monotonic()-started>120:raise RuntimeError('Fresh admission unavailable')
    time.sleep(1)
   if not daemon:
    start=subprocess.run(['/usr/bin/nvidia-cuda-mps-control','-d'],env=env,capture_output=True,text=True,timeout=30);assert start.returncode==0;daemon=True;save(OUT/'manager.json',dict(identity=identity(os.getpid()),pipe=str(pipe),source=verify(),scope='four_case_TRG_engineering_smoke'))
   with (OUT/f'case{i}.log').open('ab') as log:p=subprocess.Popen([sys.executable,str(OUT/'full_role_worker.py'),'--index',str(i)],cwd=OUT,env=env,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
   workers.append(p);time.sleep(.1);save(OUT/'workers.json',[dict(pid=p.pid,identity=identity(p.pid),case=k) for k,p in enumerate(workers)])
   start=time.monotonic()
   while not (OUT/f'case{i}/model_ready.json').exists():
    if p.poll() is not None or list(OUT.glob('failure_case*.json')):raise RuntimeError('Worker failed; preserve attempt, no retry')
    if time.monotonic()-start>600:raise RuntimeError('Loading timeout')
    time.sleep(1)
  started=time.monotonic()
  while any(p.poll() is None for p in workers):
   if time.monotonic()-started>7200:raise RuntimeError('Bounded run timeout; preserve ownership')
   time.sleep(1)
  assert len(workers)==4 and all(p.returncode==0 for p in workers)
  save(OUT/'completed.json',dict(passed=True,cases=4,methods=['T','R','G'],engineering_only=True,accuracy_computed=False))
 finally:
  # Do not kill healthy in-flight requests on an unrelated worker failure.
  start=time.monotonic()
  while any(p.poll() is None for p in workers):
   if time.monotonic()-start>7200:raise RuntimeError('Healthy workers still active; keep service and ownership evidence')
   time.sleep(1)
  if daemon:
   before=owned_scene();stop=subprocess.run(['/usr/bin/nvidia-cuda-mps-control'],input='quit\n',env=env,capture_output=True,text=True,timeout=30);time.sleep(2);save(OUT/'mps_stop.json',dict(returncode=stop.returncode,owned_before=before,owned_after=owned_scene()))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args();main(a.execute)
