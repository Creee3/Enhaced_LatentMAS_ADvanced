"""One bounded foreground test controller; only launches this authorized proxy."""
import json,os,subprocess,sys,time,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
OLD=ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z'
sys.path.insert(0,str(OLD))
from cloud_common import identity,load,save
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',LATENTMAS_SHARED_LEDGER=str(ROOT/'cumulative_budget.json'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def wait_for(test,timeout=1200):
 start=time.monotonic()
 while not test():
  failures=list(OUT.glob('failure_*.json'))
  if failures:raise RuntimeError('worker failure: '+failures[0].name)
  if time.monotonic()-start>timeout:raise RuntimeError('bounded wait exceeded')
  time.sleep(1)
def launch(mode,index=0):
 argv=[sys.executable,str(OUT/'worker.py'),mode,'--index',str(index)]
 with (OUT/(mode+str(index)+'.log')).open('ab') as log:
  p=subprocess.Popen(argv,cwd=OUT,env=env,stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
 time.sleep(.1);meta=dict(pid=p.pid,identity=identity(p.pid),mode=mode,index=index);save(OUT/(mode+str(index)+'_identity.json'),meta);return p
def admit(first,name):
 start=time.monotonic()
 while True:
  fresh=subprocess.run(['nvidia-smi','-i','0','--query-gpu=utilization.gpu,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
  u,f,t=map(int,fresh.strip().split(','));save(OUT/(name+'_admission.json'),dict(util=u,free=f,total=t,unix=time.time()))
  if (u<=30 and f/t>=.70) if first else f>=21*1024:return
  if time.monotonic()-start>120:raise RuntimeError('fresh admission still unavailable')
  time.sleep(1)
def main():
 a=load(OUT/'launch.json');wait_for(lambda:(OUT/'prepared.json').exists() and identity(a['prepare_pid']) is None)
 import torch
 packets=[torch.load(OUT/f'packet{i}.pt',weights_only=True) for i in range(4)]
 lens=[j['ids'].shape[-1] for j in packets]
 prep_trace=[json.loads(b) for b in (OUT/'prefix/trace.jsonl').read_bytes().split(b'\n') if b]
 prep_cost=sum(x['actual_positions'] for x in prep_trace if x['event']=='request_settled')
 upper=prep_cost+sum(3*n+16+256 for n in lens)*2+4*(3*max(lens)+16+256)
 assert upper<=100000
 save(OUT/'preflight.json',dict(prefix_cost=prep_cost,total_conservative_bound=upper,prompt_lengths=lens,zero_extra_forward=True))
 # One model sequential and one model batch use same saved real KV packets.
 for mode in ['serial','batch']:
  meta=load(OUT/(mode+'0_identity.json'))
  if meta and identity(meta['pid']):
   actual=identity(meta['pid']);assert actual==meta['identity']
   wait_for(lambda:identity(meta['pid']) is None)
   assert load(OUT/mode/'done.json')['passed']
  if (OUT/mode/'done.json').exists():
   meta=load(OUT/(mode+'0_identity.json'));assert identity(meta['pid']) is None and load(OUT/mode/'done.json')['passed'];continue
  admit(True,mode)
  p=launch(mode);wait_for(lambda:p.poll() is not None);assert p.returncode==0 and load(OUT/mode/'done.json')['passed']
 # Four separate resident models, staged loading with fresh resource admission.
 processes=[]
 for i in range(4):
  admit(not i,'process'+str(i))
  p=launch('process',i);processes.append(p);wait_for(lambda i=i:(OUT/f'process{i}'/'ready.json').exists())
 cpus=sorted(os.sched_getaffinity(0));assert len(cpus)>=4
 for i,p in enumerate(processes):
  assert identity(p.pid) and identity(p.pid)['uid']==os.getuid()
  for task in Path('/proc',str(p.pid),'task').iterdir():os.sched_setaffinity(int(task.name),{cpus[i]})
 for rep in range(2):
  wait_for(lambda:all((OUT/f'process{i}'/f'waiting{rep}').exists() for i in range(4)))
  (OUT/f'go{rep}').touch()
  wait_for(lambda:all((OUT/f'process{i}'/f'result{rep}.json').exists() for i in range(4)))
 wait_for(lambda:all(p.poll() is not None for p in processes));assert all(p.returncode==0 for p in processes)
 result=dict(passed=True,engineering_only=True,serial=load(OUT/'serial/done.json'),batch=load(OUT/'batch/done.json'),processes=[load(OUT/f'process{i}/done.json') for i in range(4)],actual_positions=load(ROOT/'cumulative_budget.json')['actual_tokens']-a['ledger_before'])
 assert result['actual_positions']<=100000
 assert not any(x['metadata'].get('experiment')==OUT.name for x in load(ROOT/'cumulative_budget.json')['active_reservations'].values())
 save(OUT/'result.json',result)
if __name__=='__main__':
 try:main()
 except BaseException as e:save(OUT/'controller_failure.json',dict(error=repr(e),unix=time.time()));raise
