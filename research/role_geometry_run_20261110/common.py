"""Private-run state, locks, provenance and exact keys; no model imports."""
import contextlib,hashlib,json,os,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025')
SCIENCE=ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z'
BASE=ROOT/'evidence/gpqa_repair_8096_20261028T161254Z/bundle/runtime'
def method_plan():
 import json
 plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
 expected=['Full','Gen','H','L','H-OBF','L-OBF','G','T','R','R-shuffle','Pseudo-R','Random-matched']
 assert plan['kind']=='role_geometry120' and plan['methods']==expected
 return plan
METHODS=method_plan()['methods'] if (HERE/'plan.json').exists() else []
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def append(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush()
def rows(p):return [json.loads(b) for b in Path(p).read_bytes().split(b'\n') if b.strip()] if Path(p).exists() else []
def key(x):return (x['dataset'],x['question_id'],x['latent_steps'],x['method'])
def records():return [x for p in (HERE/'attempts').glob('*/results.jsonl') for x in rows(p)]
@contextlib.contextmanager
def locked():
 import fcntl
 with (HERE/'state.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def identity(pid):
 try:
  p=Path('/proc')/str(pid);stat=p/'stat';raw=stat.read_text();fields=raw[raw.rindex(')')+2:].split()
  if fields[0]=='Z':return None
  env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
  return dict(pid=int(pid),uid=p.stat().st_uid,start_ticks=fields[19],argv=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x],CUDA_VISIBLE_DEVICES=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode())
 except (PermissionError,FileNotFoundError,ProcessLookupError):return None
def verify():
 lock=load(HERE/'run_lock.json')
 for name,meta in lock['files'].items():assert sha(HERE/name)==meta['sha256'],name
 sys.path.insert(0,str(SCIENCE));from cloud_common import verify as old_verify
 old=old_verify();assert old['run_lock_sha256']==lock['science_run_lock_sha256']
 return dict(passed=True,run_lock_sha256=sha(HERE/'run_lock.json'),new_files=len(lock['files']),science=old)
def choose(queue,phase):
 eligible=set(queue['engineering_indices']) if phase=='engineering' else None
 active=[t for t in queue['tasks'] if t['state'] in ['pending','running'] and t['methods'] and (eligible is None or t['index'] in eligible)]
 if not active:return None
 if phase=='engineering':return next((t for t in active if t['state']=='pending'),None)
 batch=min(t['batch'] for t in active)
 return next((t for t in active if t['batch']==batch and t['state']=='pending'),None)
def claim(phase,out):
 with locked():
  q=load(HERE/'queue.json');task=choose(q,phase)
  if task is None:return None
  task.update(state='running',owner=identity(os.getpid()),attempt=str(out),started=time.time());save(HERE/'queue.json',q);return dict(task)
def commit(task,row,out):
 with locked():
  q=load(HERE/'queue.json');live=q['tasks'][task['index']];assert live['state']=='running' and live['owner']['pid']==os.getpid() and row['method'] in live['methods']
  assert key(row) not in {key(x) for x in records()},'duplicate new result key'
  append(out/'results.jsonl',row);live['methods'].remove(row['method']);live['committed_methods'].append(row['method'])
  if not live['methods']:live.update(state='complete',completed=time.time())
  save(HERE/'queue.json',q)
def hold(task,error):
 if task is None:return
 with locked():
  q=load(HERE/'queue.json');live=q['tasks'][task['index']]
  if live['state']!='complete':live.update(state='held',error=repr(error),failed=time.time());save(HERE/'queue.json',q)
