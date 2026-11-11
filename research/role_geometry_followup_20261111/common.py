"""Independent state for the authorized 120-question role intervention."""
import contextlib, hashlib, json, os, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025')
SCIENCE=ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z'
BASE=ROOT/'evidence/gpqa_repair_8096_20261028T161254Z/bundle/runtime'
OLD=ROOT/'evidence/role_geometry120_s42_l10_20261110T073338Z'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()));tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,p)
def append(p,x):
 with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def rows(p):return [json.loads(b) for b in Path(p).read_bytes().split(b'\n') if b.strip()] if Path(p).exists() else []
def records():return [r for p in (HERE/'attempts').glob('*/results.jsonl') for r in rows(p)]
def identity(pid):
 try:
  p=Path('/proc')/str(pid);raw=(p/'stat').read_text();f=raw[raw.rindex(')')+2:].split()
  if f[0]=='Z':return None
  env=dict(z.split(b'=',1) for z in (p/'environ').read_bytes().split(b'\0') if b'=' in z)
  return dict(pid=int(pid),uid=p.stat().st_uid,start_ticks=f[19],argv=[z.decode() for z in (p/'cmdline').read_bytes().split(b'\0') if z],CUDA_VISIBLE_DEVICES=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode())
 except (PermissionError,FileNotFoundError,ProcessLookupError):return None
def verify():
 lock=load(HERE/'run_lock.json')
 for n,m in lock['files'].items():assert sha(HERE/n)==m['sha256'],n
 assert sha(OLD/'run_lock.json')==lock['old_run_lock_sha256']
 old=load(OLD/'run_lock.json')
 for n,m in old['files'].items():assert sha(OLD/n)==m['sha256'],n
 assert sha(BASE/'model_lock.json')==lock['model_lock_sha256']
 sys.path.insert(0,str(SCIENCE));from cloud_common import verify as old_verify
 science=old_verify();assert science['run_lock_sha256']==lock['science_run_lock_sha256']
 return dict(passed=True,run_lock_sha256=sha(HERE/'run_lock.json'),new_files=len(lock['files']),science=science)
@contextlib.contextmanager
def locked():
 import fcntl
 with (HERE/'state.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def choose(queue,phase):
 eng=set(queue['engineering_indices'])
 return next((t for t in queue['tasks'] if t['state']=='pending' and (phase=='formal' or t['index'] in eng)),None)
def claim(phase,out):
 with locked():
  q=load(HERE/'queue.json');t=choose(q,phase)
  if t is None:return None
  t.update(state='running',owner=identity(os.getpid()),attempt=str(out));save(HERE/'queue.json',q);return dict(t)
def commit(task,row,out):
 with locked():
  q=load(HERE/'queue.json');t=q['tasks'][task['index']];assert t['state']=='running' and t['owner']['pid']==os.getpid()
  assert row['index'] not in {r['index'] for r in records()}
  append(out/'results.jsonl',row);t.update(state='complete',committed_methods=['diagnostic'],methods=[],completed=time.time());save(HERE/'queue.json',q)
def hold(task,error):
 if task is None:return
 with locked():
  q=load(HERE/'queue.json');t=q['tasks'][task['index']]
  if t['state']!='complete':t.update(state='held',error=repr(error));save(HERE/'queue.json',q)
