import contextlib,fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
from queue_logic import choose,terminal
RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
BASE=ROOT/'evidence/gpqa_repair_8096_20261028T161254Z/bundle'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def load(p,default=None):return json.loads(Path(p).read_text(encoding='utf-8')) if Path(p).exists() else default
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    with tmp.open('wb') as f:f.write((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode());f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def append(p,x):
    with Path(p).open('ab') as f:f.write((json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n').encode());f.flush();os.fsync(f.fileno())
@contextlib.contextmanager
def qlock():
    with (RUN/'queue.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX);yield
def identity(pid):
    p=Path('/proc')/str(pid)
    try:
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
        return dict(pid=pid,uid=p.stat().st_uid,start_ticks=fields[19],argv=[s.decode() for s in (p/'cmdline').read_bytes().split(b'\0') if s],CUDA_VISIBLE_DEVICES=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode())
    except (FileNotFoundError,ProcessLookupError):return None
def gpu_rows():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
    return [dict(gpu=int(a),uuid=b.strip(),utilization=float(c),free_mib=float(d),total_mib=float(e)) for a,b,c,d,e in (line.split(',') for line in raw.strip().splitlines())]
def verify():
    lock=load(RUN/'run_lock.json')
    for n,v in lock['files'].items():
        p=RUN/n
        if not p.is_file() or p.stat().st_size!=v['bytes'] or sha(p)!=v['sha256']:raise RuntimeError('run SHA mismatch: '+n)
    original=load(BASE/'source_lock.json')
    if sha(BASE/'source_lock.json')!=lock['base_source_lock_sha256']:raise RuntimeError('bundle lock changed')
    for n,v in original['files'].items():
        if sha(BASE/n)!=v['sha256']:raise RuntimeError('bundle SHA mismatch: '+n)
    return dict(run_files=len(lock['files']),bundle37=len(original['files']),all_sha_match=True,run_lock_sha256=sha(RUN/'run_lock.json'))
def all_rows():
    out=[]
    for p in (RUN/'attempts').glob('*/results.jsonl'):
        raw=p.read_bytes()
        if raw and not raw.endswith(b'\n'):raise RuntimeError('partial durable result: '+str(p))
        out.extend(json.loads(x) for x in raw.split(b'\n') if x)
    return out
def key(r):return (r['dataset'],r['question_id'],r['latent_steps'],r['method'])
