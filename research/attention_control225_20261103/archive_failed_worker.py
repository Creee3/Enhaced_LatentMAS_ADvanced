"""Archive one stopped owned worker, without stopping or touching healthy ones."""
import argparse,gzip,hashlib,io,json,tarfile
from pathlib import Path,PurePosixPath
from deploy import PRIVATE,call

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pid',type=int,required=True);a=ap.parse_args()
    active=json.loads((PRIVATE/'cloud_active.json').read_text(encoding='utf-8'))
    script='''import json,sys,tarfile,hashlib,io,time
from pathlib import Path
run=Path(RUN_VALUE);sys.path.insert(0,str(run))
from cloud_common import identity,load
pid=PID_VALUE;worker=next(w for w in load(run/'workers.json') if w['pid']==pid);out=Path(worker['out'])
assert identity(pid) is None
failure=load(out/'failure.json');exit=load(out/'exit.json')
assert failure and exit and not exit['completed']
budget=load(run.parents[1]/'cumulative_budget.json')
own={k:v for k,v in budget['active_reservations'].items() if v['pid']==pid}
assert not own
files={}
def add(name,p):
 assert p.is_file() and not p.is_symlink();files[name]=p.read_bytes()
lock=load(run/'run_lock.json')
for name in list(lock['files'])+list(lock['snapshots'])+['run_lock.json','queue.json','manager.json','manager_status.json','manager_failure.json','manager_exit.json','manager.log','workers.json','launch_receipt.json','launch_preflight.json']:
 if (run/name).is_file():add('run_snapshot/'+name,run/name)
for p in out.rglob('*'):
 if p.is_file():add('stopped_attempt/'+p.relative_to(out).as_posix(),p)
add('stopped_attempt/worker.log',Path(worker['log']))
base=run.parents[1]/'evidence/gpqa_repair_8096_20261028T161254Z/bundle'
for name in list(load(base/'source_lock.json')['files'])+['source_lock.json']:
 add('referenced_bundle37/'+name,base/name)
tracefile=out/'trace.jsonl'
trace=[json.loads(x) for x in tracefile.read_bytes().split(b'\\n') if x] if tracefile.exists() else []
raw=out/'raw/requests_raw.jsonl'
rawrows=[json.loads(x) for x in raw.read_bytes().split(b'\\n') if x] if raw.exists() else []
proof=dict(pid=pid,gpu=worker['gpu'],initial_identity=worker['initial_identity'],admission=worker['admission'],
 stopped=True,current=failure.get('current'),committed=exit['committed'],request_starts=sum(x['event']=='request_started' for x in trace),
 raw_records=len(rawrows),own_reserved=0,unknown=budget['unknown_cost_upper_bound_tokens'],manual_settlements=0,
 actual=budget['actual_tokens'],healthy_workers_untouched=True,source_lock_sha256=hashlib.sha256((run/'run_lock.json').read_bytes()).hexdigest())
files['failure_cost_and_ownership.json']=(json.dumps(proof,ensure_ascii=False,indent=2)+'\\n').encode()
manifest={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in files.items()}
files['archive_manifest.json']=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\\n').encode()
class Writer:
 def __init__(self):self.h=hashlib.sha256();self.size=0
 def write(self,b):self.h.update(b);self.size+=len(b);return sys.stdout.buffer.write(b)
 def flush(self):sys.stdout.buffer.flush()
w=Writer()
with tarfile.open(fileobj=w,mode='w|gz') as tar:
 for n,b in files.items():
  info=tarfile.TarInfo(n);info.size=len(b);tar.addfile(info,io.BytesIO(b))
w.flush()
(run/'management'/('failure_'+str(pid)+'_stream.json')).write_text(json.dumps(dict(sha256=w.h.hexdigest(),bytes=w.size)))
'''.replace('RUN_VALUE',repr(active['run'])).replace('PID_VALUE',str(a.pid))
    raw=call(script,180);digest=hashlib.sha256(raw).hexdigest()
    proof=json.loads(call('from pathlib import Path\nprint((Path('+repr(active['run'])+')/"management"/'+repr('failure_'+str(a.pid)+'_stream.json')+').read_text())'))
    assert digest==proof['sha256'] and len(raw)==proof['bytes']
    archive=PRIVATE/('failed_pid'+str(a.pid)+'_full_'+digest[:12]+'.tar.gz');archive.write_bytes(raw)
    plain=gzip.decompress(raw);dest=PRIVATE/('failed_pid'+str(a.pid)+'_verified_'+digest[:12]);dest.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(plain)) as tar:
        members=tar.getmembers();names=[m.name for m in members];assert len(names)==len(set(names))
        manifest=json.load(tar.extractfile('archive_manifest.json'));assert set(names)==set(manifest)|{'archive_manifest.json'}
        for m in members:
            name=PurePosixPath(m.name);assert m.isfile() and not name.is_absolute() and '..' not in name.parts
            b=tar.extractfile(m).read();p=dest.joinpath(*name.parts);assert p.resolve().is_relative_to(dest.resolve())
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
            if m.name in manifest:
                e=manifest[m.name];assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
    cost=json.loads((dest/'failure_cost_and_ownership.json').read_text(encoding='utf-8'))
    result=dict(archive_sha256=digest,bytes=len(raw),members=len(names),stream_sha=True,gzip_crc=True,tar_exact_members=True,all_member_sha=True,safe_full_extract_sha=True,cost=cost)
    name='failure_pid'+str(a.pid)+'_verified.json'
    (PRIVATE/name).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    import base64
    payload=base64.b64encode((PRIVATE/name).read_bytes()).decode()
    call('import base64\nfrom pathlib import Path\n(Path('+repr(active['run'])+')/"management"/'+repr(name)+').write_bytes(base64.b64decode('+repr(payload)+'))')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
