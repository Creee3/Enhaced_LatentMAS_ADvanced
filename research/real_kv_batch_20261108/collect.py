"""Verify stopped test, real cost, and a complete private stream archive."""
import base64,gzip,hashlib,io,json,sys,tarfile
from pathlib import Path,PurePosixPath
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'TRG225_a100_20261106'))
from capture_current import call
PRIVATE=Path('D:/liuxin_workspace/ICML2027/advanced_MOC/experiments/real_kv_batch_20261108')
def main():
 a=json.loads((PRIVATE/'launch.json').read_text());remote=a['out']
 script=r'''
from pathlib import Path
import json,sys,io,tarfile,hashlib,os
r=Path(REMOTE);root=r.parents[1];old=root/'evidence/TRG225_a100_s42_l10_20261106T155639Z';sys.path.insert(0,str(old));from cloud_common import load,save,identity,verify
assert load(r/'result.json')['passed'] and not list(r.glob('failure_*.json')) and not (r/'controller_failure.json').exists()
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  actual=identity(int(p.name))
  assert not (actual and str(r/'worker.py') in actual['argv'])
ids=set();positions=0;forwards=0
for p in r.glob('*/trace.jsonl'):
 rows=[json.loads(b) for b in p.read_bytes().split(b'\n') if b]
 started={x['reservation_id'] for x in rows if x['event']=='request_started'}
 settled=[x for x in rows if x['event']=='request_settled'];current={x['reservation_id'] for x in settled}
 assert len(current)==len(settled) and current==started and not ids&current
 assert all(not x['failed'] and x['independent_unknown']==0 for x in settled)
 ids|=current;positions+=sum(x['actual_positions'] for x in settled);forwards+=sum(x['forward_calls'] for x in settled)
b=load(root/'cumulative_budget.json');assert not any(x.get('metadata',{}).get('experiment')==r.name for x in b['active_reservations'].values())
assert positions<=100000
save(r/'cost_verified.json',dict(passed=True,unique_rids=len(ids),actual_positions=positions,native_forward_calls=forwards,unknown_added=0,own_reserved=0,source=verify(),ledger_delta=b['actual_tokens']-load(r/'launch.json')['ledger_before']))
files={p.relative_to(r).as_posix():p.read_bytes() for p in r.rglob('*') if p.is_file() and not p.is_symlink()}
manifest={n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in files.items()};files['manifest.json']=json.dumps(manifest).encode()
class Writer:
 def __init__(self):self.h=hashlib.sha256();self.size=0
 def write(self,b):self.h.update(b);self.size+=len(b);return sys.stdout.buffer.write(b)
 def flush(self):sys.stdout.buffer.flush()
w=Writer()
with tarfile.open(fileobj=w,mode='w|gz') as t:
 for n,v in files.items():
  m=tarfile.TarInfo(n);m.size=len(v);t.addfile(m,io.BytesIO(v))
w.flush();save(r/'archive_stream.json',dict(sha256=w.h.hexdigest(),bytes=w.size))
'''.replace('REMOTE',repr(remote))
 raw=call(script,600);digest=hashlib.sha256(raw).hexdigest()
 receipt=json.loads(call('from pathlib import Path\nprint((Path('+repr(remote)+')/"archive_stream.json").read_text())'))
 assert receipt==dict(sha256=digest,bytes=len(raw));(PRIVATE/('full_'+digest[:12]+'.tar.gz')).write_bytes(raw)
 dest=PRIVATE/('verified_'+digest[:12]);dest.mkdir(exist_ok=False)
 with tarfile.open(fileobj=io.BytesIO(gzip.decompress(raw))) as t:
  members=t.getmembers();names=[m.name for m in members];manifest=json.load(t.extractfile('manifest.json'));assert len(names)==len(set(names)) and set(names)==set(manifest)|{'manifest.json'}
  for m in members:
   name=PurePosixPath(m.name);assert m.isfile() and not name.is_absolute() and '..' not in name.parts and ':' not in m.name and '\\' not in m.name
   v=t.extractfile(m).read();p=dest.joinpath(*name.parts);assert p.resolve().is_relative_to(dest.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
   if m.name in manifest:assert len(v)==manifest[m.name]['bytes'] and hashlib.sha256(v).hexdigest()==manifest[m.name]['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
 for n in ['result.json','cost_verified.json']:(PRIVATE/n).write_bytes((dest/n).read_bytes())
 proof=dict(passed=True,archive_sha256=digest,bytes=len(raw),members=len(names),remote_stream_sha=True,gzip_crc=True,exact_members=True,per_member_sha=True,safe_full_extract_readback_sha=True)
 b=(json.dumps(proof,indent=2)+'\n').encode();(PRIVATE/'archive_verified.json').write_bytes(b)
 call('from pathlib import Path\nimport base64\nPath('+repr(remote+'/archive_verified.json')+').write_bytes(base64.b64decode('+repr(base64.b64encode(b).decode())+'))')
 print(json.dumps(proof))
if __name__=='__main__':main()
