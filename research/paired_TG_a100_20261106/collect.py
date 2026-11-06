"""One completed full inspect and verified private archive, no raw stdout."""
import gzip,hashlib,io,json,subprocess,tarfile
from pathlib import Path,PurePosixPath
REPO=Path(__file__).resolve().parents[2]
PRIVATE=REPO.parent/'advanced_MOC/experiments/latentmas_paired_TG_a100_20261106'
def call(s):
 p=subprocess.run(['ssh','-o','BatchMode=yes','DevContainer','/home/ubuntu/.conda/envs/kv-graph/bin/python -'],input=s.encode(),capture_output=True,timeout=300)
 if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-3000:])
 return p.stdout
def main():
 PRIVATE.mkdir(exist_ok=True)
 for name in ['cloud_active.json','paired_verified.json']:
  b=call("from pathlib import Path\nprint(Path('/home/ubuntu/cloud_migration_20261106/paired_TG/"+name+"').read_text())");(PRIVATE/name).write_bytes(b)
 active=json.loads((PRIVATE/'cloud_active.json').read_bytes());run=active['run']
 raw=call("import sys,runpy\nfrom pathlib import Path\nr=Path("+repr(run)+");sys.path.insert(0,str(r));runpy.run_path(str(r/'inspect_run.py'),run_name='__main__')")
 (PRIVATE/'completed_inspect.json').write_bytes(raw)
 script='''import pathlib,json,hashlib,sys,tarfile,io
r=pathlib.Path(RUN);root=r.parents[1];files={}
for p in r.rglob('*'):
 if p.is_file() and not p.is_symlink():files['run/'+p.relative_to(r).as_posix()]=p.read_bytes()
base=root/'evidence/gpqa_repair_8096_20261028T161254Z/bundle'
for n in list(json.loads((base/'source_lock.json').read_text())['files'])+['source_lock.json']:files['bundle37/'+n]=(base/n).read_bytes()
files['ledger/cumulative_budget.json']=(root/'cumulative_budget.json').read_bytes()
manifest={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in files.items()};files['manifest.json']=json.dumps(manifest).encode()
class Writer:
 def __init__(self):self.h=hashlib.sha256();self.size=0
 def write(self,b):self.h.update(b);self.size+=len(b);return sys.stdout.buffer.write(b)
 def flush(self):sys.stdout.buffer.flush()
w=Writer()
with tarfile.open(fileobj=w,mode='w|gz') as tar:
 for n,b in files.items():
  info=tarfile.TarInfo(n);info.size=len(b);tar.addfile(info,io.BytesIO(b))
w.flush();(r/'management/archive_stream.json').write_text(json.dumps(dict(sha256=w.h.hexdigest(),bytes=w.size)))
'''.replace('RUN',repr(run))
 raw=call(script);digest=hashlib.sha256(raw).hexdigest();receipt=json.loads(call("from pathlib import Path\nprint((Path("+repr(run)+")/'management/archive_stream.json').read_text())"))
 assert receipt==dict(sha256=digest,bytes=len(raw));(PRIVATE/'completed_run.tar.gz').write_bytes(raw)
 plain=gzip.decompress(raw);dest=PRIVATE/'verified';dest.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(plain)) as tar:
  members=tar.getmembers();names=[m.name for m in members];manifest=json.load(tar.extractfile('manifest.json'));assert len(names)==len(set(names)) and set(names)==set(manifest)|{'manifest.json'}
  for m in members:
   path=PurePosixPath(m.name);assert m.isfile() and not path.is_absolute() and '..' not in path.parts
   b=tar.extractfile(m).read();p=dest.joinpath(*path.parts);assert p.resolve().is_relative_to(dest.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
   if m.name in manifest:assert len(b)==manifest[m.name]['bytes'] and hashlib.sha256(b).hexdigest()==manifest[m.name]['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
 proof=dict(passed=True,archive_sha256=digest,bytes=len(raw),members=len(names),remote_stream_sha=True,gzip_crc=True,exact_members=True,per_member_sha=True,safe_full_extract_readback_sha=True)
 (PRIVATE/'archive_verified.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
 import base64
 payload=base64.b64encode((PRIVATE/'archive_verified.json').read_bytes()).decode()
 call("import base64\nfrom pathlib import Path\n(Path("+repr(run)+")/'management/archive_verified.json').write_bytes(base64.b64decode("+repr(payload)+"))")
if __name__=='__main__':main()
