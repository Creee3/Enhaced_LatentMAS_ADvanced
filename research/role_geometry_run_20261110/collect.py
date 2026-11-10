"""Bounded-memory tar stream, SHA/CRC/member/safe-extraction readback verification."""
import base64,gzip,hashlib,io,json,subprocess,tarfile,time
from pathlib import Path,PurePosixPath
from remote import PRIVATE,PYTHON,call,active

def file_sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def collect(tag):
 run=active()['run'];script=r"""
from pathlib import Path
import sys,json,tarfile,io,hashlib
r=Path(RUN);sys.path.insert(0,str(r));from common import load,save,sha,verify,ROOT,SCIENCE,BASE
source=verify();tag=TAG
from common import identity
live_dirs=set()
for worker in load(r/'management/workers.json') if (r/'management/workers.json').exists() else []:
 a=identity(worker['pid'])
 if a is not None and a['start_ticks']==worker['identity']['start_ticks']:live_dirs.add(Path(worker['out']))
save(r/'archive_reference.json',dict(source=source,science=str(SCIENCE),science_lock=load(SCIENCE/'run_lock.json'),bundle_lock=load(BASE.parent/'source_lock.json'),model_lock=load(BASE/'model_lock.json'),pretrained_weights_copied=False,budget=load(ROOT/'cumulative_budget.json')))
files={};blobs={}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
for p in r.rglob('*'):
 if not p.is_file() or p.is_symlink() or '__pycache__' in p.parts or p.name.endswith('_archive_stream.json'):continue
 n=p.relative_to(r).as_posix()
 if tag=='failure_scene' and any(base in p.parents for base in live_dirs):continue
 if p.name in ['current.json','queue.json'] or p.parent==r/'management':blobs[n]=p.read_bytes()
 else:files[n]=p
for prefix,base,lock in [('external_reference/science',SCIENCE,load(SCIENCE/'run_lock.json')),('external_reference/bundle',BASE.parent,load(BASE.parent/'source_lock.json'))]:
 for n,expected in lock['files'].items():
  p=base/n;assert sha(p)==expected['sha256'] and not p.name.endswith(('.safetensors','.bin','.pt'));files[prefix+'/'+n]=p
manifest={n:dict(bytes=p.stat().st_size,sha256=digest(p)) for n,p in files.items()}
manifest.update({n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in blobs.items()})
class Writer:
 def __init__(self):self.h=hashlib.sha256();self.size=0
 def write(self,b):self.h.update(b);self.size+=len(b);return sys.stdout.buffer.write(b)
 def flush(self):sys.stdout.buffer.flush()
w=Writer()
with tarfile.open(fileobj=w,mode='w|gz') as t:
 b=json.dumps(manifest).encode();m=tarfile.TarInfo('manifest.json');m.size=len(b);t.addfile(m,io.BytesIO(b))
 for n,p in sorted(files.items()):
  m=tarfile.TarInfo(n);m.size=manifest[n]['bytes']
  with p.open('rb') as f:t.addfile(m,f)
  assert p.stat().st_size==m.size and digest(p)==manifest[n]['sha256'],'file changed while archived: '+n
 for n,b in sorted(blobs.items()):m=tarfile.TarInfo(n);m.size=len(b);t.addfile(m,io.BytesIO(b))
w.flush();save(r/(tag+'_archive_stream.json'),dict(sha256=w.h.hexdigest(),bytes=w.size))
""".replace('RUN',repr(run)).replace('TAG',repr(tag))
 provisional=PRIVATE/(tag+'_'+str(time.time_ns())+'.partial.tar.gz')
 with provisional.open('xb') as out:
  p=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','DevContainer',PYTHON+' -'],stdin=subprocess.PIPE,stdout=out,stderr=subprocess.PIPE)
  _,err=p.communicate(script.encode('utf-8'),timeout=1800)
 if p.returncode:raise RuntimeError(err.decode(errors='replace')[-4000:])
 digest=file_sha(provisional);receipt=json.loads(call('from pathlib import Path\nprint((Path('+repr(run)+')/'+repr(tag+'_archive_stream.json')+').read_text())'));assert receipt==dict(sha256=digest,bytes=provisional.stat().st_size)
 archive=PRIVATE/(tag+'_'+digest[:12]+'.tar.gz');provisional.rename(archive);dest=PRIVATE/('verified_'+tag+'_'+digest[:12]);dest.mkdir(exist_ok=False)
 # Read to gzip EOF explicitly to verify trailer CRC and uncompressed length.
 with gzip.open(archive,'rb') as g:
  for _ in iter(lambda:g.read(1024*1024),b''):pass
 names=set();manifest=None
 with tarfile.open(archive,mode='r|gz') as t:
  for m in t:
   n=PurePosixPath(m.name);assert m.isfile() and not n.is_absolute() and '..' not in n.parts and ':' not in m.name and '\\' not in m.name and m.name not in names;names.add(m.name)
   inp=t.extractfile(m);target=dest.joinpath(*n.parts);assert target.resolve().is_relative_to(dest.resolve());target.parent.mkdir(parents=True,exist_ok=True)
   h=hashlib.sha256();size=0
   with target.open('xb') as f:
    for b in iter(lambda:inp.read(1024*1024),b''):f.write(b);h.update(b);size+=len(b)
   if m.name=='manifest.json':manifest=json.loads(target.read_text());continue
   assert manifest is not None and m.name in manifest and size==manifest[m.name]['bytes'] and h.hexdigest()==manifest[m.name]['sha256']==file_sha(target)
 assert names==set(manifest)|{'manifest.json'} and file_sha(archive)==digest
 proof=dict(passed=True,tag=tag,archive_sha256=digest,bytes=archive.stat().st_size,members=len(names),remote_stream_sha=True,gzip_crc=True,exact_members=True,per_member_sha=True,safe_full_extract_readback_sha=True,bounded_memory_stream=True)
 b=(json.dumps(proof,indent=2)+'\n').encode();(PRIVATE/(tag+'_archive_verified.json')).write_bytes(b);call('from pathlib import Path\nimport base64\n(Path('+repr(run)+')/'+repr(tag+'_archive_verified.json')+').write_bytes(base64.b64decode('+repr(base64.b64encode(b).decode())+'))')
 print(json.dumps(proof));return proof
if __name__=='__main__':collect(__import__('sys').argv[1])
