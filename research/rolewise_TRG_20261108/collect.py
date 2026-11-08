"""Streaming archive with remote SHA and local full safe-extraction readback."""
import base64,gzip,hashlib,io,json,sys,tarfile
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent
PRIVATE=Path('D:/liuxin_workspace/ICML2027/advanced_MOC/experiments/latentmas_rolewise_TRG576_20261108')
sys.path.insert(0,str(HERE.parent/'TRG225_a100_20261106'));from capture_current import call

def collect(phase,*,failure=False,final=False):
 remote=json.loads((PRIVATE/'cloud_active.json').read_text())['run'];tag=phase+('_failure' if failure else '_final' if final else '')
 script=r'''
from pathlib import Path
import sys,json,subprocess,tarfile,io,hashlib
r=Path(REMOTE);sys.path[:0]=[str(r),str(r.parent/'TRG225_a100_s42_l10_20261106T155639Z')];from common import load,save,sha,verify,ROOT,SCIENCE,BASE,identity
phase=PHASE;failure=FAILURE;final=FINAL;tag=TAG
source=verify();folder=r/'management'/phase;meta=load(folder/'identity.json');actual=identity(meta['identity']['pid']);assert actual is None or actual['start_ticks']!=meta['identity']['start_ticks']
for w in load(folder/'workers.json'):
 actual=identity(w['pid']);assert actual is None or actual['start_ticks']!=w['identity']['start_ticks']
if not failure:
 audit=subprocess.run([sys.executable,str(r/'audit.py'),phase],cwd=r,capture_output=True,text=True);assert audit.returncode==0,audit.stderr
 if final:assert (r/'final_summary.json').exists()
else:save(r/'failure_budget_snapshot.json',load(ROOT/'cumulative_budget.json'))
refs=dict(source=source,science_run=str(SCIENCE),science_run_archive_sha256='0e54558e885eeafb998ef90a5a70fd5d7d0864802ddb15c1165bf7657ee187c4',source_lock=load(SCIENCE/'run_lock.json'),bundle_lock=load(BASE.parent/'source_lock.json'),model_lock=load(BASE/'model_lock.json'),pretrained_weights_copied=False);save(r/'source_reference.json',refs)
files={p.relative_to(r).as_posix():p.read_bytes() for p in r.rglob('*') if p.is_file() and not p.is_symlink() and not p.name.endswith('_archive_stream.json')}
for prefix,base,lock in [('external_reference/science',SCIENCE,load(SCIENCE/'run_lock.json')),('external_reference/bundle',BASE.parent,load(BASE.parent/'source_lock.json'))]:
 for n,expected in lock['files'].items():
  p=base/n;assert sha(p)==expected['sha256'] and not p.name.endswith(('.safetensors','.bin','.pt'))
  files[prefix+'/'+n]=p.read_bytes()
files['manifest.json']=json.dumps({n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in files.items()}).encode()
class Writer:
 def __init__(self):self.h=hashlib.sha256();self.size=0
 def write(self,b):self.h.update(b);self.size+=len(b);return sys.stdout.buffer.write(b)
 def flush(self):sys.stdout.buffer.flush()
w=Writer()
with tarfile.open(fileobj=w,mode='w|gz') as t:
 for n,v in files.items():m=tarfile.TarInfo(n);m.size=len(v);t.addfile(m,io.BytesIO(v))
w.flush();save(r/(tag+'_archive_stream.json'),dict(sha256=w.h.hexdigest(),bytes=w.size))
'''.replace('REMOTE',repr(remote)).replace('PHASE',repr(phase)).replace('FAILURE',repr(failure)).replace('FINAL',repr(final)).replace('TAG',repr(tag))
 raw=call(script,1200);digest=hashlib.sha256(raw).hexdigest();receipt=json.loads(call('from pathlib import Path\nprint((Path('+repr(remote)+')/'+repr(tag+'_archive_stream.json')+').read_text())'));assert receipt==dict(sha256=digest,bytes=len(raw))
 archive=PRIVATE/(tag+'_'+digest[:12]+'.tar.gz');archive.write_bytes(raw);dest=PRIVATE/('verified_'+tag+'_'+digest[:12]);dest.mkdir(exist_ok=False)
 with tarfile.open(fileobj=io.BytesIO(gzip.decompress(raw))) as t:
  members=t.getmembers();names=[m.name for m in members];manifest=json.load(t.extractfile('manifest.json'));assert len(names)==len(set(names)) and set(names)==set(manifest)|{'manifest.json'}
  for m in members:
   n=PurePosixPath(m.name);assert m.isfile() and not n.is_absolute() and '..' not in n.parts and ':' not in m.name and '\\' not in m.name
   value=t.extractfile(m).read();path=dest.joinpath(*n.parts);assert path.resolve().is_relative_to(dest.resolve());path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(value)
   if m.name in manifest:assert len(value)==manifest[m.name]['bytes'] and hashlib.sha256(value).hexdigest()==manifest[m.name]['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
 proof=dict(passed=True,phase=phase,failure_archive=failure,final=final,archive_sha256=digest,bytes=len(raw),members=len(names),remote_stream_sha=True,gzip_crc=True,exact_members=True,per_member_sha=True,safe_full_extract_readback_sha=True)
 filename=tag+'_archive_verified.json';b=(json.dumps(proof,indent=2)+'\n').encode();(PRIVATE/filename).write_bytes(b);call('from pathlib import Path\nimport base64\nPath('+repr(remote+'/'+filename)+').write_bytes(base64.b64decode('+repr(base64.b64encode(b).decode())+'))')
 for name in [phase+'_audit_verified.json','final_summary.json','final_report.md']:
  if (dest/name).exists():(PRIVATE/name).write_bytes((dest/name).read_bytes())
 print(json.dumps(dict(archive=proof)));return proof
if __name__=='__main__':collect(sys.argv[1],failure='--failure' in sys.argv,final='--final' in sys.argv)
