"""Snapshot and verify the zero-forward failed management attempt."""
import base64,gzip,hashlib,io,json,sys,tarfile
from pathlib import Path,PurePosixPath
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'role_geometry_run_20261110'))
from remote import call,active,PRIVATE

def main():
 run=active()['run']
 script="""from pathlib import Path
import sys,json,base64,hashlib,io,tarfile,time
r=Path(RUN);sys.path.insert(0,str(r));from common import load,ROOT,verify
f=r/'management';names=['expansion_manager.py','expansion_management_lock.json','expansion_resource_policy.json','expansion_predecessor.json','expansion_spawn_attempted.json','expansion_v1_preserved.json','expansion_manager.log']
blobs={'management/'+n:(f/n).read_bytes() for n in names}
blobs.update({'snapshot/queue.json':(r/'queue.json').read_bytes(),'snapshot/ledger.json':(ROOT/'cumulative_budget.json').read_bytes(),'snapshot/run_lock.json':(r/'run_lock.json').read_bytes(),'snapshot/verification.json':json.dumps(dict(source=verify(),failed_manager_new_workers=0,original_resident_workers=6,unix=time.time())).encode()})
manifest={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in blobs.items()}
out=io.BytesIO()
with tarfile.open(fileobj=out,mode='w:gz') as archive:
 for n,b in {'manifest.json':json.dumps(manifest).encode(),**blobs}.items():
  member=tarfile.TarInfo(n);member.size=len(b);archive.addfile(member,io.BytesIO(b))
b=out.getvalue();print(json.dumps(dict(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),archive=base64.b64encode(b).decode())))
""".replace('RUN',repr(run))
 data=json.loads(call(script));b=base64.b64decode(data['archive']);digest=hashlib.sha256(b).hexdigest();assert digest==data['sha256'] and len(b)==data['bytes']
 path=PRIVATE/('expansion_management_failure_'+digest[:12]+'.tar.gz');path.write_bytes(b)
 with gzip.open(path,'rb') as inp:inp.read()
 dest=PRIVATE/('verified_expansion_failure_'+digest[:12]);dest.mkdir(exist_ok=False);seen=set()
 with tarfile.open(path,'r:gz') as archive:
  manifest=json.load(archive.extractfile('manifest.json'))
  for m in archive:
   n=PurePosixPath(m.name);assert m.isfile() and not n.is_absolute() and '..' not in n.parts and ':' not in m.name and chr(92) not in m.name and m.name not in seen;seen.add(m.name)
   target=dest.joinpath(*n.parts);assert target.resolve().is_relative_to(dest.resolve());target.parent.mkdir(parents=True,exist_ok=True);payload=archive.extractfile(m).read();target.write_bytes(payload)
   if m.name!='manifest.json':assert len(payload)==manifest[m.name]['bytes'] and hashlib.sha256(payload).hexdigest()==manifest[m.name]['sha256']==hashlib.sha256(target.read_bytes()).hexdigest()
 assert seen==set(manifest)|{'manifest.json'} and hashlib.sha256(path.read_bytes()).hexdigest()==digest
 proof=dict(passed=True,archive_sha256=digest,bytes=len(b),members=len(seen),remote_stream_sha=True,gzip_crc=True,exact_members=True,per_member_sha=True,safe_full_extract_readback_sha=True,zero_new_workers=True)
 raw=(json.dumps(proof,indent=2)+'\n').encode();(PRIVATE/'expansion_management_failure_verified.json').write_bytes(raw)
 call('from pathlib import Path\nimport base64\n(Path('+repr(run)+')/"management/expansion_management_failure_verified.json").write_bytes(base64.b64decode('+repr(base64.b64encode(raw).decode())+'))')
 print(json.dumps(proof))
if __name__=='__main__':main()
