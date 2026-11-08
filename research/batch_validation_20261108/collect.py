"""Archive a stopped diagnostic with exact parent-packet and cost provenance."""
import argparse,base64,gzip,hashlib,io,json,sys,tarfile
from pathlib import Path,PurePosixPath
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'TRG225_a100_20261106'))
from capture_current import call
ROOT=Path('D:/liuxin_workspace/ICML2027/advanced_MOC/experiments')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('package',choices=['batch_validation_20261108','stream_validation_20261108','threaded_stream_validation_20261108']);a=ap.parse_args();private=ROOT/a.package;launch=json.loads((private/'launch.json').read_text());parent_manifest=json.loads((ROOT/'real_kv_batch_20261108/verified_4e1462811bbf/manifest.json').read_text());expected={f'packet{i}.pt':parent_manifest[f'packet{i}.pt']['sha256'] for i in range(4)}
 script=r'''
from pathlib import Path
import sys,json,hashlib,tarfile,io
r=Path(REMOTE);root=r.parents[1];old=root/'evidence/TRG225_a100_s42_l10_20261106T155639Z';sys.path.insert(0,str(old));from cloud_common import load,save,identity,verify,sha
launch=load(r/'launch.json');assert identity(launch['pid']) is None and load(r/'result.json')['passed'] and not (r/'failure.json').exists()
parent=Path(launch['parent']);expected=EXPECTED
assert load(parent/'archive_verified.json')['archive_sha256']=='4e1462811bbf1dffbfaad7e88cbfaa2888acc8e27ddeec49eccba6ca14aaefa0'
assert all(sha(parent/n)==v for n,v in expected.items())
tr=[json.loads(x) for x in (r/'trace.jsonl').read_bytes().split(b'\n') if x];starts={x['reservation_id'] for x in tr if x['event']=='request_started'};settled=[x for x in tr if x['event']=='request_settled'];ids={x['reservation_id'] for x in settled}
assert len(ids)==len(settled) and starts==ids and all(not x['failed'] and x['independent_unknown']==0 for x in settled)
cost=sum(x['actual_positions'] for x in settled);assert cost==load(r/'result.json')['charged_positions']
b=load(root/'cumulative_budget.json');assert not any(x.get('metadata',{}).get('experiment')==r.name for x in b['active_reservations'].values())
save(r/'cost_verified.json',dict(passed=True,unique_RIDs=len(ids),actual_positions=cost,unknown_added=0,own_reserved=0,process_exited=True,parent_packet_shas=expected,source=verify()))
files={p.relative_to(r).as_posix():p.read_bytes() for p in r.rglob('*') if p.is_file() and not p.is_symlink()};files['manifest.json']=json.dumps({n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in files.items()}).encode()
class Writer:
 def __init__(self):self.h=hashlib.sha256();self.size=0
 def write(self,b):self.h.update(b);self.size+=len(b);return sys.stdout.buffer.write(b)
 def flush(self):sys.stdout.buffer.flush()
w=Writer()
with tarfile.open(fileobj=w,mode='w|gz') as tar:
 for n,v in files.items():m=tarfile.TarInfo(n);m.size=len(v);tar.addfile(m,io.BytesIO(v))
w.flush();save(r/'archive_stream.json',dict(sha256=w.h.hexdigest(),bytes=w.size))
'''.replace('REMOTE',repr(launch['out'])).replace('EXPECTED',repr(expected))
 raw=call(script,300);digest=hashlib.sha256(raw).hexdigest();receipt=json.loads(call('from pathlib import Path\nprint((Path('+repr(launch['out'])+')/"archive_stream.json").read_text())'));assert receipt==dict(sha256=digest,bytes=len(raw));(private/('full_'+digest[:12]+'.tar.gz')).write_bytes(raw)
 dest=private/('verified_'+digest[:12]);dest.mkdir(exist_ok=False)
 with tarfile.open(fileobj=io.BytesIO(gzip.decompress(raw))) as tar:
  members=tar.getmembers();names=[m.name for m in members];manifest=json.load(tar.extractfile('manifest.json'));assert len(names)==len(set(names)) and set(names)==set(manifest)|{'manifest.json'}
  for m in members:
   name=PurePosixPath(m.name);assert m.isfile() and not name.is_absolute() and '..' not in name.parts and ':' not in m.name and '\\' not in m.name
   v=tar.extractfile(m).read();p=dest.joinpath(*name.parts);assert p.resolve().is_relative_to(dest.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
   if m.name in manifest:assert len(v)==manifest[m.name]['bytes'] and hashlib.sha256(v).hexdigest()==manifest[m.name]['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
 for n in ['result.json','cost_verified.json']:(private/n).write_bytes((dest/n).read_bytes())
 proof=dict(passed=True,archive_sha256=digest,bytes=len(raw),members=len(names),remote_stream_sha=True,gzip_crc=True,exact_members=True,per_member_sha=True,safe_full_extract_readback_sha=True);b=(json.dumps(proof,indent=2)+'\n').encode();(private/'archive_verified.json').write_bytes(b)
 call('from pathlib import Path\nimport base64\nPath('+repr(launch['out']+'/archive_verified.json')+').write_bytes(base64.b64decode('+repr(base64.b64encode(b).decode())+'))');print(json.dumps(dict(package=a.package,archive=proof,cost=json.loads((private/'cost_verified.json').read_text())['actual_positions'])))
if __name__=='__main__':main()
