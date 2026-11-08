"""Full archive of MPS prefix/native-generation smoke, with detached proof."""
import ast,base64,gzip,hashlib,io,json,sys,tarfile
from pathlib import Path,PurePosixPath
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'TRG225_a100_20261106'))
from capture_current import call
PRIVATE=Path('D:/liuxin_workspace/ICML2027/advanced_MOC/experiments/mps_native_smoke_20261108')
def main():
 a=json.loads((PRIVATE/'launch.json').read_text());script=r'''
from pathlib import Path
import json,sys,hashlib,tarfile,io,torch
r=Path(REMOTE);root=r.parents[1];old=root/'evidence/TRG225_a100_s42_l10_20261106T155639Z';sys.path.insert(0,str(old));from cloud_common import load,save,identity,verify,sha
a=load(r/'launch.json');rebuilt=Path(a['rebuilt']);result=load(r/'result.json');assert result['passed'] and load(r/'prefix_gate.json')['passed'] and not list(r.rglob('failure_*.json')) and identity(a['pid']) is None
stop=load(r/'mps_stop.json');assert stop['returncode']==0 and not stop['owned_after'];scene=load(r/'mps_scene.json');assert scene['server_list'].strip() and any(Path(x['argv'][0]).name=='nvidia-cuda-mps-server' and x['CUDA_VISIBLE_DEVICES']==a['gpu_uuid'] for x in scene['owned_processes'])
seen=set();cost=0
for p in r.rglob('trace.jsonl'):
 rows=[json.loads(b) for b in p.read_bytes().split(b'\n') if b];starts={x['reservation_id'] for x in rows if x['event']=='request_started'};settled=[x for x in rows if x['event']=='request_settled'];ids={x['reservation_id'] for x in settled}
 assert ids==starts and len(ids)==len(settled) and not ids&seen;seen|=ids;assert all(not x['failed'] and x['independent_unknown']==0 for x in settled);cost+=sum(x['actual_positions'] for x in settled)
for i in range(2):
 ready=load(r/f'hf{i}/ready.json');receipt=load(r/f'hf{i}/result.json');packet=torch.load(rebuilt/f'packet{i}.pt',weights_only=True)
 assert ready['bf16_cuda'] and ready['model_files_verified']==15 and ready['parameter_bytes']==16381470720 and abs(ready['target_norm']-1.3758320808410645)<1e-6
 assert receipt['first128_match'] and (receipt['natural_eos'] or receipt['capped']) and receipt['actual_positions']==packet['ids'].shape[-1]+receipt['generated_tokens']-1 and receipt['sdpa_checks']==36*receipt['generated_tokens']
b=load(root/'cumulative_budget.json');assert not any(x.get('metadata',{}).get('experiment') in [r.name,rebuilt.name] for x in b['active_reservations'].values())
assert cost<=load(r/'preflight.json')['total_upper_bound']<=50000
save(r/'cost_verified.json',dict(passed=True,actual_positions=cost,unique_RIDs=len(seen),unknown_added=0,own_reserved=0,source=verify(),ledger_delta=b['actual_tokens']-a['ledger_before']))
# Optional full-trajectory comparison to the historical A100 R answer, without accuracy.
oldrows=[]
for p in (old/'attempts').glob('*/results.jsonl'):oldrows += [json.loads(b) for b in p.read_bytes().split(b'\n') if b]
questions=load(old/'questions.json');comparison=[]
for i in range(2):
 historical=next(x for x in oldrows if x['method']=='R' and x['question_id']==questions[i]['question_id']);packet=torch.load(rebuilt/f'packet{i}.pt',weights_only=True);new=load(r/f'hf{i}/result.json')
 comparison.append(dict(packet=i,selected_positions_match=historical['selected_position_sha']==packet['position_sha'],all_generated_tokens_match=historical['judger_receipt']['generated_ids']==new['tokens']))
save(r/'historical_token_comparison.json',comparison)
files={p.relative_to(r).as_posix():p.read_bytes() for p in r.rglob('*') if p.is_file() and not p.is_symlink()};files['manifest.json']=json.dumps({n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in files.items()}).encode()
class Writer:
 def __init__(self):self.h=hashlib.sha256();self.size=0
 def write(self,b):self.h.update(b);self.size+=len(b);return sys.stdout.buffer.write(b)
 def flush(self):sys.stdout.buffer.flush()
w=Writer()
with tarfile.open(fileobj=w,mode='w|gz') as t:
 for n,v in files.items():m=tarfile.TarInfo(n);m.size=len(v);t.addfile(m,io.BytesIO(v))
w.flush();save(r/'archive_stream.json',dict(sha256=w.h.hexdigest(),bytes=w.size))
'''.replace('REMOTE',repr(a['out']))
 raw=call(script,600);digest=hashlib.sha256(raw).hexdigest();receipt=json.loads(call('from pathlib import Path\nprint((Path('+repr(a['out'])+')/"archive_stream.json").read_text())'));assert receipt==dict(sha256=digest,bytes=len(raw));(PRIVATE/('full_'+digest[:12]+'.tar.gz')).write_bytes(raw);dest=PRIVATE/('verified_'+digest[:12]);dest.mkdir(exist_ok=False)
 with tarfile.open(fileobj=io.BytesIO(gzip.decompress(raw))) as t:
  members=t.getmembers();names=[m.name for m in members];manifest=json.load(t.extractfile('manifest.json'));assert len(names)==len(set(names)) and set(names)==set(manifest)|{'manifest.json'}
  for m in members:
   n=PurePosixPath(m.name);assert m.isfile() and not n.is_absolute() and '..' not in n.parts and ':' not in m.name and '\\' not in m.name
   v=t.extractfile(m).read();p=dest.joinpath(*n.parts);assert p.resolve().is_relative_to(dest.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
   if m.name in manifest:assert len(v)==manifest[m.name]['bytes'] and hashlib.sha256(v).hexdigest()==manifest[m.name]['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
 for n in ['result.json','cost_verified.json','historical_token_comparison.json']:(PRIVATE/n).write_bytes((dest/n).read_bytes())
 proof=dict(passed=True,archive_sha256=digest,bytes=len(raw),members=len(names),remote_stream_sha=True,gzip_crc=True,exact_members=True,per_member_sha=True,safe_full_extract_readback_sha=True);b=(json.dumps(proof,indent=2)+'\n').encode();(PRIVATE/'archive_verified.json').write_bytes(b);call('from pathlib import Path\nimport base64\nPath('+repr(a['out']+'/archive_verified.json')+').write_bytes(base64.b64decode('+repr(base64.b64encode(b).decode())+'))');print(json.dumps(dict(archive=proof,cost=json.loads((PRIVATE/'cost_verified.json').read_text()),historical=json.loads((PRIVATE/'historical_token_comparison.json').read_text()))))
if __name__=='__main__':main()
