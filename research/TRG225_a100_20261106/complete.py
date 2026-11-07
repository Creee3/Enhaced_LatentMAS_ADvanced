"""Completion audit and full archive; immutable scorer runs only after validation."""
import ast,base64,gzip,hashlib,io,json,tarfile
from pathlib import Path,PurePosixPath
from capture_current import PRIVATE,HERE,call,active

def put(run,name,data):
 (PRIVATE/name).write_bytes(data)
 call('from pathlib import Path\nimport base64\nPath('+repr(run+'/management/'+name)+').write_bytes(base64.b64decode('+repr(base64.b64encode(data).decode())+'))')

def main():
 a=active();run=a['run'];full=json.loads(max(PRIVATE.glob('inspect_*.json'),key=lambda p:p.stat().st_mtime).read_bytes())
 assert full['combined_unique']==675 and full['remaining']==0 and full['queue']['complete']==225
 assert not full['manager_identity'] and full['manager_exit']['completed'] and not full['own_reservations']
 assert all(not w['identity'] and w['exit']['completed'] and not w['failure'] for w in full['workers'])
 tree=ast.parse((HERE/'verify_worker.py').read_text());template=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('import pathlib,json,hashlib'))
 old="assert actual and all(actual[k]==w['initial_identity'][k] for k in ['uid','start_ticks','argv','CUDA_VISIBLE_DEVICES'])"
 assert old in template
 template=template.replace(old,"assert actual is None\nassert load(pathlib.Path(w['out'])/'exit.json')['completed']")
 proofs=[]
 for w in full['workers']:
  proof=json.loads(call(template.replace('RUN',repr(run)).replace('PID',str(w['pid']))));assert proof['passed'] and proof['formal'];proofs.append(proof)
 assert sum(p['checked_answers'] for p in proofs)==671
 audit=r'''
from pathlib import Path
import sys,json,hashlib
r=Path(RUN_PATH);sys.path.insert(0,str(r));from cloud_common import load,save,sha,verify
def rows(p):return [json.loads(b) for b in p.read_bytes().split(b'\n') if b]
allrows=rows(r/'reused_results.jsonl');seen=set();tokens=0
for w in load(r/'workers.json'):
 out=Path(w['out']);tr=rows(out/'trace.jsonl');raw=rows(out/'raw/requests_raw.jsonl')
 starts={x['reservation_id'] for x in tr if x['event']=='request_started'}
 settled=[x for x in tr if x['event']=='request_settled'];ids={x['reservation_id'] for x in settled}
 outputs={x['reservation_id'] for x in raw if x['event']=='request_output'}
 assert len(ids)==len(settled) and starts==ids==outputs and not ids&seen;seen|=ids
 assert all(not x['failed'] and x['independent_unknown']==0 for x in settled)
 tokens+=sum(x['actual_positions'] for x in settled);allrows+=rows(out/'results.jsonl')
questions=load(r/'questions.json');expected={(q['dataset'],q['question_id'],10,m) for q in questions for m in ['T','R','G']}
keys=[(x['dataset'],x['question_id'],x['latent_steps'],x['method']) for x in allrows];assert len(keys)==675 and len(set(keys))==675 and set(keys)==expected
for q in questions:
 group=[x for x in allrows if x['dataset']==q['dataset'] and x['question_id']==q['question_id']]
 if q['question_id'] not in {x['question_id'] for x in rows(r/'reused_results.jsonl')}:
  assert all(x['shared_request_ids']==group[0]['shared_request_ids'] and x['prefix_transfers']==group[0]['prefix_transfers'] for x in group)
b=load(r.parents[1]/'cumulative_budget.json');assert b['reserved_tokens']==0 and not b['active_reservations'] and b['unknown_cost_upper_bound_tokens']==58106
assert tokens==b['actual_tokens']-74917051
m=r/'management';lock=load(m/'gpu01_management_lock.json');assert all(sha(m/n)==v for n,v in lock['files'].items())
proof=dict(passed=True,answers=675,new_answers=671,reused=4,unique_RIDs=len(seen),new_actual_tokens=tokens,source=verify(),all_raw_trace_receipts_match=True,all_fresh_cases_shared_prefix=True,accuracy_computed=False)
save(m/'completion_evidence_verified.json',proof);print(json.dumps(proof))
'''.replace('RUN_PATH',repr(run))
 raw=call(audit);put(run,'completion_evidence_verified.json',raw)
 # Extend the previously validated completed-pilot collector, without another inspect.
 source=(HERE.parent/'paired_TG_a100_20261106/collect.py').read_text()
 script=next(n.value for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('import pathlib,json,hashlib,sys,tarfile,io'))
 marker="base=root/'evidence/gpqa_repair_8096_20261028T161254Z/bundle'"
 extra="""pre=json.loads((r/'launch_preflight.json').read_text())
for label,parent in [('reused_pilot',pathlib.Path(pre['pilot_run'])),('prior_A100_failure',root/'evidence/attention_G_a100_matched225_s42_l10_20261106T151824Z')]:
 for p in parent.rglob('*'):
  if p.is_file() and not p.is_symlink():files[label+'/'+p.relative_to(parent).as_posix()]=p.read_bytes()
"""
 assert marker in script;script=script.replace(marker,extra+marker).replace('RUN',repr(run))
 raw=call(script,600);digest=hashlib.sha256(raw).hexdigest()
 receipt=json.loads(call('from pathlib import Path\nprint((Path('+repr(run)+')/"management/archive_stream.json").read_text())'))
 assert receipt==dict(sha256=digest,bytes=len(raw));archive=PRIVATE/('completed_'+digest[:12]+'.tar.gz');archive.write_bytes(raw)
 dest=PRIVATE/('completed_verified_'+digest[:12]);dest.mkdir(exist_ok=False)
 with tarfile.open(fileobj=io.BytesIO(gzip.decompress(raw))) as tar:
  members=tar.getmembers();names=[m.name for m in members];manifest=json.load(tar.extractfile('manifest.json'))
  assert len(names)==len(set(names)) and set(names)==set(manifest)|{'manifest.json'}
  for m in members:
   name=PurePosixPath(m.name);assert m.isfile() and not name.is_absolute() and '..' not in name.parts and ':' not in m.name and '\\' not in m.name
   b=tar.extractfile(m).read();p=dest.joinpath(*name.parts);assert p.resolve().is_relative_to(dest.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
   if m.name in manifest:assert len(b)==manifest[m.name]['bytes'] and hashlib.sha256(b).hexdigest()==manifest[m.name]['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
 proof=dict(passed=True,archive_sha256=digest,bytes=len(raw),members=len(names),remote_stream_sha=True,gzip_crc=True,exact_members=True,per_member_sha=True,safe_full_extract_readback_sha=True)
 put(run,'completion_archive_verified.json',(json.dumps(proof,indent=2)+'\n').encode());print(json.dumps(proof),flush=True)
 scorer=HERE/'final_summary.py';payload=base64.b64encode(scorer.read_bytes()).decode()
 result=call('import base64,runpy,sys\nfrom pathlib import Path\nr=Path('+repr(run)+');p=r/"management/final_summary_frozen.py";p.write_bytes(base64.b64decode('+repr(payload)+'));sys.argv=[str(p),"--run",str(r)];runpy.run_path(str(p),run_name="__main__")')
 put(run,'final_summary.json',call('from pathlib import Path\nprint((Path('+repr(run)+')/"management/final_summary.json").read_text())'))
 print(result.decode())
if __name__=='__main__':main()
