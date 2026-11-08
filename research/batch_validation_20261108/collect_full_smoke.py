"""Audit and archive the authorized full TRG smoke, without scoring answers."""
import gzip,hashlib,io,json,sys,tarfile
from pathlib import Path,PurePosixPath
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'TRG225_a100_20261106'))
from capture_current import call
PRIVATE=Path('D:/liuxin_workspace/ICML2027/advanced_MOC/experiments/batch_validation_20261108')

def main():
 a=json.loads((PRIVATE/'smoke_launch.json').read_text());remote=a['prepared_run']
 script=r'''
from pathlib import Path
import sys,json,subprocess,hashlib,tarfile,io
r=Path(REMOTE);root=r.parents[1];old=root/'evidence/TRG225_a100_s42_l10_20261106T155639Z';sys.path.insert(0,str(old));from cloud_common import identity,load,save,verify
assert (r/'completed.json').exists() and not list(r.glob('failure_case*.json'))
a=load(r/'launch_receipt.json');actual=identity(a['pid']);assert actual is None or actual['start_ticks']!=a['identity']['start_ticks']
assert not load(r/'mps_stop.json')['owned_after'] and load(r/'mps_stop.json')['returncode']==0
subprocess.run([sys.executable,str(r/'audit.py'),str(r)],capture_output=True,text=True,check=True)
rows=lambda p:[json.loads(b) for b in p.read_bytes().splitlines() if b]
prior=[]
for p in (old/'attempts').glob('*/results.jsonl'):prior+=rows(p)
prior+=rows(old/'reused_results.jsonl')
assert len({(x['question_id'],x['method']) for x in prior})==len(prior)==675
questions=load(old/'questions.json');comparison=[]
for i in range(4):
 for current in rows(r/f'case{i}/results.jsonl'):
  historical=next(x for x in prior if x['question_id']==questions[i]['question_id'] and x['method']==current['method'])
  comparisons=dict(packet=i,method=current['method'],all_generated_tokens_match=historical['judger_receipt']['generated_ids']==current['receipt']['generated_ids'],final_positions_match=historical['selected_position_sha']==current['position_sha'],prefix_transfer_positions_match=[x['selected_position_sha256'] for x in historical['prefix_transfers']]==[x['selected_position_sha256'] for x in current['prefix_transfers']],original_end_match=historical['original_end']==current['original_end'],prefix_stage_positions_match=[(x['original_before'],x['original_end'],x['prompt_len']) for x in historical['prefix_stages']]==[(x['original_before'],x['original_end'],x['prompt_len']) for x in current['prefix_stages']])
  comparison.append(comparisons)
save(r/'historical_trajectory_comparison.json',comparison)
audit=load(r/'audit_verified.json');b=load(root/'cumulative_budget.json');before=a['ledger_before'];delta=b['actual_tokens']-before['actual_tokens'];assert delta==audit['actual_positions'] and b['unknown_cost_upper_bound_tokens']==before['unknown_cost_upper_bound_tokens'] and b['reserved_tokens']==0
snapshot_checks={n:dict(bytes=(old/n).stat().st_size==meta['bytes'],sha256=hashlib.sha256((old/n).read_bytes()).hexdigest()==meta['sha256']) for n,meta in load(old/'run_lock.json')['snapshots'].items()};assert all(v['bytes'] and v['sha256'] for v in snapshot_checks.values())
trajectory_passed=all(all(x[k] for k in ['all_generated_tokens_match','final_positions_match','prefix_transfer_positions_match','original_end_match','prefix_stage_positions_match']) for x in comparison)
summary=dict(passed=trajectory_passed,structural_audit_passed=True,keys=audit['keys'],cases=4,methods=['T','R','G'],historical_complete_token_matches=sum(x['all_generated_tokens_match'] for x in comparison),historical_final_position_matches=sum(x['final_positions_match'] for x in comparison),historical_prefix_matches=sum(x['prefix_transfer_positions_match'] and x['prefix_stage_positions_match'] for x in comparison),actual_positions=audit['actual_positions'],unique_RIDs=audit['unique_RIDs'],unknown_added=0,own_reserved=0,accuracy_computed=False,source=verify(),snapshot_checks=snapshot_checks,ledger_after={k:b[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']},wall_seconds_including_staged_loading=((r/'completed.json').stat().st_mtime_ns-a['launched_ns'])/1e9)
timeline=[];per_case=[]
for i in range(4):
 d=r/f'case{i}';ready=(d/'model_ready.json').stat().st_mtime;done=(d/'done.json').stat().st_mtime;trace=rows(d/'trace.jsonl');timeline.extend([(ready,1),(done,-1)]);records=rows(d/'results.jsonl')
 per_case.append(dict(packet=i,model_initialization_seconds=ready-trace[0]['unix'],whole_case_seconds=done-trace[0]['unix'],generated_tokens={x['method']:len(x['receipt']['generated_ids']) for x in records},judger_seconds_excluding_selection={x['method']:x['seconds'] for x in records}))
active=0;peak=0
for _,delta in sorted(timeline):active+=delta;peak=max(active,peak)
summary.update(per_case=per_case,peak_simultaneous_model_ready_clients=peak,steady_four_client_throughput_benchmark=False)
save(r/'smoke_summary.json',summary)
source=dict(science_run=str(old),source_lock=load(old/'run_lock.json'),source=verify(),completed_science_archive_sha256='0e54558e885eeafb998ef90a5a70fd5d7d0864802ddb15c1165bf7657ee187c4',validation_archives=load(r/'validation_report.json')['evidence_archives'],pretrained_weights_copied=False)
save(r/'source_reference.json',source)
files={p.relative_to(r).as_posix():p.read_bytes() for p in r.rglob('*') if p.is_file() and not p.is_symlink() and p.name not in ['archive_stream.json','archive_verified.json']};files['manifest.json']=json.dumps({n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in files.items()}).encode()
class Writer:
 def __init__(self):self.h=hashlib.sha256();self.size=0
 def write(self,b):self.h.update(b);self.size+=len(b);return sys.stdout.buffer.write(b)
 def flush(self):sys.stdout.buffer.flush()
w=Writer()
with tarfile.open(fileobj=w,mode='w|gz') as t:
 for n,v in files.items():m=tarfile.TarInfo(n);m.size=len(v);t.addfile(m,io.BytesIO(v))
w.flush();save(r/'archive_stream.json',dict(sha256=w.h.hexdigest(),bytes=w.size))
'''.replace('REMOTE',repr(remote))
 raw=call(script,600);digest=hashlib.sha256(raw).hexdigest();receipt=json.loads(call('from pathlib import Path\nprint((Path('+repr(remote)+')/"archive_stream.json").read_text())'));assert receipt==dict(sha256=digest,bytes=len(raw))
 (PRIVATE/('full_TRG4_'+digest[:12]+'.tar.gz')).write_bytes(raw);dest=PRIVATE/('verified_TRG4_'+digest[:12]);dest.mkdir(exist_ok=False)
 with tarfile.open(fileobj=io.BytesIO(gzip.decompress(raw))) as t:
  members=t.getmembers();names=[m.name for m in members];manifest=json.load(t.extractfile('manifest.json'));assert len(names)==len(set(names)) and set(names)==set(manifest)|{'manifest.json'}
  for m in members:
   n=PurePosixPath(m.name);assert m.isfile() and not n.is_absolute() and '..' not in n.parts and ':' not in m.name and '\\' not in m.name
   v=t.extractfile(m).read();p=dest.joinpath(*n.parts);assert p.resolve().is_relative_to(dest.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
   if m.name in manifest:assert len(v)==manifest[m.name]['bytes'] and hashlib.sha256(v).hexdigest()==manifest[m.name]['sha256']==hashlib.sha256(p.read_bytes()).hexdigest()
 for n in ['audit_verified.json','historical_trajectory_comparison.json','smoke_summary.json']:(PRIVATE/n).write_bytes((dest/n).read_bytes())
 proof=dict(passed=True,archive_sha256=digest,bytes=len(raw),members=len(names),remote_stream_sha=True,gzip_crc=True,exact_members=True,per_member_sha=True,safe_full_extract_readback_sha=True);b=(json.dumps(proof,indent=2)+'\n').encode();(PRIVATE/'smoke_archive_verified.json').write_bytes(b)
 import base64
 call('from pathlib import Path\nimport base64\nPath('+repr(remote+'/archive_verified.json')+').write_bytes(base64.b64decode('+repr(base64.b64encode(b).decode())+'))')
 print(json.dumps(dict(summary=json.loads((PRIVATE/'smoke_summary.json').read_text()),archive=proof)))
if __name__=='__main__':main()
