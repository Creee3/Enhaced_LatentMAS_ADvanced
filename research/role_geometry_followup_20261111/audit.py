"""Exact diagnostic coverage, source/native positions, shapes and RID billing."""
import json,sys
from pathlib import Path
from common import HERE,ROOT,load,save,sha,verify,records,identity,rows
def audit(phase):
 import numpy as np
 source=verify();plan=load(HERE/'plan.json');q=load(HERE/'queue.json');pre=load(HERE/'preflight.json');items=load(HERE/'questions.json');want=set(plan['engineering_indices']) if phase=='engineering' else set(range(120));rr=[r for r in records() if r['index'] in want]
 assert len(rr)==len(want) and {r['index'] for r in rr}==want
 directories={r['index']:p.parent for p in (HERE/'attempts').glob('*/results.jsonl') for r in rows(p)};rids=[];positions=0;files=0;branch_cells=0
 for row in rr:
  index=row['index'];out=directories[index];ready=load(out/'model_ready.json');assert ready['weight_files']==15 and ready['bf16_cuda'] and ready['identity_norm'] and ready['parameter_bytes']==16381470720
  assert row['run_lock_sha256']==source['run_lock_sha256'] and row['no_judger_or_final_answers'] and len(row['receipts'])==21 and len(row['diagnostic_files'])==6
  assert row['actual_positions']==pre['proofs'][index]['upper_positions'];positions+=row['actual_positions'];rids+=row['request_ids']
  raw=rows(out/'raw/requests_raw.jsonl');settled={x['reservation_id']:x for x in rows(out/'trace.jsonl') if x['event']=='request_settled'}
  inputs={x['reservation_id']:x for x in raw if x['event']=='request_input'};outputs={x['reservation_id']:x for x in raw if x['event']=='request_output'}
  assert len([x for x in rows(out/'trace.jsonl') if x['event']=='model_file_verified'])==15
  for rec in row['receipts']:
   rid=rec['reservation_id'];assert not rec['failed'] and rec['proof']['forwards']==rec['native_forward_calls']==11 and rec['proof']['sdpa_checks']==396
   assert rec['actual_positions']==rec['prompt_len']+10==rec['proof']['positions'];assert rec['original_before']==rec['initial_past_positions']
   assert inputs[rid]['bound']==rec['actual_positions']==outputs[rid]['actual_positions']==settled[rid]['actual_positions']
   assert not settled[rid]['failed'] and settled[rid]['independent_unknown']==0 and outputs[rid]['generated_ids']==[]
   ids=inputs[rid]['input']['input_ids'];assert len(ids)==rec['prompt_len'];import hashlib
   h=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest();expected=pre['proofs'][index]
   assert h in {x['token_sha'] for x in expected['native']+expected['branches']}
  for f in row['diagnostic_files']:
   path=out/f['file'];assert sha(path)==f['sha256'] and path.stat().st_size==f['bytes'];files+=1
   with np.load(path,allow_pickle=False) as z:
    for name in z.files:
     if z[name].dtype.kind=='f':assert np.isfinite(z[name]).all(),name
    meta=json.loads(str(z['metadata_json']));assert meta['index']==index
    if 'branches' in meta:
     assert z['prompt_kv_bf16_bits'].shape==(6,36,2,8,32,128) and z['latent_kv_bf16_bits'].shape==(6,36,2,8,10,128)
     assert z['prompt_kv_bf16_bits'].dtype==np.uint16 and z['latent_kv_bf16_bits'].dtype==np.uint16
     assert len(meta['branches'])==6 and len({b['fork_rng_sha'] for b in meta['branches']})==1
     assert len({tuple(b['prompt_positions']) for b in meta['branches']})==1 and len({b['question_token_sha'] for b in meta['branches']})==1
     assert z['pair_same_role'].sum()==3 and len(z['pair_indices'])==15;branch_cells+=6
    else:
     pp=z['positions'];assert len(pp)==3 and pp.shape[1]>=2 and len(set(pp.flatten().tolist()))==pp.size
     assert z['window_kv_bf16_bits'].shape==(3,36,2,8,pp.shape[1],128)
 assert len(rids)==len(set(rids))==len(want)*21 and files==len(want)*6 and branch_cells==len(want)*18
 b=load(ROOT/'cumulative_budget.json');owned=[rid for rid,x in b['active_reservations'].items() if x.get('metadata',{}).get('experiment')==HERE.name];assert not owned
 if phase=='formal':
  assert all(t['state']=='complete' for t in q['tasks']);assert load(HERE/'management/finished.json')['all_complete']
  for w in load(HERE/'management/workers.json'):assert identity(w['pid']) is None
  assert identity(load(HERE/'management/identity.json')['identity']['pid']) is None
  for path in (HERE/'management').glob('mps_*_stop.json'):assert not load(path)['owned_after']
 result=dict(passed=True,phase=phase,cases=len(want),keys=len(want),unique_rids=len(rids),actual_positions=positions,files=files,branch_cells=branch_cells,own_reservations=0,run_lock_sha256=source['run_lock_sha256'],no_accuracy_scored=True)
 save(HERE/(phase+'_audit_verified.json'),result);return result
if __name__=='__main__':print(json.dumps(audit(sys.argv[1])))
