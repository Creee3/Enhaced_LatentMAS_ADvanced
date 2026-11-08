"""Prepared post-smoke structural/receipt audit; never computes accuracy."""
import hashlib,json,sys
from pathlib import Path
def main(run):
 import numpy as np
 ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025');SCI=ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z';sys.path.insert(0,str(SCI));from cloud_common import identity,load,verify
 rows=lambda p:[json.loads(b) for b in p.read_bytes().split(b'\n') if b]
 assert load(run/'completed.json')['passed'] and not load(run/'mps_stop.json')['owned_after']
 all_keys=set();all_rids=set();cost=0
 for w in load(run/'workers.json'):
  actual=identity(w['pid']);assert actual is None or actual['start_ticks']!=w['identity']['start_ticks']
  out=run/f"case{w['case']}";ready=load(out/'model_ready.json');assert ready['bf16_cuda'] and ready['identity_norm'] and ready['parameter_bytes']==16381470720 and abs(ready['target_norm']-1.3758320808410645)<1e-6
  trace=rows(out/'trace.jsonl');assert len({x['file'] for x in trace if x['event']=='model_file_verified'})==15
  outputs={x['reservation_id']:x for x in rows(out/'raw/requests_raw.jsonl') if x['event']=='request_output'};settled=[x for x in trace if x['event']=='request_settled'];ids={x['reservation_id'] for x in settled}
  assert len(ids)==len(settled) and set(outputs)==ids and not ids&all_rids;all_rids|=ids
  assert all(not x['failed'] and x['independent_unknown']==0 for x in settled);cost+=sum(x['actual_positions'] for x in settled)
  records=rows(out/'results.jsonl');assert len(records)==3 and {r['method'] for r in records}=={'T','R','G'}
  for r in records:
   k=(r['packet'],r['method']);assert k not in all_keys;all_keys.add(k);receipt=r['receipt'];assert outputs[receipt['reservation_id']]['generated_ids']==receipt['generated_ids'] and hashlib.sha256(json.dumps(receipt['generated_ids']).encode()).hexdigest()==r['token_sha']
   assert r['actual_kv_bytes']==19169280 and r['source_k_equal_measured'] and r['source_v_equal_measured'] and len(r['diagnostics'])==36
   with np.load(out/(r['method']+'_positions.npz'),allow_pickle=False) as f:
    pos=f['positions'];assert pos.shape==(36,8,130) and np.all(np.diff(pos,axis=-1)>0) and np.all(pos<r['original_end']) and np.all(pos>=0)
    assert hashlib.sha256(pos.tobytes()).hexdigest()==r['position_sha'] and np.all((pos<4).sum(-1)==4)
    for s in r['prefix_stages']:assert np.all(((pos>=s['original_end']-10)&(pos<s['original_end'])).sum(-1)==10)
    s=r['prefix_stages'][-1];start=s['original_before']+s['prompt_len']-8;assert np.all(((pos>=start)&(pos<start+8)).sum(-1)==8)
   proof=receipt['position_proof'];assert proof['positions']==receipt['actual_positions'] and proof['sdpa_checks']==proof['forwards']*36
   assert r['shared_request_ids']==records[0]['shared_request_ids'] and r['prefix_transfers']==records[0]['prefix_transfers']
 assert all_keys=={(i,m) for i in range(4) for m in ['T','R','G']}
 b=load(ROOT/'cumulative_budget.json');assert not any(v.get('metadata',{}).get('experiment')==run.name for v in b['active_reservations'].values())
 result=dict(passed=True,keys=12,unique_RIDs=len(all_rids),actual_positions=cost,own_reserved=0,source=verify(),accuracy_computed=False);(run/'audit_verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main(Path(sys.argv[1]))
