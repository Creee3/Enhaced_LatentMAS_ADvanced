"""Deterministic coverage, native tensor provenance, billing and process audit."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from common import HERE,ROOT,METHODS,load,save,rows,records,key,identity,verify,sha

def main(phase):
 source=verify();q=load(HERE/'queue.json');ix=set(q['engineering_indices']) if phase=='engineering' else set(range(576));items=load(HERE/'questions.json');rs=[x for x in records() if x['index'] in ix]
 wanted={(x['dataset'],x['question_id'],10,m) for x in items if x['index'] in ix for m in METHODS}
 assert len(rs)==len(wanted)==len({key(x) for x in rs}) and {key(x) for x in rs}==wanted
 assert all(q['tasks'][i]['state']=='complete' and not q['tasks'][i]['methods'] for i in ix)
 for phase_name in (['engineering'] if phase=='engineering' else ['engineering','formal']):
  folder=HERE/'management'/phase_name;owner=load(folder/'identity.json')['identity'];actual=identity(owner['pid']);assert actual is None or actual['start_ticks']!=owner['start_ticks']
  assert load(folder/'finished.json')['all_eligible_complete'] and not (folder/'failure.json').exists()
  stops=list(folder.glob('mps_*_stop.json'));assert stops and all(load(p)['returncode']==0 and not load(p)['owned_after'] for p in stops)
  for w in load(folder/'workers.json'):
   actual=identity(w['pid']);assert actual is None or actual['start_ticks']!=w['identity']['start_ticks'];out=Path(w['out']);ready=load(out/'model_ready.json')
   assert not (out/'failure.json').exists() and load(out/'exit.json')['normal']
   assert ready['weight_files']==15 and ready['bf16_cuda'] and ready['parameter_bytes']==16381470720 and ready['identity_norm'] and abs(ready['target_norm']-1.3758320808410645)<1e-6
 all_ids=set();raw_by_id={};row_out={};billing=0
 for out in (HERE/'attempts').iterdir():
  trace=rows(out/'trace.jsonl');began={x['reservation_id'] for x in trace if x['event']=='request_started'};settled=[x for x in trace if x['event']=='request_settled'];ids={x['reservation_id'] for x in settled}
  assert began==ids and len(ids)==len(settled) and not ids&all_ids
  assert len({x['file'] for x in trace if x['event']=='model_file_verified'})==15
  raw={x['reservation_id']:x for x in rows(out/'raw/requests_raw.jsonl') if x['event']=='request_output'};assert set(raw)==ids
  assert all(not x['failed'] and x['independent_unknown']==0 for x in settled);billing+=sum(x['actual_positions'] for x in settled);all_ids|=ids;raw_by_id.update(raw)
  for r in rows(out/'results.jsonl'):row_out[key(r)]=out
 selected=set()
 for row in rs:
  item=items[row['index']];assert all(row[k]==item[k] for k in ['dataset','question_id','category','is_diamond'])
  assert row['seed']==42 and row['latent_steps']==10 and row['source_lock_sha']==source['run_lock_sha256']
  assert len(row['request_ids'])==len(set(row['request_ids']))==(4 if row['method']=='Full' else 7) and len(row['shared_request_ids'])==1
  selected.update(row['request_ids']);rec=row['judger_receipt'];assert rec['reservation_id'] in row['request_ids'] and raw_by_id[rec['reservation_id']]['generated_ids']==rec['generated_ids']
  assert hashlib.sha256(json.dumps(rec['generated_ids']).encode()).hexdigest()==row['token_sha'] and len(rec['generated_ids'])==row['generated_tokens'] and 0<row['generated_tokens']<=8192
  assert rec['position_proof']['positions']==rec['actual_positions'] and rec['position_proof']['sdpa_checks']==36*rec['position_proof']['forwards']
  assert len(row['transfers'])==len(row['position_files'])==len(row['prefix_stages'])==3
  previous=0
  for hop,(transfer,file,stage) in enumerate(zip(row['transfers'],row['position_files'],row['prefix_stages']),1):
   assert transfer['handoff']==hop and transfer['method']==row['method'] and transfer['source_k_equal_measured'] and transfer['source_v_equal_measured'] and transfer['original_RoPE']
   path=row_out[key(row)]/file['file'];assert sha(path)==file['file_sha256']
   with np.load(path,allow_pickle=False) as p:
    pos,ind=p['positions'],p['storage_indices'];assert pos.shape==ind.shape==(36,8,transfer['retained'])
    assert np.all(pos>=0) and np.all(pos<stage['original_end']) and np.all(np.diff(pos,axis=-1)>0)
    assert hashlib.sha256(pos.tobytes()).hexdigest()==file['position_sha']==transfer['position_sha']
    if row['method']=='Full':assert np.all(pos==np.arange(stage['original_end']))
    else:
     sink=4 if hop==1 else 0;kept=min(32,stage['prompt_len']-sink)
     assert transfer['retained']==previous+sink+kept+10
     assert np.all(ind[...,:previous]==np.arange(previous))
     lo=stage['original_before']+max(sink,stage['prompt_len']-min(8,kept));hi=stage['original_before']+stage['prompt_len'];assert np.all(((pos>=lo)&(pos<hi)).sum(-1)==hi-lo)
    assert np.all((pos<4).sum(-1)==4)
    for s in row['prefix_stages'][:hop]:assert np.all(((pos>=s['original_end']-10)&(pos<s['original_end'])).sum(-1)==10)
   previous=transfer['retained'];assert transfer['actual_kv_bytes']==previous*147456
  assert row['actual_cumulative_relay_bytes']==sum(row['actual_kv_bytes_by_handoff'])==sum(t['actual_kv_bytes'] for t in row['transfers'])
 for i in ix:
  pair=[r for r in rs if r['index']==i];assert len(pair)==2 and len({r['shared_planner_kv_sha'] for r in pair})==len({r['fork_rng_sha'] for r in pair})==len({tuple(r['shared_request_ids']) for r in pair})==1
  assert len({rid for r in pair for rid in r['request_ids']})==10
 assert selected==all_ids and len(all_ids)==len(ix)*10
 if phase=='engineering':assert all(any(r['parse_valid'] for r in rs if r['dataset']==d) for d in ['gpqa','mmlu_pro','gsm8k'])
 budget=load(ROOT/'cumulative_budget.json');assert not [v for v in budget['active_reservations'].values() if v.get('metadata',{}).get('experiment')==HERE.name]
 result=dict(passed=True,phase=phase,keys=len(rs),cases=len(ix),unique_RIDs=len(all_ids),actual_positions=billing,unknown_added=0,own_reserved=0,run_lock_sha256=source['run_lock_sha256'],source=source,accuracy_computed=False,all_owned_services_stopped=True)
 save(HERE/(phase+'_audit_verified.json'),result);print(json.dumps(result))

if __name__=='__main__':main(sys.argv[1])
