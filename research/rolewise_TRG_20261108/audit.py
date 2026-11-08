"""Independent structural and billing audit; never computes accuracy."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from common import HERE,ROOT,METHODS,load,save,rows,records,key,identity,verify,sha

def main(phase):
 source=verify();queue=load(HERE/'queue.json');indices=set(queue['engineering_indices']) if phase=='engineering' else set(range(576));items=load(HERE/'questions.json');wanted={(x['dataset'],x['question_id'],10,m) for x in items if x['index'] in indices for m in METHODS};allrows=records();rs=[x for x in allrows if x['index'] in indices]
 assert len({key(x) for x in allrows})==len(allrows) and {key(x) for x in rs}==wanted and len(rs)==len(wanted)
 assert all(queue['tasks'][i]['state']=='complete' and not queue['tasks'][i]['methods'] for i in indices)
 phases=['engineering'] if phase=='engineering' else ['engineering','formal']
 for stage in phases:
  folder=HERE/'management'/stage;meta=load(folder/'identity.json');live=identity(meta['identity']['pid']);assert live is None or live['start_ticks']!=meta['identity']['start_ticks']
  assert load(folder/'finished.json')['all_eligible_complete'] and not (folder/'failure.json').exists()
  stops=list(folder.glob('mps_*_stop.json'));assert stops
  assert all(load(p)['returncode']==0 and not load(p)['owned_after'] for p in stops)
  for worker in load(folder/'workers.json'):
   live=identity(worker['pid']);assert live is None or live['start_ticks']!=worker['identity']['start_ticks']
   out=Path(worker['out']);assert not (out/'failure.json').exists() and load(out/'exit.json')['normal']
   ready=load(out/'model_ready.json');assert ready['weight_files']==15 and ready['bf16_cuda'] and ready['parameter_bytes']==16381470720 and ready['identity_norm'] and abs(ready['target_norm']-1.3758320808410645)<1e-6
 all_rids=set();billing=0;unknown=0;selected_rids=set();request_owner={};output_by_rid={};row_out={}
 for out in (HERE/'attempts').iterdir():
  trace=rows(out/'trace.jsonl');started={x['reservation_id'] for x in trace if x['event']=='request_started'};settled=[x for x in trace if x['event']=='request_settled'];ids={x['reservation_id'] for x in settled};assert started==ids and len(ids)==len(settled) and not ids&all_rids
  assert len({x['file'] for x in trace if x['event']=='model_file_verified'})==15
  raw={x['reservation_id']:x for x in rows(out/'raw/requests_raw.jsonl') if x['event']=='request_output'};assert set(raw)==ids
  all_rids|=ids;billing+=sum(x['actual_positions'] for x in settled);unknown+=sum(x['independent_unknown'] for x in settled);assert all(not x['failed'] and x['independent_unknown']==0 for x in settled)
  request_owner.update({rid:str(out) for rid in ids});output_by_rid.update(raw)
  for row in rows(out/'results.jsonl'):row_out[key(row)]=out
 for row in rs:
  item=items[row['index']];assert (row['dataset'],row['question_id'],row['category'],row['is_diamond'])==(item['dataset'],item['question_id'],item['category'],item['is_diamond'])
  assert row['seed']==42 and row['source_lock_sha']==source['run_lock_sha256'] and len(row['request_ids'])==len(set(row['request_ids']))==7 and len(row['shared_request_ids'])==2
  assert set(row['request_ids'])<=all_rids;selected_rids.update(row['request_ids']);receipt=row['judger_receipt'];rid=receipt['reservation_id'];assert rid in row['request_ids'] and output_by_rid[rid]['generated_ids']==receipt['generated_ids']
  assert hashlib.sha256(json.dumps(receipt['generated_ids']).encode()).hexdigest()==row['token_sha'] and len(receipt['generated_ids'])==row['generated_tokens'] and 0<row['generated_tokens']<=8192
  proof=receipt['position_proof'];assert proof['positions']==receipt['actual_positions'] and proof['sdpa_checks']==36*proof['forwards']
  assert len(row['prefix_stages'])==len(row['transfers'])==len(row['position_files'])==3
  assert [t['handoff'] for t in row['transfers']]==[1,2,3]
  for hop,(transfer,file) in enumerate(zip(row['transfers'],row['position_files']),1):
   assert transfer['mode']==row['method'][0] and transfer['source_k_equal_measured'] and transfer['source_v_equal_measured'] and transfer['original_RoPE'] and transfer['receiver_self_not_relayed'] and transfer['prompt_kept']<=96
   expected=4+10*hop+transfer['prompt_kept'];assert expected==transfer['retained']
   path=row_out[key(row)]/file['file'];assert sha(path)==file['file_sha256']
   with np.load(path,allow_pickle=False) as p:
    pos=p['positions'];ix=p['storage_indices'];assert pos.shape==ix.shape==(36,8,expected) and np.all(np.diff(pos,axis=-1)>0) and np.all(pos>=0) and np.all(pos<transfer['original_end'])
    assert hashlib.sha256(pos.tobytes()).hexdigest()==file['position_sha']==transfer['position_sha'] and np.all((pos<4).sum(-1)==4)
    for stage in row['prefix_stages'][:hop]:assert np.all(((pos>=stage['original_end']-10)&(pos<stage['original_end'])).sum(-1)==10)
    stage=row['prefix_stages'][hop-1];end=stage['original_before']+stage['prompt_len'];lo=max(4,stage['original_before'],end-8);assert np.all(((pos>=lo)&(pos<end)).sum(-1)==end-lo)
   assert transfer['actual_kv_bytes']==expected*147456
  assert row['actual_cumulative_relay_bytes']==sum(row['actual_kv_bytes_by_handoff'])==sum(t['actual_kv_bytes'] for t in row['transfers'])
 for i in indices:
  trio=[x for x in rs if x['index']==i];assert len(trio)==3
  assert len({tuple(x['shared_request_ids']) for x in trio})==len({x['shared_planner_kv_sha'] for x in trio})==len({x['fork_rng_sha'] for x in trio})==1
  assert len(set(r for x in trio for r in x['request_ids']))==17
 if phase=='engineering':
  for dataset in ['gpqa','mmlu_pro','gsm8k']:
   assert any(x['parse_valid'] for x in rs if x['dataset']==dataset),'All engineering outputs in one task family lack a program-parseable answer; preserve evidence before formal rollout'
 if phase=='formal':assert selected_rids==all_rids and len(all_rids)==576*17
 else:assert selected_rids==all_rids and len(all_rids)==4*17
 budget=load(ROOT/'cumulative_budget.json');own=[(rid,v) for rid,v in budget['active_reservations'].items() if v.get('metadata',{}).get('experiment')==HERE.name];assert not own
 result=dict(passed=True,phase=phase,keys=len(rs),cases=len(indices),unique_RIDs=len(all_rids),actual_positions=billing,unknown_added=unknown,own_reserved=0,run_lock_sha256=source['run_lock_sha256'],source=source,accuracy_computed=False,all_owned_services_stopped=True)
 save(HERE/(phase+'_audit_verified.json'),result);print(json.dumps(result))
if __name__=='__main__':main(sys.argv[1])
