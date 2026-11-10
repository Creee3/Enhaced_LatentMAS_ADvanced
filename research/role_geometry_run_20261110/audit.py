"""Exact keys, pinned sources, original KV positions, diagnostics, native RID costs."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from common import HERE,ROOT,METHODS,load,save,rows,records,key,identity,verify,sha

def audit(phase):
 source=verify();q=load(HERE/'queue.json');items=load(HERE/'questions.json');ix=set(q['engineering_indices']) if phase=='engineering' else set(range(120));rs=[x for x in records() if x['index'] in ix]
 wanted={(x['dataset'],x['question_id'],10,m) for x in items if x['index'] in ix for m in METHODS}
 assert len(rs)==len(wanted)==len({key(x) for x in rs}) and {key(x) for x in rs}==wanted
 assert all(q['tasks'][i]['state']=='complete' and not q['tasks'][i]['methods'] for i in ix)
 workers=load(HERE/'management/workers.json');out_by_key={};all_ids=set();raw_by_id={};billed=0;case_rids={};verified_files=set()
 for w in workers:
  out=Path(w['out']);actual=identity(w['pid']);live=actual is not None and actual['start_ticks']==w['identity']['start_ticks']
  assert not (out/'failure.json').exists();ready=load(out/'model_ready.json');assert ready['weight_files']==15 and ready['bf16_cuda'] and ready['parameter_bytes']==16381470720 and ready['identity_norm'] and abs(ready['target_norm']-1.3758320808410645)<1e-6
  if phase=='engineering' and live:assert load(out/'current.json')['phase']=='waiting_engineering'
  if phase!='engineering':assert not live and load(out/'exit.json')['normal']
  trace=rows(out/'trace.jsonl');started={x['reservation_id'] for x in trace if x['event']=='request_started'};settled=[x for x in trace if x['event']=='request_settled'];ids={x['reservation_id'] for x in settled};assert started==ids and len(ids)==len(settled) and not ids&all_ids
  assert len({x['file'] for x in trace if x['event']=='model_file_verified'})==15
  raw={x['reservation_id']:x for x in rows(out/'raw/requests_raw.jsonl') if x['event']=='request_output'};assert set(raw)==ids
  assert all(not x['failed'] and x['independent_unknown']==0 for x in settled);all_ids|=ids;raw_by_id.update(raw);billed+=sum(x['actual_positions'] for x in settled)
  for x in rows(out/'results.jsonl'):out_by_key[key(x)]=out
  for x in rows(out/'case_done.jsonl'):case_rids[x['index']]=set(x['common_diagnostic_request_ids'])
 selected=set()
 for row in rs:
  item=items[row['index']];assert all(row[k]==item[k] for k in ['dataset','dataset_group','question_id','category','is_diamond']) and row['seed']==42 and row['latent_steps']==10 and row['source_lock_sha']==source['run_lock_sha256']
  assert len(row['request_ids'])==len(set(row['request_ids']))==7 and len(row['shared_request_ids'])==2;selected.update(row['request_ids'])
  rec=row['judger_receipt'];assert raw_by_id[rec['reservation_id']]['generated_ids']==rec['generated_ids'] and hashlib.sha256(json.dumps(rec['generated_ids']).encode()).hexdigest()==row['token_sha'] and 0<row['generated_tokens']==len(rec['generated_ids'])<=8192
  assert rec['position_proof']['positions']==rec['actual_positions'] and rec['position_proof']['sdpa_checks']==36*rec['position_proof']['forwards']
  assert len(row['prefix_stages'])==len(row['transfers'])==len(row['position_files'])==len(row['receiver_fidelity'])==3 and len(row['diagnostic_files'])==9
  previous=0
  for hop,(t,f,stage) in enumerate(zip(row['transfers'],row['position_files'],row['prefix_stages']),1):
   assert t['handoff']==hop and t['method']==row['method'] and t['source_k_equal_measured'] and t['protected_latent_sink_equal_measured'] and t['original_RoPE']
   if row['method'] not in ['H-OBF','L-OBF']:assert t['source_v_equal_measured']
   else:assert t.get('pca_rank',t.get('rank',None)) in ([2] if row['method']=='H-OBF' else [4]) or t.get('obf_rank') in ([2] if row['method']=='H-OBF' else [4])
   path=out_by_key[key(row)]/f['file'];assert sha(path)==f['file_sha256']
   with np.load(path,allow_pickle=False) as z:
    pos,ind=z['positions'],z['storage_indices'];assert pos.shape==ind.shape==(36,8,t['retained']) and np.all(pos>=0) and np.all(pos<stage['original_end']) and np.all(np.diff(pos,axis=-1)>0)
    assert hashlib.sha256(pos.tobytes()).hexdigest()==f['position_sha']==t['position_sha']
    if row['method']=='Full':assert np.all(pos==np.arange(stage['original_end']))
    elif row['method']=='Gen':assert pos.shape[-1]==4+10*hop
    elif row['method'] in ['H','L','H-OBF','L-OBF']:
     assert pos.shape[-1]==previous+(4 if hop==1 else 0)+min(32,stage['prompt_len']-(4 if hop==1 else 0))+10
     if row['method'] in ['L','L-OBF']:assert np.all(pos==pos[:,:1,:])
    else:
     assert pos.shape[-1]==96+4+10*hop;lo=stage['original_before']+stage['prompt_len']-8;hi=stage['original_before']+stage['prompt_len'];assert np.all(((pos>=lo)&(pos<hi)).sum(-1)==8)
    assert np.all((pos<4).sum(-1)==4)
    for s in row['prefix_stages'][:hop]:assert np.all(((pos>=s['original_end']-10)&(pos<s['original_end'])).sum(-1)==10)
   previous=t['retained'];assert t['actual_kv_bytes']==previous*147456
  for f in row['diagnostic_files']+row['common_state_diagnostic_files']:
   path=out_by_key[key(row)]/f['file']
   if str(path) in verified_files:continue
   assert sha(path)==f['sha256'] and f['readback_verified'] and not f['raw_kv_retained']
   with np.load(path,allow_pickle=False) as z:
    assert 'metadata_json' in z.files
    for n in z.files:
     a=z[n];assert a.dtype.kind!='O'
     if a.dtype.kind in 'fc':assert np.isfinite(a).all(),(path.name,n)
   verified_files.add(str(path))
 for i in ix:
  group=[x for x in rs if x['index']==i];assert len(group)==12 and len({x['shared_planner_kv_sha'] for x in group})==len({x['fork_rng_sha'] for x in group})==len({tuple(x['shared_request_ids']) for x in group})==1
  assert len({rid for row in group for rid in row['request_ids']})==62 and len(case_rids[i])==22;selected|=case_rids[i]
 assert all_ids==selected and len(all_ids)==84*len(ix)
 budget=load(ROOT/'cumulative_budget.json');assert not any(x.get('metadata',{}).get('experiment')==HERE.name for x in budget['active_reservations'].values())
 if phase!='engineering':
  owner=load(HERE/'management/identity.json')['identity'];a=identity(owner['pid']);assert a is None or a['start_ticks']!=owner['start_ticks'];assert load(HERE/'management/finished.json')['all_complete'] and not (HERE/'management/failure.json').exists()
  stops=list((HERE/'management').glob('mps_*_stop.json'));assert stops and all(load(p)['returncode']==0 and not load(p)['owned_after'] for p in stops)
 result=dict(passed=True,phase=phase,cases=len(ix),keys=len(rs),unique_RIDs=len(all_ids),actual_positions=billed,unknown_added=0,own_reserved=0,run_lock_sha256=source['run_lock_sha256'],model_weight_sha_files_per_worker=15,diagnostic_files=len(verified_files),accuracy_computed=False,resident_gate=phase=='engineering',all_owned_services_stopped=phase!='engineering',unix=time.time())
 save(HERE/(phase+'_audit_verified.json'),result);print(json.dumps(result));return result
if __name__=='__main__':audit(sys.argv[1])
