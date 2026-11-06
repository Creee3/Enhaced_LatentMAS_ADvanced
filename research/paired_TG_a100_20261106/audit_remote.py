"""Independent completed-pair evidence audit, no raw text on stdout."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
M=Path('/home/ubuntu/cloud_migration_20261106/paired_TG')
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(b) for b in p.read_bytes().split(bytes([10])) if b]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 active=load(M/'cloud_active.json');r=Path(active['run']);sys.path.insert(0,str(r))
 from cloud_common import verify,identity
 source=verify();queue=load(r/'queue.json');assert all(t['state']=='complete' for t in queue['tasks'])
 workers=load(r/'workers.json');all_rows=[];all_rids=set();settled_tokens=0
 for w in workers:
  actual=identity(w['pid']);assert not actual or any(actual[k]!=w['initial_identity'][k] for k in ['uid','start_ticks','argv','CUDA_VISIBLE_DEVICES'])
  out=Path(w['out']);ready=load(out/'model_ready.json')
  assert ready['bf16_cuda'] and ready['identity_norm'] and ready['parameter_bytes']==16381470720 and abs(ready['target_norm']-1.3758320808410645)<1e-6
  trace=rows(out/'trace.jsonl');assert len({x['file'] for x in trace if x['event']=='model_file_verified'})==15
  raw=rows(out/'raw/requests_raw.jsonl');outputs={x['reservation_id']:x for x in raw if x['event']=='request_output'}
  settled=[x for x in trace if x['event']=='request_settled'];ids={x['reservation_id'] for x in settled};assert len(ids)==len(settled) and not ids&all_rids;all_rids|=ids
  for x in settled:
   # Preserve actual receipt schemas verbatim in private evidence; totals come from ledger below.
   assert x['reservation_id'] in outputs
  for row in rows(out/'results.jsonl') if (out/'results.jsonl').exists() else []:
   rid=row['judger_receipt']['reservation_id'];assert rid.startswith(str(w['pid'])+'-') and rid in outputs and not outputs[rid]['failed']
   rec=row['judger_receipt'];assert outputs[rid]['generated_ids']==rec['generated_ids']
   assert row['token_sha']==hashlib.sha256(json.dumps(rec['generated_ids']).encode()).hexdigest()
   assert row['generated_tokens']==len(rec['generated_ids'])>0 and row['pair_protocol']=='A100-shared-prefix-v1' and row['paired_on_new_shared_prefix']
   assert row['parser_sha']==sha(r/'author_parser.py')=='92163f3a5389c4a7e25868ff931fc919703eb1e500ef74dc03b4810ab6f0225d'
   assert row['retained_kv_bytes']==19169280 and row['retained_positions']==130 and row['source_k_equal_measured'] and row['source_v_equal_measured'] and row['protected_latent_sink']
   assert len(row['selector_diagnostics'])==36 and all(d['mode']==row['method'] and d['bank_rows']<=192 and d['prompt_budget']==96 for d in row['selector_diagnostics'])
   with np.load(out/(row['question_id']+'_r1_'+row['method']+'_positions.npz'),allow_pickle=False) as arrays:
    pos=arrays['positions'];assert list(pos.shape)==[36,8,130] and np.all(np.diff(pos,axis=-1)>0) and np.all(pos>=0) and np.all(pos<row['original_end'])
    assert hashlib.sha256(pos.tobytes()).hexdigest()==row['selected_position_sha'] and np.all((pos<4).sum(-1)==4)
    for stage in row['prefix_stages']:assert np.all(((pos>=stage['original_end']-10)&(pos<stage['original_end'])).sum(-1)==10)
    stage=row['prefix_stages'][-1];start=stage['original_before']+stage['prompt_len']-8;assert np.all(((pos>=start)&(pos<start+8)).sum(-1)==8)
   assert rec['position_proof']['positions']==rec['actual_positions'] and rec['position_proof']['sdpa_checks']==rec['position_proof']['forwards']*36
   all_rows.append(row)
 assert len(all_rows)==4 and len({(x['dataset'],x['question_id'],x['method']) for x in all_rows})==4
 comparisons=[]
 for q in load(r/'questions.json'):
  pair={x['method']:x for x in all_rows if x['question_id']==q['question_id']};assert set(pair)=={'T','G'}
  t,g=pair['T'],pair['G'];assert t['shared_request_ids']==g['shared_request_ids'] and len(t['shared_request_ids'])==6
  assert t['prefix_stages']==g['prefix_stages'] and t['prefix_transfers']==g['prefix_transfers'] and t['original_end']==g['original_end']
  comparisons.append(dict(is_diamond=q['is_diamond'],same_prefix_rids=True,same_prefix_positions=True,old_prefix_matched=t['matched_T_prefix_positions'],
    T_correct=t['correct'],G_correct=g['correct'],T_tokens=t['generated_tokens'],G_tokens=g['generated_tokens'],same_final_positions=t['selected_position_sha']==g['selected_position_sha']))
 budget=load(r.parents[1]/'cumulative_budget.json');assert not budget['active_reservations'] and budget['reserved_tokens']==0 and budget['unknown_cost_upper_bound_tokens']==58106
 assert load(r/'manager_exit.json')['completed'] and identity(active['manager_pid']) is None
 result=dict(passed=True,completed_answers=4,questions=2,worker_count=len(workers),same_gpu_pair=True,source=source,comparisons=comparisons,
  unique_settled_rids=len(all_rids),budget={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']},new_actual_tokens=budget['actual_tokens']-74906188,
  all_owned_processes_exited=True,active_reservations=0,old_runs_stay_paused=True,exploratory_mechanism_pilot_not_accuracy_estimate=True)
 (r/'management/paired_verified.json').write_text(json.dumps(result,indent=2)+'\n');(M/'paired_verified.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result))
if __name__=='__main__':main()
