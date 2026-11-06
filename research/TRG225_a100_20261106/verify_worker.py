"""Independent current-attempt structural audit, all3 labels needed for formal."""
import argparse,base64,json
from capture_current import active,call,PRIVATE
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pid',type=int,required=True);args=ap.parse_args();a=active()
 script='''import pathlib,json,hashlib,sys,numpy as np,time
r=pathlib.Path(RUN);sys.path.insert(0,str(r));from cloud_common import identity,load,verify,sha
w=next(w for w in load(r/'workers.json') if w['pid']==PID);actual=identity(PID)
assert actual and all(actual[k]==w['initial_identity'][k] for k in ['uid','start_ticks','argv','CUDA_VISIBLE_DEVICES'])
out=pathlib.Path(w['out']);ready=load(out/'model_ready.json');source=verify()
assert ready['source']['run_sha']==source['run_lock_sha256'] and ready['bf16_cuda'] and ready['identity_norm'] and ready['parameter_bytes']==16381470720 and abs(ready['target_norm']-1.3758320808410645)<1e-6
def rows(p):return [json.loads(b) for b in p.read_bytes().split(bytes([10])) if b] if p.exists() else []
trace=rows(out/'trace.jsonl');base=r.parents[1]/'evidence/gpqa_repair_8096_20261028T161254Z/bundle'
assert {x['file'] for x in trace if x['event']=='model_file_verified'}==set(load(base/'runtime/model_lock.json')['files'])
settled=[x for x in trace if x['event']=='request_settled'];assert len(settled)==len({x['reservation_id'] for x in settled})
raw=rows(out/'raw/requests_raw.jsonl');outputs={x['reservation_id']:x for x in raw if x['event']=='request_output'}
new=rows(out/'results.jsonl');checked=[];formal={m:False for m in ['T','R','G']}
for row in new:
 rec=row['judger_receipt'];rid=rec['reservation_id'];assert rid.startswith(str(PID)+'-') and rid in outputs and not outputs[rid]['failed']
 assert outputs[rid]['generated_ids']==rec['generated_ids'] and hashlib.sha256(json.dumps(rec['generated_ids']).encode()).hexdigest()==row['token_sha']
 assert row['paired_on_new_shared_prefix'] and row['retained_kv_bytes']==19169280 and row['source_k_equal_measured'] and row['source_v_equal_measured'] and row['protected_latent_sink']
 assert row['parser_sha']==sha(r/'author_parser.py')=='92163f3a5389c4a7e25868ff931fc919703eb1e500ef74dc03b4810ab6f0225d'
 assert len(row['selector_diagnostics'])==36 and all(d['mode']==row['method'] and d['bank_rows']<=192 and d['prompt_budget']==96 for d in row['selector_diagnostics'])
 with np.load(out/(row['question_id']+'_r1_'+row['method']+'_positions.npz'),allow_pickle=False) as f:
  pos=f['positions'];assert list(pos.shape)==[36,8,130] and np.all(np.diff(pos,axis=-1)>0) and np.all(pos>=0) and np.all(pos<row['original_end'])
  assert hashlib.sha256(pos.tobytes()).hexdigest()==row['selected_position_sha'] and np.all((pos<4).sum(-1)==4)
  for s in row['prefix_stages']:assert np.all(((pos>=s['original_end']-10)&(pos<s['original_end'])).sum(-1)==10)
  s=row['prefix_stages'][-1];start=s['original_before']+s['prompt_len']-8;assert np.all(((pos>=start)&(pos<start+8)).sum(-1)==8)
 assert rec['actual_positions']==rec['position_proof']['positions'] and rec['position_proof']['sdpa_checks']==rec['position_proof']['forwards']*36
 formal[row['method']]=True;checked.append(dict(method=row['method'],rid=rid))
for q in {x['question_id'] for x in new}:
 group=[x for x in new if x['question_id']==q]
 assert all(x['shared_request_ids']==group[0]['shared_request_ids'] and x['prefix_transfers']==group[0]['prefix_transfers'] for x in group)
pilot={x['question_id']:x for x in rows(r/'reused_results.jsonl') if x['method']=='T'}
for x in new:
 if x['question_id'] in pilot:assert x['prefix_transfers']==pilot[x['question_id']]['prefix_transfers'] or [t['selected_position_sha256'] for t in x['prefix_transfers']]==[t['selected_position_sha256'] for t in pilot[x['question_id']]['prefix_transfers']]
budget=load(r.parents[1]/'cumulative_budget.json');assert sum(budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens'])<=115000000
proof=dict(passed=True,formal=all(formal.values()),formal_methods=formal,checked_answers=len(new),identity=actual,model_file_sha_count=15,bf16_cuda=True,identity_norm=True,target_norm=ready['target_norm'],source=source,actual_KV_and_original_gather=True,positions_and_token_sha=True,own_inflight=sum(x['pid']==PID for x in budget['active_reservations'].values()),unix=time.time())
(r/'management'/('worker_'+str(PID)+'_verified.json')).write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''.replace('RUN',repr(a['run'])).replace('PID',str(args.pid))
 raw=call(script);proof=json.loads(raw);(PRIVATE/('worker_'+str(args.pid)+'_verified.json')).write_bytes(raw)
 print(json.dumps({k:proof[k] for k in ['passed','formal','formal_methods','checked_answers','identity','own_inflight']}))
if __name__=='__main__':main()
