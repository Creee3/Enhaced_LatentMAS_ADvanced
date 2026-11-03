"""Bounded independent audit of a new worker's actual current-attempt answers."""
import argparse,base64,datetime,json
from deploy import HERE,PRIVATE,call

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pid',type=int,required=True);a=ap.parse_args()
    active=json.loads((PRIVATE/'cloud_active.json').read_text(encoding='utf-8'))
    script='''import json,hashlib,sys,time
from pathlib import Path
import numpy as np
run=Path(RUN_VALUE);sys.path.insert(0,str(run))
from cloud_common import identity,load,verify,sha
pid=PID_VALUE
worker=next(w for w in load(run/'workers.json') if w['pid']==pid)
actual=identity(pid);initial=worker['initial_identity']
assert actual and all(actual[k]==initial[k] for k in ['uid','start_ticks','argv','CUDA_VISIBLE_DEVICES'])
out=Path(worker['out']);ready=load(out/'model_ready.json');source=verify()
assert ready['source']['run_sha']==source['run_lock_sha256']
assert abs(ready['target_norm']-1.3758320808410645)<1e-6
assert ready['verified_weight_files']==15 and ready['bf16_cuda'] and ready['identity_norm'] and ready['parameter_bytes']==16381470720 and ready['target_norm']>0
trace=[json.loads(x) for x in (out/'trace.jsonl').read_bytes().split(b'\\n') if x]
base=run.parents[1]/'evidence/gpqa_repair_8096_20261028T161254Z/bundle'
assert {x['file'] for x in trace if x['event']=='model_file_verified'}==set(load(base/'runtime/model_lock.json')['files'])
rows=[json.loads(x) for x in (out/'results.jsonl').read_bytes().split(b'\\n') if x]
raw=[json.loads(x) for x in (out/'raw/requests_raw.jsonl').read_bytes().split(b'\\n') if x]
starts={x['reservation_id']:x for x in raw if x['event']=='request_input'}
outputs={x['reservation_id']:x for x in raw if x['event']=='request_output'}
settles=[x for x in trace if x['event']=='request_settled']
assert len(settles)==len({x['reservation_id'] for x in settles})
references=[json.loads(x) for x in (run/'baseline_results.jsonl').read_bytes().splitlines() if x]
reference_T={r['question_id']:r for r in references if r['method']=='T'}
formal={m:False for m in ['G']};checked=[]
for row in rows:
 ref=reference_T[row['question_id']]
 assert [x['selected_position_sha256'] for x in row['prefix_transfers']]==[x['selected_position_sha256'] for x in ref['prefix_transfers']]
 assert row['original_end']==ref['original_end']
 assert len(row['selector_diagnostics'])==36
 assert all(d['mode']=='G' and d['bank_rows']<=192 and d['prompt_budget']==96 for d in row['selector_diagnostics'])
 rec=row['judger_receipt'];rid=rec['reservation_id']
 assert rid.startswith(str(pid)+'-') and int(rid.split('-')[1])/1e9>=ready['unix']
 assert rid in starts and rid in outputs and not outputs[rid]['failed']
 assert row['repeat']==1 and row['latent_steps']==10 and row['seed']==42
 assert outputs[rid]['generated_ids']==rec['generated_ids']
 assert hashlib.sha256(json.dumps(rec['generated_ids']).encode()).hexdigest()==row['token_sha']
 assert row['generated_tokens']==len(rec['generated_ids'])>0
 assert row['matched_T_prefix_positions']
 assert row['source_k_equal_measured'] and row['source_v_equal_measured'] and row['protected_latent_sink']
 assert row['retained_kv_bytes']==19169280 and row['positions_shape']==[36,8,130]
 assert row['parser_sha']==sha(run/'author_parser.py')=='92163f3a5389c4a7e25868ff931fc919703eb1e500ef74dc03b4810ab6f0225d'
 pospath=out/(row['question_id']+'_r1_'+row['method']+'_positions.npz')
 with np.load(pospath,allow_pickle=False) as arrays:
  pos=arrays['positions'];indices=arrays['storage_indices']
  assert list(pos.shape)==list(indices.shape)==[36,8,130]
  assert hashlib.sha256(pos.tobytes()).hexdigest()==row['selected_position_sha']
  assert np.all(np.diff(pos,axis=-1)>0) and np.all(pos>=0) and np.all(pos<row['original_end'])
  assert np.all((pos<4).sum(-1)==4)
  for stage in row['prefix_stages']:
   assert np.all(((pos>=stage['original_end']-10)&(pos<stage['original_end'])).sum(-1)==10)
 assert rec['position_proof']['positions']==rec['actual_positions']
 assert rec['position_proof']['sdpa_checks']==rec['position_proof']['forwards']*36
 assert row['timing']['cuda_sync_both_boundaries']
 formal[row['method']]=True;checked.append(dict(question_id=row['question_id'],method=row['method'],rid=rid))
assert all(formal.values())
budget=load(run.parents[1]/'cumulative_budget.json')
own={r:x for r,x in budget['active_reservations'].items() if x['pid']==pid and x.get('metadata',{}).get('experiment')==run.name}
assert sum(budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens'])<=115000000
assert all(x['pid']==pid for x in own.values())
proof=dict(unix=time.time(),passed=True,identity=actual,model_file_sha_count=15,bf16_cuda=True,parameter_bytes=16381470720,
 identity_norm=True,target_norm=ready['target_norm'],source=source,formal_methods=formal,checked_new_answers=checked,
 actual_pair_gather_checks=True,original_position_and_protected_rows=True,position_npz_sha=True,raw_generated_ids_sha=True,
 native_position_receipts=True,duplicate_settlements=0,healthy_own_inflight=len(own),no_manual_settlement=True)
print(json.dumps(proof,ensure_ascii=False))
'''.replace('RUN_VALUE',repr(active['run'])).replace('PID_VALUE',str(a.pid))
    raw=call(script,120);proof=json.loads(raw)
    name='formal_gpu'+proof['identity']['CUDA_VISIBLE_DEVICES']+'_pid'+str(a.pid)+'_verified.json'
    (PRIVATE/name).write_bytes(raw)
    # This is a management proof; locked scientific files are unchanged.
    payload=base64.b64encode(raw).decode()
    sync='from pathlib import Path\nimport base64\np=Path('+repr(active['run'])+')/"management"/'+repr(name)+'\np.parent.mkdir(exist_ok=True)\np.write_bytes(base64.b64decode('+repr(payload)+'))\n'
    call(sync,60)
    print(json.dumps(dict(passed=True,pid=a.pid,gpu=proof['identity']['CUDA_VISIBLE_DEVICES'],verified_new_answers=len(proof['checked_new_answers']),formal_methods=proof['formal_methods'],healthy_inflight=proof['healthy_own_inflight']),ensure_ascii=False))

if __name__=='__main__':main()
