"""Independent first-new-method worker verification; no answer semantics or accuracy."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from common import HERE,BASE,METHODS,load,save,sha,identity,verify

def stable(path):
 if not path.exists():return []
 return [json.loads(x) for x in path.read_bytes().split(b'\n')[:-1] if x]

def main(phase):
 source=verify();folder=HERE/'management'/phase;checks=[]
 if not (folder/'workers.json').exists():return
 previous=load(folder/'workers_verified.json') if (folder/'workers_verified.json').exists() else dict(workers=[])
 verified={w['pid']:w for w in previous['workers'] if w['all_methods_verified']}
 for w in load(folder/'workers.json'):
  if w['pid'] in verified:checks.append(verified[w['pid']]);continue
  out=Path(w['out'])
  if not (out/'model_ready.json').exists():continue
  live=identity(w['pid']);ready=load(out/'model_ready.json');trace=stable(out/'trace.jsonl')
  assert {x['file'] for x in trace if x['event']=='model_file_verified'}==set(load(BASE/'model_lock.json')['files'])
  assert ready['bf16_cuda'] and ready['parameter_bytes']==16381470720 and ready['identity_norm'] and abs(ready['target_norm']-1.3758320808410645)<1e-6
  rs=stable(out/'results.jsonl');sample={}
  for r in rs:sample.setdefault(r['method'],r)
  raw={x['reservation_id']:x for x in stable(out/'raw/requests_raw.jsonl') if x['event']=='request_output'}
  for row in sample.values():
   rec=row['judger_receipt'];assert raw[rec['reservation_id']]['generated_ids']==rec['generated_ids']
   assert hashlib.sha256(json.dumps(rec['generated_ids']).encode()).hexdigest()==row['token_sha']
   assert rec['position_proof']['sdpa_checks']==36*rec['position_proof']['forwards'] and rec['position_proof']['positions']==rec['actual_positions']
   previous_len=0
   for hop,(f,t,s) in enumerate(zip(row['position_files'],row['transfers'],row['prefix_stages']),1):
    path=out/f['file'];assert sha(path)==f['file_sha256']
    with np.load(path,allow_pickle=False) as pack:
     pos,ind=pack['positions'],pack['storage_indices'];assert pos.shape==ind.shape==(36,8,t['retained']) and np.all(np.diff(pos,axis=-1)>0)
     assert hashlib.sha256(pos.tobytes()).hexdigest()==f['position_sha']==t['position_sha'] and np.all(pos>=0) and np.all(pos<s['original_end'])
     assert t['source_k_equal_measured'] and t['source_v_equal_measured'] and t['original_RoPE'] and t['actual_kv_bytes']==pos.shape[-1]*147456
     assert np.all((pos<4).sum(-1)==4)
     for stage in row['prefix_stages'][:hop]:assert np.all(((pos>=stage['original_end']-10)&(pos<stage['original_end'])).sum(-1)==10)
     if row['method'] in ['H','L']:
      assert t['independent_author_selection_equal'] and t['sender_attention_steps']==10 and t['mandatory_recent']==0 and pos.shape[-1]==42*hop+4
      assert np.all(ind[...,:previous_len]==np.arange(previous_len))
      if row['method']=='L':assert np.all(pos==pos[:,:1,:])
     elif row['method']=='Full':assert np.all(pos==np.arange(s['original_end']))
     else:
      assert pos.shape[-1]==(42*hop+4 if row['method']=='receiver' else 96+10*hop+4)
      lo=s['original_before']+s['prompt_len']-8;hi=s['original_before']+s['prompt_len'];assert np.all(((pos>=lo)&(pos<hi)).sum(-1)==8)
    previous_len=t['retained']
  checks.append(dict(pid=w['pid'],identity=live or ready['identity'],gpu=w['gpu_index'],methods=sorted(sample),verified_new_rows=len(sample),all_methods_verified=set(sample)==set(METHODS),weights=15,bf16_cuda_identity_norm=True,raw_tokens_original_KV_RoPE_positions_bytes=True))
 proof=dict(passed=True,phase=phase,source=source,workers=checks,unix=time.time(),accuracy_computed=False);save(folder/'workers_verified.json',proof)
 print(json.dumps(dict(phase=phase,ready_checked=len(checks),formal_verified=sum(x['all_methods_verified'] for x in checks))))

if __name__=='__main__':main(__import__('sys').argv[1])
