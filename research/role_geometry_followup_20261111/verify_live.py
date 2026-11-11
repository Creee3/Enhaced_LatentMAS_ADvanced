"""Independent ready/finished-case inspection; no generation or accuracy scoring."""
import json,sys,hashlib,numpy as np
from common import HERE,ROOT,BASE,load,save,rows,identity,verify,sha
from measure import describe
def verify_live():
 import torch
 source=verify();checks=[];workers=load(HERE/'management/workers.json');lock=load(BASE/'model_lock.json');assert len(lock['files'])==15
 for w in workers:
  out=__import__('pathlib').Path(w['out']);live=identity(w['pid']);ready=out/'model_ready.json';result=dict(pid=w['pid'],live=live is not None,ready=ready.exists(),complete_cases=0,independent_sample_checks=0)
  if live is not None:assert all(live[k]==w['identity'][k] for k in ['uid','start_ticks','argv','CUDA_VISIBLE_DEVICES'])
  if ready.exists():
   r=load(ready);assert r['weight_files']==15 and r['bf16_cuda'] and r['identity_norm'] and r['parameter_bytes']==16381470720 and abs(r['target_norm']-1.3758320808410645)<1e-6
   verified=[x for x in rows(out/'trace.jsonl') if x['event']=='model_file_verified'];assert len(verified)==15 and {x['file'] for x in verified}==set(lock['files'])
  for row in rows(out/'results.jsonl'):
   assert len(row['receipts'])==21 and len(row['contexts'])==3;result['complete_cases']+=1
   for f in row['diagnostic_files']:
    path=out/f['file'];assert sha(path)==f['sha256']
    with np.load(path,allow_pickle=False) as z:
     if 'prompt_kv_bf16_bits' not in z:continue
     for kind in ['prompt','latent']:
      # One independently reconstructed full layer, both K/V and all heads;
      # stored native bits exactly determine the reported spectra.
      bits=z[kind+'_kv_bf16_bits'][:,0].copy();t=torch.from_numpy(bits).view(torch.bfloat16).float().numpy();m,b,rank=describe(t)
      np.testing.assert_allclose(m,z[kind+'_metrics'][:,0],rtol=1e-10,atol=1e-10);result['independent_sample_checks']+=1
   # Verify a bounded first complete case per worker, not all live progress.
   break
  checks.append(result)
 result=dict(passed=True,source=source,workers=checks,model_forward_calls=0,ready_workers=sum(x['ready'] for x in checks),independently_verified_workers=sum(x['independent_sample_checks']>0 for x in checks));save(HERE/'management/live_verified.json',result);return result
if __name__=='__main__':print(json.dumps(verify_live()))
