"""Synthetic, separately metered engineering proxy. Never edits frozen runs."""
import contextlib,gc,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025')
BASE=ROOT/'evidence/gpqa_repair_8096_20261028T161254Z/bundle/runtime'
RUN=ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z'
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(RUN),str(BASE)]
from runtime_common import Ledger,append,sha
def trace(event,**kw):append(OUT/'trace.jsonl',dict(unix=time.time(),event=event,**kw))
def main():
 import torch
 from transformers import AutoModelForCausalLM
 from stage_ops import observe
 config=dict(experiment_id=OUT.name,budget=dict(ledger_path=str(ROOT/'cumulative_budget.json'),cumulative_position_limit=115000000))
 ledger=Ledger(config,trace,'synthetic_throughput_proxy')
 lock=json.loads((BASE/'model_lock.json').read_text())
 for n,e in lock['files'].items():assert sha(Path(lock['path'])/n)==e['sha256']
 torch.set_num_threads(2)
 model=AutoModelForCausalLM.from_pretrained(lock['path'],local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval()
 assert all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters())
 ready=dict(pid=os.getpid(),uid=os.getuid(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),weights_sha_verified=15)
 (OUT/'ready.json').write_text(json.dumps(ready))
 records=[];charged=0
 for warmup,repeats,steps in [(True,1,16),(False,2,128)]:
  for rep in range(repeats):
   combos=[(b,a) for b in [1,4,8] for a in [False,True]]
   if rep%2:combos.reverse()
   for batch,audited in combos:
    bound=batch*(130+steps);assert charged+bound<=20000
    rid=ledger.begin(bound,dict(batch=batch,audited=audited,warmup=warmup,repeat=rep))
    actual=0;forwards=0;failed=True
    try:
     ids=(torch.arange(130,device='cuda').reshape(1,-1)+100).repeat(batch,1);past=None;tokens=[]
     torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
     ctx=observe(model,0,latent=False) if audited else contextlib.nullcontext()
     with torch.inference_mode(),ctx:
      for step in range(steps):
       length=ids.shape[-1]
       out=model(input_ids=ids,past_key_values=past,use_cache=True,return_dict=True)
       actual+=batch*length;forwards+=1;past=out.past_key_values
       ids=out.logits[:,-1,:].argmax(-1,keepdim=True);tokens.append(ids)
     torch.cuda.synchronize();elapsed=time.perf_counter()-start
     result=dict(batch=batch,audited=audited,warmup=warmup,repeat=rep,output_tokens=batch*steps,elapsed_s=elapsed,total_tokens_s=batch*steps/elapsed,per_stream_tokens_s=steps/elapsed,peak_bytes=torch.cuda.max_memory_allocated(),first_sequence_sha=hashlib.sha256(torch.cat(tokens,dim=1)[0].cpu().numpy().tobytes()).hexdigest(),positions=actual)
     records.append(result);append(OUT/'measurements.jsonl',result);failed=False
    finally:
     ledger.settle(rid,actual,forwards,failed=failed);charged+=actual
    del out,past,ids,tokens;gc.collect();torch.cuda.empty_cache()
 (OUT/'result.json').write_text(json.dumps(dict(passed=True,engineering_only=True,ready=ready,records=records,charged_positions=charged),indent=2))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  (OUT/'failure.json').write_text(json.dumps(dict(error=repr(e),unix=time.time())));raise
