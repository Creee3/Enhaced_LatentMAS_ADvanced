"""Original HF generate, real verified MPS prefix, original 8192/EOS rules."""
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025');OLD=ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z';BASE=ROOT/'evidence/gpqa_repair_8096_20261028T161254Z/bundle/runtime'
OUT=Path(__file__).resolve().parent;sys.path[:0]=[str(OLD),str(BASE)]
from runtime_common import Ledger,append,sha
def main(index):
 import torch
 from transformers import AutoModelForCausalLM
 from stage_ops import make_cache,observe,pairs
 from pinned_core import install_position_fix
 torch.set_num_threads(1);launch=json.loads((OUT/'launch.json').read_text());work=OUT/f'hf{index}';work.mkdir()
 trace=lambda event,**kw:append(work/'trace.jsonl',dict(unix=time.time(),event=event,**kw))
 lock=json.loads((BASE/'model_lock.json').read_text())
 for n,v in lock['files'].items():assert sha(Path(lock['path'])/n)==v['sha256']
 model=AutoModelForCausalLM.from_pretrained(lock['path'],torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',local_files_only=True).eval();assert all(p.device.type=='cuda' and p.dtype==torch.bfloat16 for p in model.parameters())
 install_position_fix(model)
 with torch.inference_mode():target=float(model.get_input_embeddings().weight.detach().to('cpu',dtype=torch.float32).norm(dim=1).mean())
 assert abs(target-1.3758320808410645)<1e-6
 packet=torch.load(Path(launch['rebuilt'])/f'packet{index}.pt',weights_only=True);original=tuple(tuple(t.to('cuda') for t in pair) for pair in packet['kv']);past=make_cache(original);ids=packet['ids'].to('cuda');n=ids.shape[-1]
 assert sum(t.numel()*t.element_size() for p in original for t in p)==19169280 and packet['absolute']+n+8192<=32768
 ledger=Ledger(dict(experiment_id=OUT.name,budget=dict(ledger_path=str(ROOT/'cumulative_budget.json'),cumulative_position_limit=115000000)),trace,'native_HF_cap8192_smoke')
 (work/'ready.json').write_text(json.dumps(dict(pid=os.getpid(),uid=os.getuid(),target_norm=target,bf16_cuda=True,parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),model_files_verified=15)))
 started=time.monotonic()
 while not (OUT/'go').exists():
  if time.monotonic()-started>300:raise RuntimeError('HF start barrier timeout')
  time.sleep(.05)
 rid=ledger.begin(n+8192,dict(packet=index,receiver='Judger',cap=8192));actual=[0,0,time.monotonic()]
 def post(mod,args,kw,value):
  actual[0]+=kw['input_ids'].shape[-1];actual[1]+=1
  if time.monotonic()-actual[2]>=1:
   trace('native_forward_progress',reservation_id=rid,completed_forwards=actual[1],completed_positions=actual[0]);actual[2]=time.monotonic()
 handle=model.register_forward_hook(post,with_kwargs=True);failed=True;torch.cuda.set_rng_state(packet['rng']);torch.cuda.synchronize();start=time.perf_counter()
 try:
  with torch.inference_mode(),observe(model,packet['absolute'],latent=False,prefix=pairs(past)) as proof:
   value=model.generate(input_ids=ids,attention_mask=torch.ones((1,130+n),dtype=torch.long,device='cuda'),past_key_values=past,cache_position=torch.arange(130,130+n,device='cuda'),max_new_tokens=8192,temperature=.6,top_p=.95,top_k=20,do_sample=True,pad_token_id=model.generation_config.pad_token_id,return_dict_in_generate=True,output_scores=False)
  torch.cuda.synchronize();elapsed=time.perf_counter()-start;tokens=value.sequences[0,n:].cpu().tolist();eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
  reference=json.loads((Path(launch['parent'])/'serial/done.json').read_text())['results'][0]['records'][index]['tokens'][0]
  assert tokens[:128]==reference[:128] and actual[0]==proof['positions'] and proof['sdpa_checks']==actual[1]*36
  result=dict(passed=True,packet=index,generated_tokens=len(tokens),natural_eos=tokens[-1] in eos,capped=len(tokens)==8192 and tokens[-1] not in eos,seconds=elapsed,first128_match=True,token_sha=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),actual_positions=actual[0],sdpa_checks=proof['sdpa_checks'],tokens=tokens)
  assert result['natural_eos'] or result['capped'];(work/'result.json').write_text(json.dumps(result));failed=False
 finally:handle.remove();ledger.settle(rid,actual[0],actual[1],failed=failed)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);a=ap.parse_args()
 try:main(a.index)
 except BaseException as e:(OUT/f'failure_hf{a.index}.json').write_text(json.dumps(dict(error=repr(e),pid=os.getpid())));raise
