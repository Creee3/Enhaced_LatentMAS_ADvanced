"""Real per-request KV/RoPE/RNG batching engineering benchmark."""
import argparse,ast,contextlib,gc,hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025')
RUN=ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z'
BASE=ROOT/'evidence/gpqa_repair_8096_20261028T161254Z/bundle/runtime'
sys.path[:0]=[str(RUN),str(BASE)]
from runtime_common import Ledger,append,sha,seed_all

def prepare(out):
 import torch,numpy as np,torch.nn.functional as F
 import experiment_worker as ew
 from stage_ops import observe,pairs,compress,make_cache,clone_cache
 from selector_dispatch import select
 tree=ast.parse((RUN/'cloud_worker.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
 names=['args_for','timed','context','stage','probe','final_select']
 defs=[n for n in main.body if isinstance(n,ast.FunctionDef) and n.name in names]
 # The original functions are nested inside try.
 defs=[n for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name in names]
 assert len(defs)==len(names)
 ns=dict(globals(),**locals());exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),'frozen_worker_functions','exec'),ns)
 ew.args_for=ns['args_for'];runner=ew.Runner(str(out/'prefix'),torch.device('cuda:0'));engine=runner.engine;model=engine.model
 engine.config.update(experiment_id=out.name,request_timeout_seconds=1800);engine.phase='engineering_real_KV_prepare'
 ns.update(runner=runner,engine=engine,model=model)
 for i,item in enumerate(json.loads((RUN/'questions.json').read_text())[:4]):
  seed_all(42);ns['current']=dict(repeat=1);cache=None;mapping=None;absolute=0;stages=[];transfers=[]
  for k,role in enumerate(['planner','critic','refiner']):
   cache,rec,_=ns['stage'](item,role,cache,absolute);stages.append(rec);absolute=rec['original_end']
   if k<2:
    attn,_=ns['probe'](item,['critic','refiner'][k],cache,absolute)
    cache,mapping,transfer=compress(cache,mapping,role=role,steps=10,prompt_len=rec['prompt_len'],original_before=rec['original_before'],method='receiver_attention_sink_recent',attention=attn);transfers.append(transfer)
  captured,_=ns['probe'](item,'judger',cache,absolute,True)
  rng=torch.cuda.get_rng_state();compact,pos,ix,_,_=ns['final_select'](cache,mapping,stages,captured,'R')
  for l,(old,new) in enumerate(zip(pairs(cache),pairs(compact))):
   for a,b in zip(old,new):assert torch.equal(torch.gather(a,2,ix[l][None,:,:,None].expand_as(b)),b)
  assert bool(((pos<4).sum(-1)==4).all())
  for s in stages:assert bool((((pos>=s['original_end']-10)&(pos<s['original_end'])).sum(-1)==10).all())
  s=stages[-1];last=s['original_before']+s['prompt_len']-8;assert bool((((pos>=last)&(pos<last+8)).sum(-1)==8).all())
  assert sum(t.numel()*t.element_size() for p in pairs(compact) for t in p)==19169280
  _,ids,_=engine.prepare(ew.stage_prompt('judger',item['question'],ns['args_for'](item['dataset'],runner.model_name)))
  packet=dict(kv=tuple(tuple(t.detach().cpu() for t in p) for p in pairs(compact)),ids=ids.cpu(),absolute=absolute,rng=rng.cpu(),position=pos.cpu(),position_sha=hashlib.sha256(pos.cpu().numpy().tobytes()).hexdigest())
  torch.save(packet,out/f'packet{i}.pt')
  del cache,compact,captured;gc.collect();torch.cuda.empty_cache()
 (out/'prepared.json').write_text(json.dumps(dict(passed=True,count=4,functions_from_frozen_worker=names,science_lock_sha=sha(RUN/'run_lock.json'))))

def benchmark(out,mode,index):
 import torch,torch.nn.functional as F
 from transformers import AutoModelForCausalLM,TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper
 from stage_ops import make_cache
 torch.set_num_threads(1)
 lock=json.loads((BASE/'model_lock.json').read_text())
 for n,e in lock['files'].items():assert sha(Path(lock['path'])/n)==e['sha256']
 model=AutoModelForCausalLM.from_pretrained(lock['path'],local_files_only=True,torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa').eval()
 assert all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters())
 work=out/(f'process{index}' if mode=='process' else mode);work.mkdir(exist_ok=True)
 trace=lambda event,**kw:append(work/'trace.jsonl',dict(unix=time.time(),event=event,**kw))
 ledger=Ledger(dict(experiment_id=out.name,budget=dict(ledger_path=str(ROOT/'cumulative_budget.json'),cumulative_position_limit=115000000)),trace,'real_KV_decode_proxy')
 packets=[torch.load(out/f'packet{i}.pt',weights_only=True) for i in range(4)]
 warp=[TemperatureLogitsWarper(.6),TopKLogitsWarper(20),TopPLogitsWarper(.95)]
 def decode(indices,steps):
  jobs=[packets[i] for i in indices];b=len(jobs);lengths=[j['ids'].shape[-1] for j in jobs];length=max(lengths)
  ids=torch.zeros((b,length),dtype=torch.long,device='cuda');mask=torch.zeros_like(ids)
  for k,j in enumerate(jobs):ids[k,-lengths[k]:]=j['ids'].to('cuda');mask[k,-lengths[k]:]=1
  attention=torch.cat([torch.ones((b,130),dtype=torch.long,device='cuda'),mask],-1)
  original=[tuple(torch.cat([j['kv'][l][v] for j in jobs],0).to('cuda') for v in [0,1]) for l in range(36)]
  past=make_cache(original);generators=[]
  for j in jobs:g=torch.Generator(device='cuda');g.set_state(j['rng']);generators.append(g)
  prefixpos=mask.cumsum(-1)-1+torch.tensor([j['absolute'] for j in jobs],device='cuda')[:,None];prefixpos=prefixpos.masked_fill(mask==0,0)
  rids=[ledger.begin(length+steps,dict(mode=mode,packet=i,steps=steps,padding=length-lengths[k])) for k,i in enumerate(indices)]
  actual=0;forwards=0;failed=True;proof=[0];tokens=[];old_sdpa=F.scaled_dot_product_attention
  def audit(q,k,v,*aa,**kw):
   l=proof[0]%36;proof[0]+=1
   if forwards<2:
    for got,want in zip([k,v],original[l]):
     if got.shape[1]!=want.shape[1]:want=want.repeat_interleave(got.shape[1]//want.shape[1],1)
     assert torch.equal(got[...,:130,:],want)
   return old_sdpa(q,k,v,*aa,**kw)
  F.scaled_dot_product_attention=audit
  try:
   torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
   with torch.inference_mode():
    for step in range(steps):
     n=ids.shape[-1];storage=past.get_seq_length();cp=torch.arange(storage,storage+n,device='cuda')
     position=prefixpos if step==0 else torch.tensor([j['absolute']+lengths[k]+step-1 for k,j in enumerate(jobs)],device='cuda')[:,None]
     assert position.shape==(b,n)
     value=model(input_ids=ids,past_key_values=past,cache_position=cp,position_ids=position,attention_mask=attention,use_cache=True,return_dict=True)
     actual+=n;forwards+=1;past=value.past_key_values;logits=value.logits[:,-1,:].float()
     for processor in warp:logits=processor(ids,logits)
     ids=torch.stack([torch.multinomial(logits[k].softmax(-1),1,generator=generators[k]) for k in range(b)])
     tokens.append(ids);attention=torch.cat([attention,torch.ones((b,1),dtype=attention.dtype,device='cuda')],-1)
   torch.cuda.synchronize();elapsed=time.perf_counter()-start
   assert proof[0]==steps*36 and past.get_seq_length()==130+length+steps-1
   seq=torch.cat(tokens,dim=1).cpu();hashes=[hashlib.sha256(seq[k].numpy().tobytes()).hexdigest() for k in range(b)]
   failed=False
   return dict(indices=indices,steps=steps,seconds=elapsed,total_tokens_s=b*steps/elapsed,peak_bytes=torch.cuda.max_memory_allocated(),hashes=hashes,tokens=seq.tolist(),sdpa_checks=proof[0],positions_per_request=actual)
  finally:
   F.scaled_dot_product_attention=old_sdpa
   for rid in rids:ledger.settle(rid,actual,forwards,failed=failed)
 def run(steps):
  sets=[list(range(4))] if mode=='batch' else [[i] for i in ([index] if mode=='process' else range(4))]
  return [decode(s,steps) for s in sets]
 run(16);(work/'ready.json').write_text(json.dumps(dict(pid=os.getpid(),mode=mode,packet=index,weights_sha=15)))
 results=[]
 for repeat in range(2):
  (work/f'waiting{repeat}').touch()
  if mode=='process':
   start=time.monotonic()
   while not (out/f'go{repeat}').exists():
    if time.monotonic()-start>300:raise RuntimeError('barrier timeout')
    time.sleep(.01)
  torch.cuda.synchronize();start=time.time();records=run(128);end=time.time()
  result=dict(repeat=repeat,start=start,end=end,records=records);results.append(result);(work/f'result{repeat}.json').write_text(json.dumps(result))
 (work/'done.json').write_text(json.dumps(dict(passed=True,mode=mode,results=results)))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','serial','batch','process']);ap.add_argument('--index',type=int,default=0);a=ap.parse_args();out=Path(__file__).resolve().parent
 try:
  if a.mode=='prepare':prepare(out)
  else:benchmark(out,a.mode,a.index)
 except BaseException as e:
  (out/f'failure_{a.mode}_{a.index}.json').write_text(json.dumps(dict(error=repr(e),pid=os.getpid())));raise
