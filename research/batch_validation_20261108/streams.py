"""One shared model, independent B1 requests, optional CUDA-stream overlap."""
import gc,hashlib,json,os,sys,time
from pathlib import Path
ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025');RUN=ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z';BASE=ROOT/'evidence/gpqa_repair_8096_20261028T161254Z/bundle/runtime'
OUT=Path(__file__).resolve().parent;sys.path[:0]=[str(RUN),str(BASE)]
from runtime_common import Ledger,append,sha
def main():
 import torch,torch.nn.functional as F
 from transformers import AutoModelForCausalLM,TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper
 from stage_ops import make_cache
 parent=Path(json.loads((OUT/'launch.json').read_text())['parent']);packets=[torch.load(parent/f'packet{i}.pt',weights_only=True) for i in range(4)]
 reference={r['indices'][0]:r['tokens'][0] for r in json.loads((parent/'serial/done.json').read_text())['results'][0]['records']}
 lock=json.loads((BASE/'model_lock.json').read_text())
 for n,v in lock['files'].items():assert sha(Path(lock['path'])/n)==v['sha256']
 torch.set_num_threads(1);model=AutoModelForCausalLM.from_pretrained(lock['path'],torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',local_files_only=True).eval();assert all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters())
 trace=lambda event,**kw:append(OUT/'trace.jsonl',dict(unix=time.time(),event=event,**kw))
 ledger=Ledger(dict(experiment_id=OUT.name,budget=dict(ledger_path=str(ROOT/'cumulative_budget.json'),cumulative_position_limit=115000000)),trace,'shared_model_streams');charged=0
 processors=[TemperatureLogitsWarper(.6),TopKLogitsWarper(20),TopPLogitsWarper(.95)]
 eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
 (OUT/'ready.json').write_text(json.dumps(dict(weights_sha=15,parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()))))
 results=[]
 for mode in ['same_stream','four_streams']:
  for repeat,steps,horizons in [(-1,16,None),(0,128,None),(1,128,None),(2,64,[8,16,32,64])]:
   jobs=[];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();total_start=time.perf_counter()
   for i,p in enumerate(packets):
    stream=torch.cuda.Stream() if mode=='four_streams' else torch.cuda.default_stream();stream.wait_stream(torch.cuda.default_stream())
    with torch.cuda.stream(stream):
     original=tuple(tuple(t.to('cuda') for t in pair) for pair in p['kv']);g=torch.Generator(device='cuda');g.set_state(p['rng'])
     ids=p['ids'].to('cuda');n=ids.shape[-1];h=horizons[i] if horizons else steps
     assert charged+n+h<=50000
     rid=ledger.begin(n+h,dict(mode=mode,repeat=repeat,packet=i,forced_eos=bool(horizons)))
     jobs.append(dict(i=i,stream=stream,original=original,past=make_cache(original),g=g,ids=ids,mask=torch.ones((1,130+n),dtype=torch.long,device='cuda'),n=n,absolute=p['absolute'],rid=rid,actual=0,forwards=0,checks=0,tokens=[],done=False,horizon=h))
   current=[None];previous=F.scaled_dot_product_attention
   def audit(q,k,v,*aa,**kw):
    j=current[0];l=j['checks']%36;j['checks']+=1
    if j['forwards']<2:
     for got,want in zip([k,v],j['original'][l]):
      if got.shape[1]!=want.shape[1]:want=want.repeat_interleave(got.shape[1]//want.shape[1],1)
      assert torch.equal(got[...,:130,:],want)
    return previous(q,k,v,*aa,**kw)
   F.scaled_dot_product_attention=audit;failed=True
   try:
    torch.cuda.synchronize();decode_start=time.perf_counter()
    with torch.inference_mode():
     for step in range(steps):
      for j in jobs:
       if j['done']:continue
       current[0]=j
       with torch.cuda.stream(j['stream']):
        n=j['ids'].shape[-1];start=j['past'].get_seq_length();cp=torch.arange(start,start+n,device='cuda');orig=j['absolute'] if not step else j['absolute']+j['n']+step-1;pos=torch.arange(orig,orig+n,device='cuda')[None,:]
        value=model(input_ids=j['ids'],past_key_values=j['past'],cache_position=cp,position_ids=pos,attention_mask=j['mask'],use_cache=True,return_dict=True);j['actual']+=n;j['forwards']+=1;j['past']=value.past_key_values
        scores=value.logits[:,-1,:].float()
        for processor in processors:scores=processor(j['ids'],scores)
        ids=torch.multinomial(scores[0].softmax(-1),1,generator=j['g']).view(1,1)
        if horizons and step+1==j['horizon']:ids=torch.full_like(ids,min(eos))
        j['tokens'].append(ids);j['ids']=ids;j['mask']=torch.cat([j['mask'],torch.ones((1,1),dtype=j['mask'].dtype,device='cuda')],-1)
        if horizons:j['done']=int(ids[0,0]) in eos
    torch.cuda.synchronize();elapsed=time.perf_counter()-decode_start
    records=[]
    for j in jobs:
     assert j['checks']==36*j['forwards'];tok=torch.cat(j['tokens'],dim=1)[0].cpu().tolist()
     length=j['horizon']-1 if horizons else steps;assert not horizons or len(tok)==j['horizon']
     records.append(dict(packet=j['i'],length=len(tok),tokens=tok,match_reference=tok[:length]==reference[j['i']][:length]))
    failed=False
   finally:
    F.scaled_dot_product_attention=previous;torch.cuda.synchronize()
    for j in jobs:ledger.settle(j['rid'],j['actual'],j['forwards'],failed=failed);charged+=j['actual']
   result=dict(mode=mode,repeat=repeat,steps=steps,forced_eos=bool(horizons),decoder_seconds=elapsed,total_wall_seconds=time.perf_counter()-total_start,total_tokens_s=sum(r['length'] for r in records)/(time.perf_counter()-total_start),peak_bytes=torch.cuda.max_memory_allocated(),records=records)
   append(OUT/'measurements.jsonl',result);results.append(result);del jobs,value,scores;gc.collect();torch.cuda.empty_cache()
 (OUT/'result.json').write_text(json.dumps(dict(passed=True,engineering_only=True,results=results,charged_positions=charged),indent=2))
if __name__=='__main__':
 try:main()
 except BaseException as e:(OUT/'failure.json').write_text(json.dumps(dict(error=repr(e),unix=time.time())));raise
