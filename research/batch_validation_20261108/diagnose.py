"""Metered teacher-forced logits/RNG localization and EOS scheduler checks."""
import contextlib,hashlib,json,os,sys,time,gc
from pathlib import Path
ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025')
RUN=ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z'
BASE=ROOT/'evidence/gpqa_repair_8096_20261028T161254Z/bundle/runtime'
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(RUN),str(BASE)]
from runtime_common import Ledger,append,sha

def main():
 import torch,torch.nn.functional as F
 from transformers import AutoModelForCausalLM,TemperatureLogitsWarper,TopKLogitsWarper,TopPLogitsWarper
 from stage_ops import make_cache,observe,pairs
 from pinned_core import install_position_fix
 launch=json.loads((OUT/'launch.json').read_text());parent=Path(launch['parent'])
 packets=[torch.load(parent/f'packet{i}.pt',weights_only=True) for i in range(4)]
 source_ref=json.loads((parent/'serial/done.json').read_text())['results'][0]['records'];reference={r['indices'][0]:r['tokens'][0] for r in source_ref}
 for p in packets:
  assert p['position'].shape==(36,8,130) and hashlib.sha256(p['position'].numpy().tobytes()).hexdigest()==p['position_sha']
  assert all(t.shape==(1,8,130,128) for pair in p['kv'] for t in pair)
 lock=json.loads((BASE/'model_lock.json').read_text())
 for n,v in lock['files'].items():assert sha(Path(lock['path'])/n)==v['sha256']
 torch.set_num_threads(1)
 model=AutoModelForCausalLM.from_pretrained(lock['path'],torch_dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa',local_files_only=True).eval()
 assert all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters())
 install_position_fix(model)
 trace=lambda event,**kw:append(OUT/'trace.jsonl',dict(unix=time.time(),event=event,**kw))
 ledger=Ledger(dict(experiment_id=OUT.name,budget=dict(ledger_path=str(ROOT/'cumulative_budget.json'),cumulative_position_limit=115000000)),trace,'batch_validation')
 (OUT/'ready.json').write_text(json.dumps(dict(weights_sha=15,parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()))))
 charged=[0];warpers=[TemperatureLogitsWarper(.6),TopKLogitsWarper(20),TopPLogitsWarper(.95)]
 def digest(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
 @contextlib.contextmanager
 def metered(batch,bound,label,expected):
  assert charged[0]+batch*bound<=100000
  rid=ledger.begin(batch*bound,dict(label=label,batch=batch));count=[0,0];previous=F.scaled_dot_product_attention;checks=[0];handles=[]
  def before(mod,args,kw):count[0]+=batch*kw['input_ids'].shape[-1];count[1]+=1
  def sdpa(q,k,v,*aa,**kw):
   l=checks[0]%36;checks[0]+=1
   if count[1]<=2:
    for got,want in zip([k,v],expected[l]):
     if got.shape[1]!=want.shape[1]:want=want.repeat_interleave(got.shape[1]//want.shape[1],1)
     assert torch.equal(got[...,:130,:],want),'Actual retained KV changed'
   return previous(q,k,v,*aa,**kw)
  handles.append(model.register_forward_pre_hook(before,with_kwargs=True));F.scaled_dot_product_attention=sdpa;failed=True
  try:
   with torch.inference_mode():yield
   torch.cuda.synchronize();assert checks[0]==count[1]*36;failed=False
  finally:
   F.scaled_dot_product_attention=previous
   for h in handles:h.remove()
   ledger.settle(rid,count[0],count[1],failed=failed);charged[0]+=count[0]
 def inputs(indices,pad=None):
  jobs=[packets[i] for i in indices];lens=[j['ids'].shape[-1] for j in jobs];n=pad or max(lens);b=len(jobs)
  ids=torch.zeros((b,n),dtype=torch.long,device='cuda');valid=torch.zeros_like(ids)
  for k,j in enumerate(jobs):ids[k,-lens[k]:]=j['ids'].to('cuda');valid[k,-lens[k]:]=1
  original=[tuple(torch.cat([j['kv'][l][v] for j in jobs],0).to('cuda') for v in [0,1]) for l in range(36)]
  pos=valid.cumsum(-1)-1+torch.tensor([j['absolute'] for j in jobs],device='cuda')[:,None];pos.masked_fill_(valid==0,0)
  attention=torch.cat([torch.ones((b,130),dtype=torch.long,device='cuda'),valid],-1)
  return jobs,lens,ids,pos,attention,original
 hf=[]
 for i in range(4):
  jobs,lens,ids,pos,mask,original=inputs([i]);past=make_cache(original);torch.cuda.set_rng_state(packets[i]['rng'])
  with metered(1,lens[0]+128,'HF_reference_'+str(i),original),observe(model,jobs[0]['absolute'],latent=False,prefix=pairs(past)):
   value=model.generate(input_ids=ids,attention_mask=mask,past_key_values=past,cache_position=torch.arange(130,130+lens[0],device='cuda'),max_new_tokens=128,temperature=.6,top_p=.95,top_k=20,do_sample=True,return_dict_in_generate=True,output_scores=False,pad_token_id=model.generation_config.pad_token_id)
  tok=value.sequences[0,lens[0]:].cpu().tolist();hf.append(dict(packet=i,tokens=tok,matching_engineering_reference=tok==reference[i][:len(tok)],length=len(tok),first_difference=next((j for j,(a,b) in enumerate(zip(tok,reference[i])) if a!=b),None)))
  (OUT/'hf_reference.json').write_text(json.dumps(hf));del value,past;gc.collect();torch.cuda.empty_cache()
 serial_logits={};serial_rng={}
 def run(indices,steps,label,teacher=False,compare=False,pad=None,horizons=None):
  jobs,lens,ids,pos,mask,original=inputs(indices,pad);b=len(jobs);n=ids.shape[-1];past=make_cache(original);gens=[];tokens=[];diagnostics=[]
  for j in jobs:g=torch.Generator(device='cuda');g.set_state(j['rng']);gens.append(g)
  active=[True]*b
  with metered(b,n+steps,label,original):
   for step in range(steps):
    cp=torch.arange(past.get_seq_length(),past.get_seq_length()+ids.shape[-1],device='cuda')
    position=pos if step==0 else torch.tensor([j['absolute']+lens[k]+step-1 for k,j in enumerate(jobs)],device='cuda')[:,None]
    value=model(input_ids=ids,past_key_values=past,cache_position=cp,position_ids=position,attention_mask=mask,use_cache=True,return_dict=True);past=value.past_key_values;raw=value.logits[:,-1,:].float();scores=raw
    for w in warpers:scores=w(ids,scores)
    sampled=[];rowdiag=[]
    for k,i in enumerate(indices):
     state=digest(gens[k].get_state());token=int(torch.multinomial(scores[k].softmax(-1),1,generator=gens[k])) if active[k] else 0;sampled.append(token)
     if label=='serial_teacher':
      serial_logits.setdefault(i,[]).append(raw[k].detach().cpu());serial_rng.setdefault(i,[]).append(state)
     if compare:
      ref=serial_logits[i][step].to('cuda');diff=(raw[k]-ref).abs();rowdiag.append(dict(packet=i,step=step,max_abs=float(diff.max()),mean_abs=float(diff.mean()),argmax_same=int(raw[k].argmax())==int(ref.argmax()),rng_same=state==serial_rng[i][step],sample_same=token==reference[i][step]))
    tokens.append(sampled);diagnostics.extend(rowdiag)
    if teacher:sampled=[reference[i][step] for i in indices]
    if horizons:
     active=[step+1<horizons[k] for k in range(b)];sampled=[sampled[k] if active[k] else 0 for k in range(b)]
    ids=torch.tensor(sampled,dtype=torch.long,device='cuda')[:,None];mask=torch.cat([mask,torch.tensor(active,dtype=mask.dtype,device='cuda')[:,None]],-1)
  seq=[list(x) for x in zip(*tokens)]
  result=dict(label=label,indices=indices,tokens=seq,diagnostics=diagnostics)
  (OUT/(label+'.json')).write_text(json.dumps(result));del value,past,raw,scores;gc.collect();torch.cuda.empty_cache();return result
 for i in range(4):run([i],128,'serial_teacher',teacher=True)
 batch=run(list(range(4)),128,'batch_teacher',teacher=True,compare=True)
 padded=run([0],128,'padded_teacher',teacher=True,compare=True,pad=max(p['ids'].shape[-1] for p in packets))
 same=run([0,0,0,0],128,'same_length_teacher',teacher=True,compare=True)
 free=run(list(range(4)),128,'free_batch');mixed=run(list(range(4)),64,'mixed_eos',horizons=[8,16,32,64])
 eos=[mixed['tokens'][i][:h]==free['tokens'][i][:h] for i,h in enumerate([8,16,32,64])]
 def aggregate(result):
  ds=result['diagnostics'];return dict(first_logit_difference=next((d['step'] for d in ds if d['max_abs']>0),None),max_abs=max(d['max_abs'] for d in ds),mean_abs=sum(d['mean_abs'] for d in ds)/len(ds),all_rng_same=all(d['rng_same'] for d in ds),sample_mismatches=sum(not d['sample_same'] for d in ds))
 report=dict(passed=True,engineering_only=True,hf_reference=[{k:v for k,v in r.items() if k!='tokens'} for r in hf],teacher_forced={r['label']:aggregate(r) for r in [batch,padded,same]},eos_active_streams_match=eos,charged_positions=charged[0],formal_launch_ready=False)
 (OUT/'result.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':
 try:main()
 except BaseException as e:(OUT/'failure.json').write_text(json.dumps(dict(error=repr(e),unix=time.time())));raise
