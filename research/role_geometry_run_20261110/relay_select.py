"""Three-handoff extension of the pinned T/R/G selectors; no model forwards."""
import hashlib
import torch
from controls import r_shuffle_labels,pseudo_source_labels,random_matched_keep

from block_selector import select as select_tr
from attention_selector import select as select_g

def provenance_map(mapping,stages,layers,heads,device):
 last=stages[-1];fresh=torch.arange(last['original_before'],last['original_end'],device=device)
 tail=fresh[None,None,:].expand(layers,heads,-1)
 result=tail.clone() if mapping is None else torch.cat([mapping,tail],-1)
 assert bool((result.diff(dim=-1)>0).all()) and bool((result>=0).all())
 return result

def masks(pos,stages,observed_length,prompt_budget=96):
 heads,total=pos.shape;assert observed_length>=total and 1<=len(stages)<=3
 labels=torch.zeros(heads,observed_length,dtype=torch.long,device=pos.device)
 protected=torch.ones(heads,observed_length,dtype=torch.bool,device=pos.device)
 known=pos<4
 for role,s in enumerate(stages,1):
  block=(pos>=s['original_before'])&(pos<s['original_end']);known|=block;labels[:,:total][block]=role
  prompt=(pos>=max(4,s['original_before']))&(pos<s['original_before']+s['prompt_len']);protected[:,:total][prompt]=False
  latent=(pos>=s['original_end']-10)&(pos<s['original_end']);assert bool((latent.sum(-1)==10).all())
 assert bool(known.all()) and bool(((pos<4).sum(-1)==4).all())
 labels[:,:total][pos<4]=0;candidate=~protected;counts=candidate.sum(-1);assert bool((counts==counts[0]).all())
 budget=min(prompt_budget,int(counts[0]));last=stages[-1];end=last['original_before']+last['prompt_len'];lo=max(4,last['original_before'],end-8)
 mandatory=torch.zeros_like(protected);mandatory[:,:total]=(pos>=lo)&(pos<end)
 assert not bool((mandatory&~candidate).any()) and bool((mandatory.sum(-1)<=budget).all())
 return labels,protected,candidate,mandatory,budget

def select_relay(xs,mapping,stages,captured,mode,seed=42):
 assert mode in ['G','R','T','R-shuffle','Pseudo-R','Random-matched'] and len(xs)==len(captured)
 layers=len(xs);heads=xs[0][0].shape[1];dev=xs[0][0].device;total=xs[0][0].shape[-2]
 allmap=provenance_map(mapping,stages,layers,heads,dev);assert allmap.shape==(layers,heads,total)
 chosen=[];reports=[];budget=None
 for layer,(a,v) in enumerate(captured):
  assert a.shape[0]==v.shape[0]==heads and a.shape[-1]==v.shape[-2] and a.shape[-1]>=total
  labels,protected,candidate,mandatory,k=masks(allmap[layer],stages,a.shape[-1]);assert budget in [None,k];budget=k
  control={}
  actual_labels=labels
  bins=torch.zeros_like(labels)
  # Per-source relative-age quantiles. Boundaries use original RoPE positions.
  for source,stage in enumerate(stages,1):
   mask=labels[:,:total]==source
   width=max(1,stage['prompt_len'])
   age=((allmap[layer]-stage['original_before']).clamp_min(0)*4//width).clamp_max(3)
   bins[:,:total][mask]=age[mask]
  if mode=='R-shuffle':
   actual_labels,control=r_shuffle_labels(labels,seed=seed+layer,return_report=True)
  elif mode=='Pseudo-R':
   actual_labels,control=pseudo_source_labels(labels,bins,seed=seed+layer,return_report=True)
  if mode=='Random-matched':
   reference,_=select_g(a,v,labels,protected,candidate,mandatory,k,'G')
   keep,control=random_matched_keep(candidate,mandatory,k,seed=seed+layer,reference=reference,labels=labels,bins=bins,protected=protected,return_report=True)
   report=dict(mode=mode,control=control)
  else:
   keep,report=(select_g if mode=='G' else select_tr)(a,v,actual_labels,protected,candidate,mandatory,k,mode if mode in ['G','R','T'] else 'R')
   report['control']=control
   if mode in ['R-shuffle','Pseudo-R']:
    report['selector_label_blocks_mse']=report.pop('true_blocks_mse');report['selector_label_total_mse']=report.pop('total_mse')
   report['matching_bins']='within-source relative prompt-position quartiles; not absolute cache age'
  ix=torch.stack([torch.where(row[:total])[0] for row in keep]);expected=4+10*len(stages)+k
  assert ix.shape==(heads,expected);chosen.append(ix);reports.append(report)
 indices=torch.stack(chosen);retained=indices.shape[-1]
 new=tuple(tuple(torch.gather(t,2,indices[l][None,:,:,None].expand(1,heads,retained,t.shape[-1])).contiguous() for t in pair) for l,pair in enumerate(xs))
 for l,(old,actual) in enumerate(zip(xs,new)):
  for a,b in zip(old,actual):assert torch.equal(torch.gather(a,2,indices[l][None,:,:,None].expand_as(b)),b)
 positions=torch.gather(allmap,2,indices)
 assert bool((positions.diff(dim=-1)>0).all())
 for s in stages:assert bool((((positions>=s['original_end']-10)&(positions<s['original_end'])).sum(-1)==10).all())
 assert bool(((positions<4).sum(-1)==4).all())
 summary=dict(mode=mode,handoff=len(stages),retained=retained,prompt_kept=budget,protected_latent=10*len(stages),protected_sink=4,source_k_equal_measured=True,source_v_equal_measured=True,original_end=stages[-1]['original_end'],actual_kv_bytes=sum(t.numel()*t.element_size() for p in new for t in p),full_bytes_from_original_positions=stages[-1]['original_end']*sum(p[0].shape[0]*p[0].shape[1]*p[0].shape[-1]*p[0].element_size()*2 for p in xs),position_sha=hashlib.sha256(positions.detach().cpu().numpy().tobytes()).hexdigest(),original_RoPE=True,receiver_self_not_relayed=True)
 return new,positions,indices,reports,summary
