"""Pinned author H/L operators; cache-constructor adapter only, independent selection checks."""
import ast,hashlib
from pathlib import Path
import torch
from common import HERE,sha

def load_author(mode,factory):
 assert mode in ['H','L']
 ns={};base=HERE/'author/BaseKVCompressor.py'
 exec(compile(base.read_text(encoding='utf-8'),str(base),'exec'),ns)
 ns['DynamicCache']=type('Factory',(),{'from_legacy_cache':staticmethod(factory)})
 name='Headwise' if mode=='H' else 'Layerwise';path=HERE/'author'/(name+'.py');tree=ast.parse(path.read_text(encoding='utf-8'))
 tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module in ['transformers','compression_methods.BaseKVCompressor'])]
 exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),ns)
 obj=ns[name](sink_size=4,kv_budget=32)
 return obj,dict(commit='36f43183e84d2fbf3b84c9bf4e3e06b7c442a523',file_sha256=sha(path),base_sha256=sha(base),cache_factory_only_adapter=True)

def relay_hl(cache,mapping,rec,mode,attention,operator,pairs,factory):
 from stage_ops import prompt_scores
 xs=pairs(cache);layers=len(xs);heads=xs[0][0].shape[1];total=xs[0][0].shape[-2];history=total-rec['prompt_len']-10
 assert len(attention)==10 and all(len(x)==layers for x in attention)
 result,_,metadata=operator.compress(past_key_values=xs,latent_steps=10,all_steps_attentions=attention,prompt_mask=torch.ones(1,rec['prompt_len'],device=xs[0][0].device,dtype=torch.long))
 selected=torch.tensor(metadata['selected_prompt_positions_matrix'][0],device=xs[0][0].device,dtype=torch.long)
 sink=4 if rec['role']=='planner' else 0;k=min(32,rec['prompt_len']-sink)
 assert selected.shape==(layers,heads,k)
 for l in range(layers):
  scores=prompt_scores(attention,l,heads,history,rec['prompt_len'],total)
  if mode=='H':expected=(torch.topk(scores[:,sink:],k,dim=-1).indices+sink).sort(-1).values
  else:
   score=scores.sum(0);score[:sink]=-torch.inf
   expected=torch.topk(score,k,dim=-1).indices.sort(-1).values[None,:].expand(heads,-1)
  assert torch.equal(selected[l],expected),'author vs independent sender attention top-k mismatch'
 dev=xs[0][0].device;left=torch.arange(history+sink,device=dev).view(1,1,-1).expand(layers,heads,-1);tail=torch.arange(total-10,total,device=dev).view(1,1,-1).expand(layers,heads,-1)
 ix=torch.cat([left,selected+history,tail],-1)
 fresh=torch.arange(rec['original_before'],rec['original_end'],device=dev).view(1,1,-1).expand(layers,heads,-1)
 allpos=fresh if mapping is None else torch.cat([mapping,fresh],-1);pos=torch.gather(allpos,2,ix)
 for l,(old,new) in enumerate(zip(xs,pairs(result))):
  for a,b in zip(old,new):assert torch.equal(torch.gather(a,2,ix[l][None,:,:,None].expand_as(b)),b)
 if mode=='L':assert torch.equal(pos,pos[:,:1].expand_as(pos))
 receipt=dict(method=mode,retained=pos.shape[-1],original_end=rec['original_end'],original_RoPE=True,source_k_equal_measured=True,source_v_equal_measured=True,history_kept=history,source_prompt_len=rec['prompt_len'],sender=rec['role'],actual_kv_bytes=sum(t.numel()*t.element_size() for p in pairs(result) for t in p),full_bytes_from_original_positions=rec['original_end']*147456,position_sha=hashlib.sha256(pos.cpu().contiguous().numpy().tobytes()).hexdigest(),prompt_budget=32,mandatory_recent=0,sender_attention_steps=10,independent_author_selection_equal=True,layer_heads_share_positions=mode=='L')
 assert receipt['actual_kv_bytes']==receipt['retained']*147456
 return result,pos,ix,receipt
