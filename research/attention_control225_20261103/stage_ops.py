"""Stagewise KV/position mechanics and noninvasive latent-attention observation."""
import ast,contextlib,hashlib,time
from pathlib import Path
import torch
import torch.nn.functional as F
from current_jive import select_current

def pairs(cache):
    if cache is None:return ()
    if hasattr(cache,'to_legacy_cache'):return tuple(cache.to_legacy_cache())
    if hasattr(cache,'layers'):return tuple((x.keys,x.values) for x in cache.layers)
    return tuple(cache)

def make_cache(xs):
    from transformers.cache_utils import DynamicCache
    if hasattr(DynamicCache,'from_legacy_cache'):return DynamicCache.from_legacy_cache(tuple(xs))
    c=DynamicCache()
    for l,(k,v) in enumerate(xs):c.update(k,v,l)
    return c

def clone_cache(cache):return make_cache(tuple(tuple(t.clone() for t in p) for p in pairs(cache))) if cache is not None else None

def load_obf(factory=make_cache):
    """Only replace cache-constructor plumbing in pinned author's HOBF AST."""
    ns={}
    base=Path(__file__).with_name('author_BaseKVCompressor.py')
    exec(compile(base.read_text(encoding='utf-8'),str(base),'exec'),ns)
    ns['DynamicCache']=type('Factory',(),{'from_legacy_cache':staticmethod(factory)})
    path=Path(__file__).with_name('author_HOBF.py')
    tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if not (isinstance(n,ast.ImportFrom) and n.module in ['transformers','compression_methods.BaseKVCompressor'])]
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),ns)
    return ns['HOBF'](sink_size=4,kv_budget=32,pca_rank=8)

@contextlib.contextmanager
def observe(model,original_before,*,latent=True,prefix=(),capture_queries=0):
    """Explicit original RoPE on ALL forwards; collect exact-score latent probes.
    The native SDPA result is returned unchanged. Score matmuls do not sample RNG.
    """
    before=[0];proof=dict(forwards=0,positions=0,sdpa_checks=0,attention_steps=[],query_attention=[])
    layer_count=len(model.model.layers);old_sdpa=F.scaled_dot_product_attention
    def pre(mod,args,kw):
        ids=kw.get('input_ids');embeds=kw.get('inputs_embeds');count=int(ids.shape[-1] if ids is not None else embeds.shape[-2])
        past=kw.get('past_key_values');start=0 if past is None else int(past.get_seq_length())
        if not proof['forwards']:before[0]=start
        cp=torch.arange(start,start+count,device=model.get_input_embeddings().weight.device)
        given=kw.get('cache_position')
        if given is not None and not torch.equal(given,cp):raise RuntimeError('storage position drift')
        kw['cache_position']=cp
        kw['position_ids']=(cp+original_before-before[0]).unsqueeze(0)
        first=int(kw['position_ids'][0,0]);last=int(kw['position_ids'][0,-1])
        if first!=original_before+proof['positions'] or last>=32768:raise RuntimeError('original RoPE drift/cap')
        proof['positions']+=count;proof['forwards']+=1
    handle=model.register_forward_pre_hook(pre,with_kwargs=True)
    def sdpa(q,k,v,*args,**kwargs):
        result=old_sdpa(q,k,v,*args,**kwargs)
        layer=proof['sdpa_checks']%layer_count;proof['sdpa_checks']+=1
        if prefix and proof['forwards']<=2:
            for actual,expected in zip([k,v],prefix[layer]):
                if actual.shape[1]!=expected.shape[1]:expected=expected.repeat_interleave(actual.shape[1]//expected.shape[1],dim=1)
                if not torch.equal(actual[...,:expected.shape[-2],:],expected):raise RuntimeError('actual attention bypassed/changed retained KV')
        if capture_queries and proof['forwards']>=1:
            if q.shape[-2] < 1: raise RuntimeError('receiver probe has no queries')
            kk=k
            if q.shape[1]!=k.shape[1]:kk=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
            scale=kwargs.get('scale')
            scores=q.float()@kk.float().transpose(-1,-2)*(scale if scale is not None else q.shape[-1]**-.5)
            mask=kwargs.get('attn_mask',args[0] if args else None)
            if mask is not None:
                if mask.dtype==torch.bool:scores=scores.masked_fill(~mask,-torch.inf)
                else:scores=scores+mask.float()
            if kwargs.get('is_causal',args[2] if len(args)>2 else False):
                raise RuntimeError('unexpected causal receiver probe')
            take=min(int(capture_queries),int(scores.shape[-2]))
            probe=scores.softmax(dim=-1)[...,-take:,:].mean(-2).detach()
            if len(proof['query_attention'])<=layer:proof['query_attention'].append((probe,1))
            else:
                previous,count=proof['query_attention'][layer]
                proof['query_attention'][layer]=(previous+probe,count+1)
        if latent and proof['forwards']>=2:
            if q.shape[-2]!=1:raise RuntimeError('latent attention must have one query')
            kk=k
            if q.shape[1]!=k.shape[1]:kk=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
            scale=kwargs.get('scale')
            scores=q.float()@kk.float().transpose(-1,-2)*(scale if scale is not None else q.shape[-1]**-.5)
            mask=kwargs.get('attn_mask',args[0] if args else None)
            if mask is not None:
                if mask.dtype==torch.bool:scores=scores.masked_fill(~mask,-torch.inf)
                else:scores=scores+mask.float()
            causal=kwargs.get('is_causal',args[2] if len(args)>2 else False)
            if causal:raise RuntimeError('unexpected single-query causal SDPA; inspect rather than guess')
            step=proof['forwards']-2
            if len(proof['attention_steps'])<=step:proof['attention_steps'].append([])
            if len(proof['attention_steps'][step])!=layer:raise RuntimeError('attention layer order drift')
            proof['attention_steps'][step].append(torch.softmax(scores,dim=-1).squeeze(-2).detach())
        return result
    F.scaled_dot_product_attention=sdpa
    try:yield proof
    finally:handle.remove();F.scaled_dot_product_attention=old_sdpa
    if proof['sdpa_checks']!=proof['forwards']*layer_count:raise RuntimeError('missing actual SDPA checks')

def prompt_scores(attentions,layer,heads,history,prompt,total):
    if isinstance(attentions,dict) and attentions.get('kind')=='receiver_probe':
        a=attentions['layers'][layer].float()
        if a.shape[0]!=heads:a=a.reshape(heads,a.shape[0]//heads,a.shape[-1]).mean(1)
        if a.shape[-1]<total:a=F.pad(a,(0,total-a.shape[-1]))
        return a[:,history:history+prompt]
    agg=None
    for step in attentions:
        a=step[layer].float()
        if a.ndim==4:a=a.squeeze(-2)
        a=F.pad(a,(0,total-a.shape[-1]))
        agg=a if agg is None else agg+a
    if agg is None:raise ValueError('no measured latent attention')
    qheads=agg.shape[1]
    scores=agg[:,:,history:history+prompt]
    if qheads!=heads:scores=scores.reshape(1,heads,qheads//heads,prompt).sum(2)
    return scores[0]

def compress(cache,mapping,*,role,steps,prompt_len,original_before,method,attention,obf=None,past_ranges=()):
    xs=pairs(cache);total=xs[0][0].shape[-2];history=total-prompt_len-steps
    sink=4 if role=='planner' else 0;k=min(32,prompt_len-sink)
    if k<0:raise ValueError('prompt too short for sink')
    layers=len(xs);heads=xs[0][0].shape[1];dev=xs[0][0].device
    fresh=torch.arange(original_before,original_before+prompt_len+steps,device=dev)
    if mapping is None:all_map=fresh.view(1,1,-1).expand(layers,heads,-1).clone()
    else:all_map=torch.cat([mapping,fresh.view(1,1,-1).expand(layers,heads,-1)],-1)
    if all_map.shape[-1]!=total:raise RuntimeError('mapping length drift')
    if method=='full':return cache,all_map,dict(method='full',retained=total,original_end=original_before+prompt_len+steps)
    chosen=[];decompositions=[]
    for l,(key,value) in enumerate(xs):
        scores=prompt_scores(attention,l,heads,history,prompt_len,total)
        if method=='receiver_attention_sink_recent':
            score=scores.float()
            score[:,:sink]=-torch.inf
            recent=min(8,k)
            recent_ix=torch.arange(max(sink,prompt_len-recent),prompt_len,device=dev).view(1,-1).expand(heads,-1)
            score.scatter_(1,recent_ix,-torch.inf)
            remaining=k-recent
            top=torch.topk(score,remaining,dim=-1).indices if remaining else torch.empty((heads,0),dtype=torch.long,device=dev)
            ix=torch.cat([recent_ix,top],-1).sort(-1).values
        elif method=='headobf':
            scores[:,:sink]=-torch.inf;ix=torch.topk(scores,k,dim=-1).indices.sort(-1).values
        elif method=='jive':
            views=[]
            for lo,hi in past_ranges:
                mask=(mapping[l]>=lo)&(mapping[l]<hi)&(mapping[l]>=4)
                sizes=mask.sum(-1)
                if not bool((sizes==sizes[0]).all()):raise RuntimeError('inconsistent prior-role row counts')
                count=int(sizes[0]);ind=torch.argsort(mask.to(torch.long),descending=True,stable=True)[:,:count]
                views.append(torch.gather(value[0],1,ind[:,:,None].expand(heads,count,value.shape[-1])))
            views.append(value[0,:,history:history+prompt_len,:])
            ix,diagnostic=select_current(views,scores,budget=32,sink=sink)
            diagnostic['layer']=l;decompositions.append(diagnostic)
        else:raise ValueError(method)
        left=torch.arange(history+sink,device=dev).view(1,-1).expand(heads,-1)
        tail=torch.arange(total-steps,total,device=dev).view(1,-1).expand(heads,-1)
        chosen.append(torch.cat([left,ix+history,tail],-1))
    indices=torch.stack(chosen);new_len=history+sink+k+steps
    if indices.shape!=(layers,heads,new_len):raise RuntimeError('budget drift')
    if method=='headobf':
        result,_,_=obf.compress(past_key_values=xs,latent_steps=steps,all_steps_attentions=attention,prompt_mask=torch.ones(1,prompt_len,dtype=torch.long,device=dev))
    else:
        result=make_cache(tuple(tuple(torch.gather(t,2,indices[l][None,:,:,None].expand(1,heads,new_len,t.shape[-1])).contiguous() for t in pair) for l,pair in enumerate(xs)))
    for l,((key,value),(nk,nv)) in enumerate(zip(xs,pairs(result))):
        idx=indices[l][None,:,:,None].expand(1,heads,new_len,key.shape[-1])
        if not torch.equal(torch.gather(key,2,idx),nk):raise RuntimeError('selected original K changed')
        expected=torch.gather(value,2,idx)
        if method=='jive' and not torch.equal(expected,nv):raise RuntimeError('JIVE changed original V')
        if not torch.equal(expected[:,:,:history+sink],nv[:,:,:history+sink]) or not torch.equal(expected[:,:,-steps:],nv[:,:,-steps:]):raise RuntimeError('protected V changed')
    new_map=torch.gather(all_map,2,indices)
    receipt=dict(method=method,role=role,storage_before=history,prompt_len=prompt_len,latent_steps=steps,
        retained=new_len,original_end=original_before+prompt_len+steps,prompt_kept=k,sink_kept=sink,
        source_k_equal=True,source_v_equal=method=='jive',protected_latent_and_sink=True,
        original_tensor_bytes=sum(t.numel()*t.element_size() for p in xs for t in p),
        retained_tensor_bytes=sum(t.numel()*t.element_size() for p in pairs(result) for t in p),
        selected_position_sha256=hashlib.sha256(new_map.cpu().contiguous().numpy().tobytes()).hexdigest(),
        headwise_positions=True,value_backfill=method=='headobf',decompositions=decompositions,external_questions_used=0)
    return result,new_map,receipt
