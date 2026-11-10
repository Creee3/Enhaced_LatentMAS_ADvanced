"""Twelve explicitly named selectors; no model forwards."""
import hashlib
import torch
from relay_select import select_relay,provenance_map
from author_ops import select_author

def select_method(cache,mapping,stages,method,sender_attention,captured,seed,pairs,make_cache):
    xs=pairs(cache);allmap=provenance_map(mapping,stages,len(xs),xs[0][0].shape[1],xs[0][0].device)
    if method=='Full':
        new,pos,ix,report=select_author(xs,mapping,stages,method,sender_attention,make_cache)
    elif method in ['Gen','H','L','H-OBF','L-OBF']:
        new,pos,ix,report=select_author(xs,mapping,stages,method,sender_attention,make_cache)
    else:
        new_xs,pos,ix,reports,report=select_relay(xs,mapping,stages,captured,method,seed=seed);new=make_cache(new_xs);report['selector_layers']=reports
    new_xs=pairs(new);retained=pos.shape[-1]
    for l,(old,after) in enumerate(zip(xs,new_xs)):
        index=ix[l][None,:,:,None].expand_as(after[0]);assert torch.equal(torch.gather(old[0],2,index),after[0])
        if method not in ['H-OBF','L-OBF']:assert torch.equal(torch.gather(old[1],2,index),after[1])
    for stage in stages:assert bool((((pos>=stage['original_end']-10)&(pos<stage['original_end'])).sum(-1)==10).all())
    assert bool(((pos<4).sum(-1)==4).all()) and bool((pos.diff(dim=-1)>0).all())
    report.update(method=method,handoff=len(stages),retained=retained,source_k_equal_measured=True,source_v_equal_measured=method not in ['H-OBF','L-OBF'],protected_latent_sink_equal_measured=True,original_RoPE=True,actual_kv_bytes=sum(t.numel()*t.element_size() for pair in new_xs for t in pair),full_bytes_from_original_positions=stages[-1]['original_end']*147456,position_sha=hashlib.sha256(pos.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),original_end=stages[-1]['original_end'])
    bytes_per_position=sum(t.shape[0]*t.shape[1]*t.shape[-1]*t.element_size() for pair in xs for t in pair)
    assert report['actual_kv_bytes']==retained*bytes_per_position
    return new,pos,ix,report
