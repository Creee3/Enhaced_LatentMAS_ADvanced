"""Native relay forwards plus explicit one-prefill diagnostic probes."""
import contextlib,hashlib,json,time
import torch
import torch.nn.functional as F
from diagnostics import receiver_fidelity

@contextlib.contextmanager
def capture_prefill(model):
    """Read existing first native forward; no extra forward and no sampling."""
    found={};started=[None]
    def start_hook(module,inputs):
        if started[0] is None:torch.cuda.synchronize();started[0]=time.monotonic()
    def norm_hook(module,inputs,output):
        if 'logits' not in found and 'hidden' not in found:found['hidden']=output[:,-1,:].detach().clone()
    def output_hook(module,inputs,output):
        if 'logits' not in found:
            torch.cuda.synchronize();found['prefill_seconds']=time.monotonic()-started[0];found['logits']=output.logits[:,-1,:].detach().clone()
    c=model.register_forward_pre_hook(start_hook);a=model.model.norm.register_forward_hook(norm_hook);b=model.register_forward_hook(output_hook)
    try:yield found
    finally:a.remove();b.remove();c.remove()

def fidelity(reference,selected):
    return receiver_fidelity(reference['logits'],selected['logits'],full_hidden=reference['hidden'],selected_hidden=selected['hidden'])

def probe(engine,model,item,receiver,cache,absolute,*,prepare_messages,observe,pairs,clone_cache,detail=True,kind='receiver_predeletion_probe'):
    messages=prepare_messages(receiver,item)
    prompt,ids,mask=engine.prepare(messages);before=cache.get_seq_length();copied=clone_cache(cache)
    full_mask=torch.cat([torch.ones(1,before,dtype=mask.dtype,device=mask.device),mask],-1)
    cp=torch.arange(before,before+ids.shape[-1],device=ids.device);captured=[]
    assert absolute+ids.shape[-1]<32768
    with torch.inference_mode(),capture_prefill(model) as features,observe(model,absolute,latent=False,prefix=pairs(cache),capture_queries=0) as proof:
        previous=F.scaled_dot_product_attention
        def capture(q,k,v,*aa,**kw):
            output=previous(q,k,v,*aa,**kw)
            if q.shape[1]!=k.shape[1]:k=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
            take=min(16,q.shape[-2]);score=q[...,-take:,:].float() @ k.float().transpose(-1,-2)
            score*=kw.get('scale') or q.shape[-1]**-.5
            amask=kw.get('attn_mask',aa[0] if aa else None)
            assert not kw.get('is_causal',aa[2] if len(aa)>2 else False)
            if amask is not None:
                if amask.shape[-2]>1:amask=amask[...,-take:,:]
                score=score.masked_fill(~amask,-torch.inf) if amask.dtype==torch.bool else score+amask.float()
            heads=model.config.num_key_value_heads;group=q.shape[1]//heads
            attn=score.softmax(-1).reshape(heads,group*take,-1).detach();vv=v[0]
            if vv.shape[0]!=heads:
                grouped=vv.reshape(heads,group,vv.shape[-2],vv.shape[-1]);assert torch.equal(grouped,grouped[:,:1].expand_as(grouped));vv=grouped[:,0]
            captured.append((attn,vv.detach()))
            return output
        if detail:F.scaled_dot_product_attention=capture
        try:
            engine.invoke(int(ids.shape[-1]),lambda:model(input_ids=ids,attention_mask=full_mask,past_key_values=copied,cache_position=cp,use_cache=True,return_dict=True),dict(dataset=item['dataset'],question_id=item['question_id'],receiver=receiver,kind=kind),dict(prompt=prompt,input_ids=ids[0].cpu().tolist(),past_positions=before),expected_past=copied)
        finally:F.scaled_dot_product_attention=previous
    rec=dict(engine.last_receipt);assert proof['forwards']==1 and proof['positions']==rec['actual_positions'] and proof['sdpa_checks']==36
    if detail:assert len(captured)==36
    rec['position_proof']={k:proof[k] for k in ['forwards','positions','sdpa_checks']}
    del copied
    return captured,rec,features
