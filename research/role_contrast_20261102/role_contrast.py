"""CPU mechanics prototype: preserve receiver-visible role contrasts.

No model execution, labels, training, V backfill or latent deletion. Fixed teacher
queries define a LOCAL surrogate, not a guarantee about future model execution.
The current snapshot/run files are not modified or imported by this prototype.
"""
import torch


def responses(attention, values, role_ids, keep, role_count):
    """Attention already includes the causal mask and recipient self-context.

    role_ids=-1 denotes shared/self context; role0..R-1 includes each producer's
    prompt AND latent positions. Selected weights must be renormalized.
    """
    a=attention[:,keep]
    mass=a.sum(-1,keepdim=True)
    if bool((mass<=0).any()):raise ValueError('Selected cache has zero query mass')
    a=a/mass;v=values[keep];ids=role_ids[keep]
    y=a@v
    contributions=torch.stack([(a*(ids==r).to(a.dtype))@v for r in range(role_count)])
    contrasts=contributions-contributions.mean(0,keepdim=True)
    return y,contrasts


def distortion(reference,response,objective):
    y,d=reference;ys,ds=response
    total=(y-ys).square().mean()
    contrast=(d-ds).square().mean()
    if objective not in ('total','role_contrast'):raise ValueError('Unknown objective')
    return total+(contrast if objective=='role_contrast' else 0),total,contrast


def select(attention,values,role_ids,protected,candidates,budget,*,objective='role_contrast',mandatory=()):
    """Select original prompt rows; protected includes every latent and sink.

    One KV head at one layer. Query heads sharing this KV head must be stacked
    along the query axis, never mixed with another layer/head's coordinates.
    Both the total-output control and candidate use the SAME <=2B attention
    shortlist and mandatory rows. Backward deletion is a heuristic, not a
    globally optimal or provably submodular solver.
    """
    a=attention.detach().to(dtype=torch.float64,device='cpu')
    v=values.detach().to(dtype=torch.float64,device='cpu')
    ids=role_ids.detach().to(dtype=torch.long,device='cpu')
    fixed=protected.detach().to(dtype=torch.bool,device='cpu')
    if a.ndim!=2 or v.ndim!=2 or a.shape[1]!=v.shape[0]:raise ValueError('Shape mismatch')
    n=v.shape[0]
    if ids.shape!=(n,) or fixed.shape!=(n,):raise ValueError('Metadata shape mismatch')
    if not torch.isfinite(a).all() or not torch.isfinite(v).all() or bool((a<0).any()):raise ValueError('Invalid tensors')
    if not torch.allclose(a.sum(-1),torch.ones(a.shape[0],dtype=a.dtype),atol=1e-6):raise ValueError('Attention must be normalized')
    pool=[int(x) for x in candidates];required={int(x) for x in mandatory}
    if len(pool)!=len(set(pool)) or any(x<0 or x>=n or fixed[x] or ids[x]<0 for x in pool):raise ValueError('Invalid prompt pool')
    if not 0<budget<=len(pool) or not required<=set(pool) or len(required)>budget:raise ValueError('Invalid budget/mandatory rows')
    if not all(bool(fixed[i]) or i in pool for i in range(n)):raise ValueError('Every position needs a protected or candidate designation')
    role_count=int(ids.max())+1
    if role_count<1 or set(ids.tolist())-set(range(role_count))-{-1}:raise ValueError('Invalid role labels')
    reference=responses(a,v,ids,torch.ones(n,dtype=torch.bool),role_count)
    saliency=a.mean(0)
    ranked=sorted((x for x in pool if x not in required),key=lambda x:(-float(saliency[x]),x))
    bank=set(required)|set(ranked[:min(len(pool),2*budget)-len(required)])
    keep=fixed.clone()
    for i in bank:keep[i]=True
    while len(bank)>budget:
        choices=[]
        for i in sorted(bank-required):
            trial=keep.clone();trial[i]=False
            loss,*_=distortion(reference,responses(a,v,ids,trial,role_count),objective)
            choices.append((float(loss),i))
        _,removed=min(choices)
        bank.remove(removed);keep[removed]=False
    loss,total,contrast=distortion(reference,responses(a,v,ids,keep,role_count),objective)
    chosen=torch.tensor(sorted(bank),dtype=torch.long)
    assert bool(keep[fixed].all()) and len(chosen)==budget
    return chosen,dict(objective=objective,loss=float(loss),total_error=float(total),contrast_error=float(contrast),
        shortlist_size=min(len(pool),2*budget),prompt_rows=budget,protected_rows=int(fixed.sum()),
        selected_per_role={str(r):sum(int(ids[i])==r for i in bank) for r in range(role_count)},
        local_proxy_only=True,model_calls=0)
