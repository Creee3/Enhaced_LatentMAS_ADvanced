"""Bounded, single-question JIVE-style geometry; no learned/calibration state.

For a fixed layer/head, roles supply variable-length prompt V matrices [tokens,d].
The common coordinate axis is the SAME model/layer/head's d value dimensions.
This is a deterministic geometry adaptation, not population-statistical JIVE.
"""
import time
import torch

def _top(c,rank):
    c=(c+c.transpose(-1,-2))*.5
    _,u=torch.linalg.eigh(c)
    return u[...,-rank:]

def _perp(c,b):
    cb=c@b
    return c-cb@b.transpose(-1,-2)-b@cb.transpose(-1,-2)+b@(b.transpose(-1,-2)@cb)@b.transpose(-1,-2)

def _private(c,u,rank):
    w=_top(_perp(c,u),rank)
    # Reorthogonalize, including degenerate zero-private-signal cases.
    w=w-u@(u.transpose(-1,-2)@w)
    q,s,_=torch.linalg.svd(w,full_matrices=False)
    return q*(s>1e-8).unsqueeze(-2)

def decompose_current(views,*,joint_rank=4,individual_rank=4,iterations=20):
    """Return feature bases for the CURRENT final view of this question.

    Rank-constrained joint/individual alternating geometry via covariance algebra.
    Preserve the best feasible reconstruction in a fixed20-iteration budget;
    do not equate a bounded approximate solution with proved convergence.
    """
    if not 2<=len(views)<=3:raise ValueError('two/three currently available roles required')
    if any(v.ndim!=3 for v in views):raise ValueError('heads,tokens,dim required')
    h,_,d=views[0].shape
    if any(v.shape[0]!=h or v.shape[2]!=d or v.shape[1]<joint_rank+individual_rank for v in views):raise ValueError('unaligned coordinates/insufficient rows')
    if any(not torch.isfinite(v).all() for v in views):raise ValueError('nonfinite prompt values')
    original_device=views[-1].device
    start=time.monotonic()
    # Small eigensystems are handled on CPU; the LLM remains unchanged on CUDA.
    xs=[v.detach().to(device='cpu',dtype=torch.float64) for v in views]
    cs=[]
    for x in xs:
        norm=x.square().sum((-1,-2),keepdim=True).sqrt()
        if bool((norm<=1e-12).any()):raise ValueError('zero view')
        y=x/norm
        cs.append(y.transpose(-1,-2)@y)
    total=sum(cs)
    def error(u,ws):
        captured=sum((u*(c@u)).sum((-1,-2))+(w*(c@w)).sum((-1,-2)) for c,w in zip(cs,ws))
        return (total.diagonal(dim1=-2,dim2=-1).sum(-1)-captured).clamp_min(0)
    u=_top(total,joint_rank);ws=[_private(c,u,individual_rank) for c in cs]
    best=error(u,ws);initial=best.clone();bu=u.clone();bws=[w.clone() for w in ws]
    history=[float(best.mean())];last=best.clone();last_change=torch.full_like(best,float('inf'))
    for _ in range(iterations):
        u=_top(sum(_perp(c,w) for c,w in zip(cs,ws)),joint_rank)
        ws=[_private(c,u,individual_rank) for c in cs]
        e=error(u,ws)
        if not torch.isfinite(e).all():raise RuntimeError('nonfinite decomposition')
        improved=e<best
        best=torch.minimum(best,e);bu=torch.where(improved[:,None,None],u,bu)
        bws=[torch.where(improved[:,None,None],w,bw) for w,bw in zip(ws,bws)]
        last_change=(e-last).abs()/last.clamp_min(1e-12);last=e
        history.append(float(best.mean()))
    orth=float((bu.transpose(-1,-2)@bws[-1]).abs().max())
    if orth>1e-7:raise RuntimeError('joint/private coordinates not orthogonal')
    report=dict(method='within_question_bounded_fixed_rank_JIVE_geometry',
        available_views=len(views),prompt_rows=[v.shape[1] for v in views],
        joint_rank=joint_rank,individual_rank=individual_rank,iteration_budget=iterations,
        initial_error=float(initial.mean()),best_error=float(best.mean()),best_history=history,
        objective_stable_heads=int((last_change<=1e-6).sum()),heads=h,
        joint_individual_orthogonality_max=orth,
        seconds=time.monotonic()-start,external_questions_used=0,
        calibration_model_used=False,solver='best feasible bounded approximation, not certified converged optimum')
    return bu.to(original_device,dtype=torch.float32),bws[-1].to(original_device,dtype=torch.float32),report

def _mmr(features,base,count,chosen):
    unit=features/features.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    sim=torch.zeros_like(base);chosen=list(chosen)
    for j in chosen:sim=torch.maximum(sim,(unit@unit[j]).abs())
    for _ in range(count):
        score=base*(1-.25*sim)
        if chosen:score[chosen]=-torch.inf
        j=int(torch.argmax(score));chosen.append(j);sim=torch.maximum(sim,(unit@unit[j]).abs())
    return chosen

def select_current(views,attention,*,sink=0,budget=32):
    value=views[-1];h,m,d=value.shape
    if attention.shape!=(h,m) or not torch.isfinite(attention).all() or bool((attention<0).any()):raise ValueError('invalid attention')
    k=min(budget,m-sink)
    if k<0:raise ValueError('invalid sink')
    if len(views)==1:
        score=attention.clone();score[:,:sink]=-torch.inf
        return torch.topk(score,k,dim=-1).indices.sort(-1).values,dict(method='planner_attention_bootstrap',available_views=1,external_questions_used=0)
    clean_views=views[:-1]+[value[:,sink:,:]]
    bj,ba,report=decompose_current(clean_views)
    z=value.float();j=z@bj;a=z@ba
    norm=z.square().sum(-1).clamp_min(1e-12)
    jf=(j.square().sum(-1)/norm).clamp(0,1)
    af=(a.square().sum(-1)/norm).clamp(0,1)
    if bool((jf+af>1.00001).any()):raise RuntimeError('component energy partition failed')
    q=attention.float()/attention.float().max(-1,keepdim=True).values.clamp_min(1e-12);q=.05+.95*q
    indices=[]
    for head in range(h):
        common=q[head]*(.05+.95*jf[head]);common[:sink]=-torch.inf
        nc=round(k*.25);anchors=torch.argsort(common,descending=True,stable=True)[:nc].tolist()
        private=q[head]*(.05+.95*af[head]);private[:sink]=-torch.inf
        indices.append(sorted(_mmr(a[head],private,k-nc,anchors)))
    report.update(common_positions=round(k*.25),private_positions=k-round(k*.25))
    return torch.tensor(indices,dtype=torch.long,device=value.device),report
