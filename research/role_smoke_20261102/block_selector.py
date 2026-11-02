"""Batched fixed-key deletion with exact renormalization, no model imports.

Each batch member is one KV head. Queries stack GQA heads, never average them.
Block 0 contains fixed sink/self; 1..3 are source roles. No V fitting.
"""
import torch

def shifted_blocks(labels):
    result = labels.clone()
    for h in range(labels.shape[0]):
        ix = torch.where(labels[h] != 0)[0]
        counts = torch.bincount(labels[h, ix], minlength=4)[1:]
        positive = counts[counts > 0]
        shift = max(1, int(positive.min()) // 2)
        result[h, ix] = torch.roll(labels[h, ix], shift)
        assert torch.equal(torch.bincount(result[h], minlength=4),
                           torch.bincount(labels[h], minlength=4))
        assert bool((result[h, ix] != labels[h, ix]).any())
    return result

def components(a, v, labels, keep):
    weighted = a * keep[:, None, :]
    z = weighted.sum(-1, keepdim=True)
    assert bool((z > 0).all())
    return torch.stack([(weighted * (labels == r)[:, None, :]) @ v / z
                        for r in range(4)], 1)

def select(a, v, labels, protected, candidates, mandatory, budget, mode):
    """All arrays [H,...]; returns shared-length storage indices per KV head.

    a[H,Q,N], v[H,N,D]; input tensors remain untouched. Bank has at most 2B
    candidate prompt rows including mandatory recent rows. Protected includes
    self keys for objective only; caller excludes self when gathering relay KV.
    """
    assert mode in ('T','R','R-shuffle')
    assert a.ndim == v.ndim == 3 and a.shape[0] == v.shape[0]
    assert a.shape[-1] == v.shape[1] == labels.shape[-1]
    assert not bool((protected & candidates).any())
    assert bool((protected | candidates).all())
    assert not bool((mandatory & ~candidates).any())
    assert bool((mandatory.sum(-1) <= budget).all())
    assert bool((candidates.sum(-1) >= budget).all())
    a, v = a.double(), v.double()
    role = shifted_blocks(labels) if mode == 'R-shuffle' else labels
    h, q, n = a.shape; d = v.shape[-1]
    # Stable tie break is original storage order, independent of method labels.
    ranks = torch.argsort(a.sum(1).masked_fill(~(candidates & ~mandatory),
                                              -torch.inf), dim=-1, descending=True, stable=True)
    available = int((candidates & ~mandatory).sum(-1).min())
    assert bool((mandatory.sum(-1) == mandatory.sum(-1)[0]).all())
    bank_n = min(2 * budget, int(candidates.sum(-1).min()))
    extra = bank_n - int(mandatory.sum(-1)[0])
    assert 0 <= extra <= available
    bank = mandatory.clone().scatter(1, ranks[:, :extra], True)
    ix = torch.stack([torch.where(row)[0] for row in bank])
    keep = protected | bank
    full = torch.ones_like(keep)
    target = components(a, v, role, full)
    if mode == 'T': target = target.sum(1, keepdim=True)
    blocks = target.shape[1]
    masks = torch.stack([(role == r) for r in range(4)], 1)
    if mode == 'T': masks = torch.ones_like(masks[:, :1])
    numer = torch.einsum('hqn,hrn,hnd->hrqd', a * keep[:, None, :], masks.double(), v)
    z = (a * keep[:, None, :]).sum(-1)
    w = torch.gather(a, 2, ix[:, None, :].expand(-1,q,-1))
    vv = torch.gather(v, 1, ix[:, :, None].expand(-1,-1,d))
    rr = torch.gather(role, 1, ix) if blocks > 1 else torch.zeros_like(ix)
    immutable = torch.gather(mandatory, 1, ix)
    active = torch.ones_like(immutable)
    # Target/V dot products are constant throughout the backward deletion.
    tv = torch.einsum('hrqd,hcd->hrqc', target, vv)
    gather_r = rr[:,None,None,:].expand(-1,1,q,-1)
    tv = tv.gather(1,gather_r).squeeze(1)
    v2 = vv.square().sum(-1)[:,None,:]
    target2 = target.square().sum((1,3))
    batch = torch.arange(h, device=a.device)
    for _ in range(bank_n - budget):
        nv = torch.einsum('hrqd,hcd->hrqc', numer, vv).gather(1,gather_r).squeeze(1)
        base2 = numer.square().sum((1,3))[:, :, None]
        nt = (numer * target).sum((1,3))[:, :, None]
        zp = z[:,:,None] - w
        residual = ((base2 - 2*w*nv + w.square()*v2) / zp.clamp_min(1e-18).square()
                    -2*(nt-w*tv)/zp.clamp_min(1e-18) + target2[:,:,None])
        loss = residual.mean(1) / (blocks*d)
        loss = loss.masked_fill(immutable | ~active | (zp <= 0).any(1), torch.inf)
        delete = loss.argmin(-1)
        assert bool(torch.isfinite(loss[batch, delete]).all())
        ww = w[batch,:,delete]
        delta = ww[:,:,None] * vv[batch,delete][:,None,:]
        numer[batch,rr[batch,delete]] -= delta
        z -= ww
        active[batch,delete] = False
    selected = protected.clone()
    selected.scatter_(1,ix,active)
    assert bool(((selected & candidates).sum(-1) == budget).all())
    observed = components(a,v,role,selected)
    true_target = components(a,v,labels,full)
    true_observed = components(a,v,labels,selected)
    report = dict(mode=mode, bank_rows=bank_n, prompt_budget=budget,
        total_mse=float((true_target.sum(1)-true_observed.sum(1)).square().mean()),
        true_blocks_mse=float((true_target-true_observed).square().mean()),
        objective_mse=float((target-(observed.sum(1,keepdim=True) if mode=='T' else observed)).square().mean()),
        shuffle_changed_fraction=float((role!=labels).double().mean()))
    return selected, report
