"""Versioned transport-codec feasibility prototype. Never loads an LLM.

All source means, bases, coefficients, exact rows, labels, and a 64-byte header
are serialized. Every per-layer/head/channel packet is padded to 33,280 bytes.
This is lossy encoding of full prompt KV, not native-row eviction or GPU-cache
memory compression. Already-rotated K stays at its original absolute position.
"""
import struct
import numpy as np
import torch

HEADER = struct.Struct('<8s14I')
MAGIC = b'RKVF0001'
PACKET_BYTES = 130 * 128 * 2
ARMS = ('Global', 'Source-Shared', 'Source-Separate', 'Shuffle-42', 'Shuffle-43', 'Shuffle-44')


def layout(lengths, d=128, latent=10):
    assert len(lengths) == 3 and min(lengths) > 12
    total = sum(lengths)
    exact_prompt = np.r_[np.arange(4), np.arange(total-8, total)]
    optional = np.setdiff1d(np.arange(total), exact_prompt)
    labels = np.repeat(np.arange(3, dtype=np.uint8), lengths)[optional]
    positions = np.concatenate([np.arange(n)+sum(lengths[:s])+s*latent for s,n in enumerate(lengths)])
    return dict(optional=optional, exact_prompt=exact_prompt, labels=labels,
                positions=positions, exact_rows=12+3*latent, n=len(optional), d=d)


def specification(lengths, arm, *, d=128, latent=10, target=PACKET_BYTES):
    info=layout(lengths,d,latent); n=info['n']; exact=info['exact_rows']
    fixed=HEADER.size+exact*d*2
    if arm == 'Global':
        k=min(d,n-1,(target-fixed-2*d)//(2*(n+d)))
        a,b,groups= k,0,1
        used=fixed+2*(d+(n+d)*k)
    else:
        separate=arm == 'Source-Separate'
        common_multiple=0 if separate else 2
        counts=np.bincount(info['labels'],minlength=3)
        cost_per_rank=2*((n+d)*common_multiple+n+3*d)
        k=min(d//(common_multiple+1),int(min(counts)-1),
              (target-fixed-n-6*d)//cost_per_rank)
        a,b,groups=common_multiple*k,k,3
        used=fixed+n+2*(3*d+(n+d)*a+(n+3*d)*b)
    if a<0 or b<0 or used>target or a+b == 0:
        raise ValueError('Budget cannot encode a nonzero direction')
    return dict(shared_rank=int(a),private_rank=int(b),groups=groups,
                content_bytes=int(used),padding_bytes=int(target-used),target_bytes=target,
                n=n,d=d,exact_rows=exact,lengths=list(lengths),latent=latent)


def basis(x, rank):
    """FP64 Gram eigendecomposition with zeroed numerical-null directions."""
    if rank == 0:return x.new_zeros((x.shape[0],x.shape[-1],0))
    e,u=torch.linalg.eigh(x.transpose(-1,-2)@x)
    e=e[...,-rank:].flip(-1);u=u[...,-rank:].flip(-1)
    tol=torch.finfo(x.dtype).eps*max(x.shape[-2:])*e[...,0:1].clamp_min(1e-30)
    return u*(e>tol).unsqueeze(-2)


def bf16_bytes(tensor):
    return tensor.detach().to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy().astype('<u2',copy=False).tobytes()


def encode(x, exact, lengths, arm, *, seed=0, target=PACKET_BYTES):
    """Encode batch=heads, optional-prompt rows only, preserving exact rows."""
    assert x.ndim==3 and x.shape[-1]==128 and torch.isfinite(x).all()
    spec=specification(lengths,arm,target=target)
    assert x.shape[-2]==spec['n'] and exact.shape==(x.shape[0],spec['exact_rows'],128)
    x=x.to(torch.float64)
    labels=layout(lengths)['labels'].copy()
    if arm.startswith('Shuffle-'):
        rng=np.random.default_rng(seed+int(arm.split('-')[1]))
        labels=labels[rng.permutation(len(labels))]
        assert np.array_equal(np.bincount(labels,minlength=3),np.bincount(layout(lengths)['labels'],minlength=3))
    q,r=spec['shared_rank'],spec['private_rank']
    if arm=='Global':
        means=x.mean(-2,keepdim=True);center=x-means
        common=basis(center,q);coeff=center@common
        private=[];private_coeff=[]
    else:
        groups=[torch.as_tensor(np.flatnonzero(labels==s),device=x.device) for s in range(3)]
        means=torch.stack([x[:,ix].mean(-2) for ix in groups],dim=1)
        center=x-means[:,torch.as_tensor(labels.astype(np.int64),device=x.device)]
        common=basis(center,q);coeff=center@common
        residual=center-coeff@common.transpose(-1,-2)
        private=[basis(residual[:,ix],r) for ix in groups]
        private_coeff=[residual[:,ix]@b for ix,b in zip(groups,private)]
    matrices=[means,common,coeff]+[v for pair in zip(private,private_coeff) for v in pair]
    # One host conversion per matrix, not one synchronization per head.
    encoded=[t.to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy().astype('<u2',copy=False) for t in matrices]
    exact_bits=exact.to(torch.bfloat16).cpu().contiguous().view(torch.uint16).numpy().astype('<u2',copy=False)
    values=[1,ARMS.index(arm),128,*lengths,10,q,r,spec['n'],spec['exact_rows'],spec['groups'],spec['content_bytes'],target]
    header=HEADER.pack(MAGIC,*values)
    packets=[]
    for h in range(x.shape[0]):
        body=header+exact_bits[h].tobytes()
        if arm!='Global':body+=labels.tobytes()
        body+=b''.join(a[h].tobytes() for a in encoded)
        assert len(body)==spec['content_bytes'],(len(body),spec)
        packets.append(body+bytes(spec['padding_bytes']))
    return packets,spec


def decode(packet, *, device='cpu'):
    """Decode from bytes only; no access to input tensors or fitted bases."""
    magic,*h=HEADER.unpack_from(packet)
    version,arm_id,d,p,c,r,latent,q,k,n,ex,groups,used,declared_bytes=h
    assert magic==MAGIC and version==1 and declared_bytes==len(packet) and d==128 and latent==10
    arm=ARMS[arm_id];lengths=[p,c,r];spec=specification(lengths,arm,target=len(packet))
    assert (q,k,n,ex,groups,used)==tuple(spec[t] for t in ('shared_rank','private_rank','n','exact_rows','groups','content_bytes'))
    offset=HEADER.size
    def read(shape):
        nonlocal offset
        size=int(np.prod(shape))*2
        a=np.frombuffer(packet,dtype='<u2',count=size//2,offset=offset).copy().reshape(shape)
        offset+=size
        return torch.from_numpy(a).view(torch.bfloat16).to(device=device,dtype=torch.float32)
    exact=read((ex,d))
    if groups==3:
        labels=np.frombuffer(packet,dtype=np.uint8,count=n,offset=offset).copy();offset+=n
        assert np.array_equal(np.bincount(labels,minlength=3),np.bincount(layout(lengths)['labels'],minlength=3))
    else:labels=None
    means=read((groups,d));common=read((d,q));coeff=read((n,q))
    restored=coeff@common.T
    if groups==1:restored+=means[0]
    else:
        restored+=means[torch.as_tensor(labels.astype(np.int64),device=device)]
        for s in range(3):
            ix=torch.as_tensor(np.flatnonzero(labels==s),device=device)
            private=read((d,k));co=read((len(ix),k))
            restored[ix]+=co@private.T
    assert offset==used and not any(packet[used:])
    return restored.to(torch.bfloat16),exact.to(torch.bfloat16),spec


def errors(x,restored):
    x=x.double();restored=restored.double()
    difference=(x-restored).square().sum().item()
    centered=(x-x.mean(-2,keepdim=True)).square().sum().item()
    energy=x.square().sum().item()
    return dict(sse=difference,centered_energy=centered,total_energy=energy)
