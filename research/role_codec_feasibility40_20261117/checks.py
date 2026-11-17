"""Independent numerical and serialization checks; no model construction."""
import json
import numpy as np
import torch
from codec import encode,decode,layout,specification,basis,ARMS,PACKET_BYTES


def main():
    torch.set_num_threads(1);torch.manual_seed(34)
    lengths=[80,95,90];info=layout(lengths);n=info['n'];checks=[]
    x=torch.randn(2,n,128).to(torch.bfloat16)
    exact=torch.randn(2,42,128).to(torch.bfloat16)
    old=x.clone();old_exact=exact.clone()
    for arm in ARMS:
        packet,spec=encode(x,exact,lengths,arm,seed=123)
        assert len(packet)==2
        for i,p in enumerate(packet):
            restored,protected,out=decode(p)
            assert len(p)==PACKET_BYTES and torch.equal(protected,exact[i])
            assert torch.isfinite(restored).all() and out==spec
        assert torch.equal(x,old) and torch.equal(exact,old_exact)
    checks.append('all_arms_exact_bytes_protected_bits_and_no_input_mutation')
    # Optimal global subspace agrees with direct SVD, independently of Gram path.
    z=torch.randn(2,160,128,dtype=torch.float64);u=basis(z,11)
    _,_,vh=torch.linalg.svd(z,full_matrices=False);v=vh[:,:11].transpose(-1,-2)
    assert torch.max(torch.abs(u@u.transpose(-1,-2)-v@v.transpose(-1,-2)))<1e-9
    checks.append('gram_projector_matches_independent_svd')
    # A rank-two matrix must survive the whole BF16 wire path at small error.
    z=(torch.randn(2,n,2)@torch.randn(2,2,128)+torch.randn(2,1,128)).bfloat16()
    packets,_=encode(z,exact,lengths,'Global')
    y=torch.stack([decode(p)[0] for p in packets])
    relative=torch.linalg.vector_norm((z-y).float())/torch.linalg.vector_norm(z.float())
    assert relative<.02,float(relative)
    checks.append('known_low_rank_roundtrip_error_below_two_percent')
    # Full-rank data should not mysteriously reconstruct exactly at low budget.
    packet,_=encode(x,exact,lengths,'Global');y=torch.stack([decode(p)[0] for p in packet])
    assert torch.linalg.vector_norm((x-y).float())/torch.linalg.vector_norm(x.float())>.2
    checks.append('random_full_rank_not_falsely_lossless')
    state=torch.get_rng_state().clone()
    a,_=encode(x,exact,lengths,'Shuffle-42',seed=123)
    b,_=encode(x,exact,lengths,'Shuffle-42',seed=123)
    assert a==b and torch.equal(state,torch.get_rng_state())
    checks.append('shuffle_deterministic_and_torch_rng_unchanged')
    try:decode(a[0][:-1])
    except (AssertionError,ValueError):pass
    else:raise AssertionError('truncated packet accepted')
    checks.append('truncated_packet_rejected')
    print(json.dumps(dict(passed=True,checks=checks,torch=torch.__version__)))


if __name__=='__main__':main()
