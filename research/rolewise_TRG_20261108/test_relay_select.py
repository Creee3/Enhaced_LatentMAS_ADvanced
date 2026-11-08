import sys,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'role_blocks_main448_20261102'),str(HERE.parent/'attention_control225_20261103')]
import torch
from relay_select import select_relay,masks
from block_selector import components,select

class RelayMechanics(unittest.TestCase):
 def test_three_handoffs_original_gather_and_source_provenance(self):
  torch.manual_seed(103);layers=2;heads=2;dim=3
  for mode in ['G','R','T']:
   mapping=None;xs=None;stages=[];absolute=0
   for hop in range(3):
    p=112;fresh=tuple((torch.randn(1,heads,p+10,dim),torch.randn(1,heads,p+10,dim)) for _ in range(layers))
    if xs is not None:xs=tuple(tuple(torch.cat([a,b],2) for a,b in zip(old,new)) for old,new in zip(xs,fresh))
    else:xs=fresh
    stages.append(dict(original_before=absolute,original_end=absolute+p+10,prompt_len=p));absolute+=p+10;length=xs[0][0].shape[-2];before=tuple(tuple(t.clone() for t in pair) for pair in xs)
    captured=[(torch.rand(heads,4,length+5).softmax(-1),torch.cat([pair[1][0],torch.randn(heads,5,dim)],1)) for pair in xs]
    xs,mapping,ix,reports,proof=select_relay(xs,mapping,stages,captured,mode)
    self.assertEqual(mapping.shape,(layers,heads,110+10*hop));self.assertEqual(proof['retained'],110+10*hop)
    for l,pair in enumerate(before):
     for a,b in zip(pair,xs[l]):self.assertTrue(torch.equal(b,torch.gather(a,2,ix[l][None,:,:,None].expand_as(b))))
    self.assertTrue(bool((mapping<absolute).all()))
    for l in range(layers):
     labels,protected,candidates,mandatory,k=masks(mapping[l],stages,mapping.shape[-1]+5)
     self.assertEqual(k,96);self.assertTrue(bool((labels[:,-5:]==0).all()));self.assertTrue(bool(protected[:,-5:].all()))
     self.assertTrue(bool(((mapping[l]>=absolute-18)&(mapping[l]<absolute-10)).sum(-1).eq(8).all()))
 def test_short_and_zero_candidate_no_fake_rows(self):
  for prompt,expected in [(4,14),(9,19),(25,35)]:
   stage=[dict(original_before=0,original_end=prompt+10,prompt_len=prompt)];length=prompt+10
   xs=((torch.randn(1,2,length,3),torch.randn(1,2,length,3)),);captured=[(torch.rand(2,3,length+4).softmax(-1),torch.randn(2,length+4,3))]
   for mode in ['G','R','T']:
    new,pos,ix,_,rec=select_relay(xs,None,stage,captured,mode);self.assertEqual(rec['retained'],expected);self.assertTrue(torch.equal(pos[0,0],torch.arange(length)))
 def test_input_cache_not_mutated(self):
  xs=((torch.randn(1,2,150,3),torch.randn(1,2,150,3)),);saved=[x.clone() for x in xs[0]];stage=[dict(original_before=0,original_end=150,prompt_len=140)]
  cap=[(torch.rand(2,4,155).softmax(-1),torch.randn(2,155,3))]
  for mode in ['T','R','G']:select_relay(xs,None,stage,cap,mode)
  self.assertTrue(all(torch.equal(x,y) for x,y in zip(xs[0],saved)))
 def test_closed_form_first_hop_with_self_matches_bruteforce(self):
  torch.manual_seed(45);a=torch.rand(2,5,9).softmax(-1);v=torch.randn(2,9,3);labels=torch.tensor([[0,1,1,1,1,1,0,0,0]]*2);protected=labels==0;candidates=~protected;mandatory=torch.zeros_like(protected)
  for mode in ['T','R']:
   actual,_=select(a,v,labels,protected,candidates,mandatory,4,mode);target=components(a.double(),v.double(),labels,torch.ones_like(protected));expected=torch.ones_like(protected)
   for h in range(2):
    loss=[]
    for j in range(1,6):
     trial=expected.clone();trial[h,j]=False;err=target-components(a.double(),v.double(),labels,trial)
     if mode=='T':err=err.sum(1,keepdim=True)
     loss.append(float(err[h].square().mean()))
    expected[h,1+min(range(5),key=loss.__getitem__)]=False
   self.assertTrue(torch.equal(actual,expected))
if __name__=='__main__':unittest.main()
