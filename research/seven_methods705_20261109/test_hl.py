"""CPU-only real-shape three-hop author H/L checks; no model or dataset labels."""
import sys,unittest
import torch
from common import SCIENCE
sys.path.insert(0,str(SCIENCE))
from stage_ops import pairs,make_cache
from hl_ops import load_author,relay_hl

class HL(unittest.TestCase):
 def chain(self,mode):
  obj,_=load_author(mode,make_cache);cache=None;mapping=None;absolute=0;lengths=[];head_difference=False
  for role in ['planner','critic','refiner']:
   plen=80;history=0 if cache is None else cache.get_seq_length()
   old=pairs(cache) if cache is not None else tuple((torch.empty(1,8,0,128,dtype=torch.bfloat16),torch.empty(1,8,0,128,dtype=torch.bfloat16)) for _ in range(36))
   fresh=torch.arange(absolute,absolute+plen+10,dtype=torch.bfloat16).view(1,1,-1,1).expand(1,8,-1,128)
   cache=make_cache(tuple(tuple(torch.cat([t,fresh],2) for t in pair) for pair in old))
   attention=[]
   for step in range(10):
    n=history+plen+step+1;score=torch.zeros(1,32,n)
    for head in range(8):
     for j in range(plen):score[:,head*4:(head+1)*4,history+j]=float(100-((j-head*5)%plen))
    score=score.softmax(-1);attention.append([score.clone() for _ in range(36)])
   rec=dict(role=role,prompt_len=plen,original_before=absolute,original_end=absolute+plen+10)
   before=mapping.clone() if mapping is not None else None
   cache,mapping,ix,receipt=relay_hl(cache,mapping,rec,mode,attention,obj,pairs,make_cache)
   if before is not None:self.assertTrue(torch.equal(mapping[...,:history],before))
   self.assertTrue(receipt['source_k_equal_measured'] and receipt['source_v_equal_measured'])
   self.assertTrue(bool(((mapping<4).sum(-1)==4).all()))
   head_difference|=not torch.equal(mapping,mapping[:,:1].expand_as(mapping))
   lengths.append(receipt['retained']);absolute+=plen+10
  self.assertEqual(lengths,[46,88,130])
  if mode=='H':self.assertTrue(head_difference)
  else:self.assertFalse(head_difference)
 def test_author_headwise_chain(self):self.chain('H')
 def test_author_layerwise_chain(self):self.chain('L')
