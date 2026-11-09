"""Exercise pinned original receiver across three real coordinate handoffs."""
import sys,unittest
import torch
from common import SCIENCE
sys.path.insert(0,str(SCIENCE))
from stage_ops import compress,make_cache,pairs
from worker import relay
class Relay(unittest.TestCase):
 def run_chain(self,mode):
  cache=None;mapping=None;absolute=0;lengths=[]
  for role in ['planner','critic','refiner']:
   plen=80;old=pairs(cache) if cache is not None else tuple((torch.empty(1,8,0,128,dtype=torch.bfloat16),torch.empty(1,8,0,128,dtype=torch.bfloat16)) for _ in range(36))
   xs=tuple(tuple(torch.cat([t,torch.arange(absolute,absolute+plen+10,dtype=torch.bfloat16).view(1,1,-1,1).expand(1,8,-1,128)],2) for t in pair) for pair in old)
   cache=make_cache(xs);rec=dict(role=role,prompt_len=plen,original_before=absolute,original_end=absolute+plen+10)
   # Native prompt_scores consumes per-layer head scores over all cache positions.
   attention=dict(kind='receiver_probe',layers=[torch.ones(32,cache.get_seq_length()) for _ in range(36)])
   cache,mapping,ind,receipt=relay(cache,mapping,rec,mode,attention,compress,pairs,torch)
   lengths.append(receipt['retained']);absolute+=plen+10
  return lengths
 def test_full_exact_all_rows(self):self.assertEqual(self.run_chain('Full'),[90,180,270])
 def test_original_receiver_budget(self):self.assertEqual(self.run_chain('receiver'),[46,88,130])
