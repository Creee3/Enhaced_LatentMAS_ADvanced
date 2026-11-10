"""Full stack selector/derived-readout integration, synthetic tensors only."""
import sys,unittest
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent
# Fixed old source contains block_selector/attention_selector; never edit it.
ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025')
SCIENCE=ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z'
if SCIENCE.exists():sys.path.append(str(SCIENCE))
else:
 local=HERE.parents[2]/'advanced_MOC/experiments/latentmas_TRG225_a100_20261106/completed_verified_0e54558e885e/run'
 sys.path.append(str(local))
from method_ops import select_method
from diagnostics import readout
from relay_select import masks

class Cache:
 def __init__(self,xs):self.xs=tuple(xs)
 def get_seq_length(self):return self.xs[0][0].shape[-2]
def pairs(cache):return cache.xs if hasattr(cache,"xs") else tuple(cache)
def make_cache(xs):return tuple(xs)
class MethodIntegration(unittest.TestCase):
 def test_all12_same_state_original_gather_and_readout(self):
  torch.manual_seed(907);layers=2;heads=2;n=174
  xs=tuple((torch.randn(1,heads,n,128).bfloat16(),torch.randn(1,heads,n,128).bfloat16()) for _ in range(layers));cache=Cache(xs)
  stages=[dict(role='planner',prompt_len=164,original_before=0,original_end=n)]
  sender=[[torch.rand(1,heads*4,164+i+1) for _ in range(layers)] for i in range(10)]
  captured=[(torch.rand(heads,12,n+17).softmax(-1),torch.cat([v[0].float(),torch.randn(heads,17,128)],1)) for _,v in xs]
  pos=torch.arange(n)[None,None,:].expand(layers,heads,-1)
  labels=torch.stack([masks(pos[l],stages,n+17)[0] for l in range(layers)])
  original=[(k.clone(),v.clone()) for k,v in xs]
  for mode in ['Full','Gen','H','L','H-OBF','L-OBF','G','T','R','R-shuffle','Pseudo-R','Random-matched']:
   out,positions,ix,report=select_method(cache,None,stages,mode,sender,captured,42,pairs,make_cache)
   arrays=readout(captured,ix,pairs(out),labels,n)
   self.assertEqual(report['retained'],positions.shape[-1]);self.assertTrue((arrays['total_error_mse']>=0).all())
   if mode=='Full':self.assertTrue((arrays['total_error_mse']==0).all())
   for (k,v),(a,b) in zip(xs,original):self.assertTrue(torch.equal(k,a) and torch.equal(v,b))
if __name__=='__main__':unittest.main()
