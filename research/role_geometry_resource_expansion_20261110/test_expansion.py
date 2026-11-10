import unittest
from expansion_manager_v2 import same_identity,admit_addition
class Guards(unittest.TestCase):
 def test_reused_pid(self):
  x=dict(pid=1,uid=1000,start_ticks='1',argv=['worker'],CUDA_VISIBLE_DEVICES='gpu');self.assertTrue(same_identity(x,x));self.assertFalse(same_identity(dict(x,start_ticks='2'),x));self.assertFalse(same_identity(None,x))
 def test_gpu0(self):
  self.assertTrue(self.decision(live_total=6,live_gpu=2,max_gpu=3,util=100,free=47000))
 def test_gpu3_busy(self):
  self.assertFalse(self.decision(live_gpu=0,max_gpu=1,util=39,free=61000))
 def test_gpu3_free(self):
  self.assertTrue(self.decision(live_gpu=0,max_gpu=1,util=30,free=61000))
 def test_limits(self):
  self.assertFalse(self.decision(live_total=8));self.assertFalse(self.decision(live_gpu=3,max_gpu=3));self.assertFalse(self.decision(all_ready=False));self.assertFalse(self.decision(blocked=True));self.assertFalse(self.decision(free=21000))
 def decision(self,**kw):
  args=dict(live_total=6,live_gpu=2,max_gpu=3,all_ready=True,blocked=False,util=0,free=47000,total=81920);args.update(kw);return admit_addition(**args)
if __name__=='__main__':unittest.main()
