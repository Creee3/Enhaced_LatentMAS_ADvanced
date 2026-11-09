import unittest
from resource_policy import admit
class Policy(unittest.TestCase):
 def test_total_cap(self):
  self.assertFalse(admit(live_total=6,live_gpu=2,all_ready=True,blocked=False,util=0,free=80000,total=81920,pending=True))
 def test_initial(self):
  self.assertTrue(admit(live_total=0,live_gpu=0,all_ready=True,blocked=False,util=0,free=80000,total=81920,pending=True))
 def test_busy_first(self):
  self.assertFalse(admit(live_total=0,live_gpu=0,all_ready=True,blocked=False,util=99,free=80000,total=81920,pending=True))
