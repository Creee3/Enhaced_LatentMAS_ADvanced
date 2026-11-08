import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from resource_policy import admit,cpu_set
from common import choose
class StatePolicy(unittest.TestCase):
 def test_six_total_and_four_per_card_caps(self):
  base=dict(live_total=5,live_gpu=3,all_ready=True,blocked=False,util=100,free=23000,total=81920,pending=True)
  self.assertTrue(admit(**base))
  for change in [dict(live_total=6),dict(live_gpu=4),dict(all_ready=False),dict(blocked=True),dict(pending=False),dict(free=21000)]:self.assertFalse(admit(**(base|change)))
 def test_first_model_has_fresh_gate(self):
  base=dict(live_total=0,live_gpu=0,all_ready=True,blocked=False,util=30,free=60000,total=81920,pending=True);self.assertTrue(admit(**base))
  self.assertFalse(admit(**(base|dict(util=31))));self.assertFalse(admit(**(base|dict(free=57000))))
 def test_no_complete_or_held_requeue_and_engineering_scope(self):
  q=dict(engineering_indices=[1],tasks=[dict(index=0,state='complete',methods=[]),dict(index=1,state='held',methods=['RRR']),dict(index=2,state='pending',methods=['TTT'])])
  self.assertIsNone(choose(q,'engineering'));self.assertEqual(choose(q,'formal')['index'],2)
  q['tasks'][1]['state']='pending';self.assertEqual(choose(q,'engineering')['methods'],['RRR'])
 def test_dynamic_cpuset(self):self.assertEqual(cpu_set('2,5-7,10'),[2,5,6,7,10])
if __name__=='__main__':unittest.main()
