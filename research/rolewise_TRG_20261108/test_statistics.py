import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from score import exact_mcnemar,wilson
class Statistics(unittest.TestCase):
 def test_exact_discordant_pairs(self):
  r=exact_mcnemar([True,True,True,False],[False,False,False,True]);self.assertEqual(r['first_only'],3);self.assertEqual(r['second_only'],1);self.assertEqual(r['two_sided_exact_p'],.625)
 def test_identical_results_not_equivalence_claim(self):self.assertEqual(exact_mcnemar([True,False],[True,False])['two_sided_exact_p'],1)
 def test_wilson_bounds_and_symmetry(self):
  lo,hi=wilson(50,100);self.assertLess(lo,.5);self.assertGreater(hi,.5);self.assertAlmostEqual(lo,1-hi);self.assertGreater(wilson(0,100)[1],0)
if __name__=='__main__':unittest.main()
