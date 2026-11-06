"""Synthetic-only tests: pairing/coverage and exact test edge cases."""
import unittest
from final_summary import exact_mcnemar,coverage
class SummaryTests(unittest.TestCase):
 def test_no_discordance(self):
  x=exact_mcnemar([True,False],[True,False]);self.assertEqual(x['two_sided_exact_p'],1);self.assertFalse(x['first_net_improvement_and_significant'])
 def test_one_sided_discordance_is_two_sided_test(self):
  x=exact_mcnemar([True]*6,[False]*6);self.assertEqual(x['two_sided_exact_p'],.03125);self.assertTrue(x['first_net_improvement_and_significant'])
  y=exact_mcnemar([False]*6,[True]*6);self.assertEqual(y['two_sided_exact_p'],.03125);self.assertFalse(y['first_net_improvement_and_significant'])
 def test_balanced_discordance_not_equivalence(self):
  x=exact_mcnemar([True,False],[False,True]);self.assertEqual(x['two_sided_exact_p'],1);self.assertEqual(x['net_correct_difference'],0)
 def test_missing_duplicate_and_wrong_key_refused(self):
  q=[dict(dataset='synthetic',question_id='case')];r=[dict(dataset='synthetic',question_id='case',latent_steps=10,method=m) for m in ['T','R','G']]
  self.assertEqual(len(coverage(r,q)),3)
  for bad in [r[:-1],r+[r[0]],[dict(x,latent_steps=40) for x in r]]:
   with self.assertRaises(AssertionError):coverage(bad,q)
if __name__=='__main__':unittest.main()
