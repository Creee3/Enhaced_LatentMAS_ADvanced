import unittest,numpy as np
from measure import describe,overlap,compare,role_effect,windows,sample_indices
from prompts import instruction_candidates,ROLES
class Mechanics(unittest.TestCase):
 def test_centered_rank_cap(self):
  x=np.random.default_rng(42).normal(size=(2,10,128));m,b,r=describe(x);self.assertTrue((m[:,-1]<=9).all());self.assertEqual(b.shape,(2,4,128))
 def test_self_overlap(self):
  x=np.random.default_rng(42).normal(size=(3,32,128));m,b,r=describe(x);np.testing.assert_allclose(overlap(b,r,b,r),1,atol=1e-12)
 def test_zero_direction_missing(self):
  m,b,r=describe(np.zeros((2,10,128)));self.assertTrue((m==0).all());self.assertTrue(np.isnan(overlap(b,r,b,r)).all())
 def test_pair_types(self):
  rng=np.random.default_rng(42);x=[rng.normal(size=(2,32,128)) for _ in range(6)];p=compare(x,[describe(z) for z in x]);self.assertEqual(len(p),15);self.assertEqual(sum(z['same_role'] for z in p),3);self.assertEqual(role_effect(p)['between_minus_within'].shape,(2,))
 def test_window_matching(self):
  for n in [32,96,101,300]:
   w=windows(n);self.assertEqual(len(set(map(len,w))),1);self.assertEqual(len(set(w[0])&set(w[1])),0);self.assertEqual(len(set(w[1])&set(w[2])),0)
 def test_sampling(self):
  self.assertEqual(len(sample_indices(300,32)),32);self.assertEqual(len(set(sample_indices(300,32))),32)
 def test_no_gold_or_answer(self):
  c=instruction_candidates();self.assertEqual(set(c),set(ROLES));self.assertTrue(all(len(set(v))==4 for v in c.values()))
if __name__=='__main__':unittest.main()
