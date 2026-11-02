import unittest
import torch
from block_selector import components, select, shifted_blocks

class Mechanics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.a=torch.rand(2,7,22,dtype=torch.float64).softmax(-1)
        self.v=torch.randn(2,22,5,dtype=torch.float64)
        self.labels=torch.tensor([[0]*2+[1]*6+[2]*6+[3]*6+[0]*2]*2)
        self.protected=torch.zeros(2,22,dtype=torch.bool)
        self.protected[:,:2]=True; self.protected[:,-2:]=True
        self.candidate=~self.protected
        self.mandatory=torch.zeros_like(self.protected);self.mandatory[:,18:20]=True

    def test_gather_budget_and_no_mutation(self):
        before=[x.clone() for x in (self.a,self.v,self.labels,self.protected)]
        for mode in ['T','R','R-shuffle']:
            keep,rec=select(self.a,self.v,self.labels,self.protected,self.candidate,self.mandatory,8,mode)
            self.assertTrue(bool(keep[self.protected|self.mandatory].all()))
            self.assertEqual(keep.sum(-1).tolist(),[12,12])
            self.assertGreaterEqual(rec['objective_mse'],0)
        for x,y in zip(before,(self.a,self.v,self.labels,self.protected)):self.assertTrue(torch.equal(x,y))

    def test_closed_form_ranking_matches_bruteforce(self):
        # All18 candidates in bank. Verify sequential numerator/denominator updates.
        for mode in ['T','R','R-shuffle']:
            role=shifted_blocks(self.labels) if mode=='R-shuffle' else self.labels
            full=torch.ones_like(self.protected);target=components(self.a,self.v,role,full)
            for budget in [17,16]:
                keep,_=select(self.a,self.v,self.labels,self.protected,self.candidate,self.mandatory,budget,mode)
                expected=full.clone()
                for _ in range(18-budget):
                    for h in range(2):
                        available=[j for j in range(2,18) if expected[h,j]]
                        losses=[]
                        for j in available:
                            trial=expected.clone();trial[h,j]=False
                            err=target-components(self.a,self.v,role,trial)
                            if mode=='T':err=err.sum(1,keepdim=True)
                            losses.append(float(err[h].square().mean()))
                        deleted=available[min(range(len(losses)),key=losses.__getitem__)]
                        expected[h,deleted]=False
                self.assertTrue(torch.equal(keep,expected))

    def test_self_changes_under_renormalization(self):
        full=torch.ones_like(self.protected);trial=full.clone();trial[:,2]=False
        self.assertFalse(torch.equal(components(self.a,self.v,self.labels,full)[:,0],
                                     components(self.a,self.v,self.labels,trial)[:,0]))

    def test_pseudo_not_a_global_permutation(self):
        fake=shifted_blocks(self.labels)
        for h in range(2):
            self.assertTrue(torch.equal(torch.bincount(fake[h]),torch.bincount(self.labels[h])))
            for r in [1,2,3]:self.assertGreater(torch.unique(fake[h,self.labels[h]==r]).numel(),1)

if __name__=='__main__':unittest.main()
