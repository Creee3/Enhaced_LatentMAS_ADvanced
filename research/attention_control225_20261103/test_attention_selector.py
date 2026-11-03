"""CPU-only mechanics: ties, GQA observations, protection, budget and V independence."""
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'role_blocks_main448_20261102'))
from attention_selector import select


class SelectorTests(unittest.TestCase):
    def fixture(self):
        a=torch.full((2,64,12), 1/12, dtype=torch.double)
        v=torch.arange(2*12*3,dtype=torch.double).reshape(2,12,3)
        labels=torch.zeros((2,12),dtype=torch.long)
        labels[:,2:]=1
        protected=torch.zeros((2,12),dtype=torch.bool);protected[:,:2]=True
        candidates=~protected
        mandatory=torch.zeros_like(protected);mandatory[:,-1]=True
        return a,v,labels,protected,candidates,mandatory

    def test_ties_protection_and_no_mutation(self):
        args=self.fixture();before=[x.clone() for x in args]
        selected,report=select(*args,4,'G')
        self.assertEqual(torch.where(selected[0])[0].tolist(),[0,1,2,3,4,11])
        self.assertEqual(report['bank_rows'],8)
        for x,y in zip(args,before):self.assertTrue(torch.equal(x,y))

    def test_all_observations_and_v_independence(self):
        args=list(self.fixture());a=args[0]
        a[:,:,5]=.2;a[:,:,6]=.19;a[:,:,7]=.18
        # One of the 64 distinct observations overturns the last free choice.
        a[:,:,8]=.179;a[:,63,8]=1.5
        a/=a.sum(-1,keepdim=True)
        selected,_=select(*args,4,'G')
        self.assertTrue(bool(selected[:,8].all()))
        args[1]=-args[1].flip(-1)*100
        other,_=select(*args,4,'G')
        self.assertTrue(torch.equal(selected,other))

    def test_mandatory_overrides_attention_and_small_bank(self):
        args=list(self.fixture());args[0][:,:,-1]=0
        selected,report=select(*args,8,'G')
        self.assertTrue(bool(selected[:,-1].all()))
        self.assertTrue(bool((selected.sum(-1)==10).all()))
        self.assertEqual(report['bank_rows'],10)

    def test_invalid_partition_rejected(self):
        args=list(self.fixture());args[4][:,0]=True
        with self.assertRaises(AssertionError):select(*args,4,'G')


if __name__=='__main__':unittest.main()
