import unittest
import torch
from role_contrast import responses,distortion,select


class RoleContrastTests(unittest.TestCase):
    def test_total_output_can_hide_role_difference(self):
        a=torch.tensor([[.2,.2,.1,.1,.4]],dtype=torch.float64)
        v=torch.tensor([[1.,0.],[-1.,0.],[0.,0.],[0.,0.],[0.,0.]],dtype=torch.float64)
        ids=torch.tensor([0,1,0,1,-1])
        all_rows=torch.ones(5,dtype=torch.bool)
        reference=responses(a,v,ids,all_rows,2)
        selected=responses(a,v,ids,torch.tensor([False,False,True,True,True]),2)
        total_loss,_,_=distortion(reference,selected,'total')
        role_loss,_,_=distortion(reference,selected,'role_contrast')
        self.assertEqual(float(total_loss),0.)
        self.assertGreater(float(role_loss),0.)

    def test_budget_protection_and_original_values(self):
        g=torch.Generator().manual_seed(42)
        a=torch.softmax(torch.randn(6,20,generator=g,dtype=torch.float64),-1)
        v=torch.randn(20,8,generator=g,dtype=torch.float64)
        original_a=a.clone();original_v=v.clone()
        ids=torch.tensor([-1,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,-1,-1])
        fixed=torch.zeros(20,dtype=torch.bool);fixed[[0,1,7,13,18,19]]=True
        pool=[i for i in range(20) if not fixed[i]]
        for objective in ('total','role_contrast'):
            ix,receipt=select(a,v,ids,fixed,pool,6,objective=objective,mandatory=[17])
            self.assertEqual(len(ix),6);self.assertEqual(len(set(ix.tolist())),6)
            self.assertIn(17,ix.tolist());self.assertTrue(all(not fixed[i] for i in ix))
            self.assertEqual(receipt['protected_rows'],6)
            self.assertEqual(sum(receipt['selected_per_role'].values()),6)
        self.assertTrue(torch.equal(a,original_a));self.assertTrue(torch.equal(v,original_v))

    def test_one_role_has_no_contrast_term(self):
        a=torch.tensor([[.3,.3,.4]],dtype=torch.float64)
        v=torch.tensor([[1.],[2.],[0.]],dtype=torch.float64)
        ids=torch.tensor([0,0,-1]);fixed=torch.tensor([False,False,True])
        i1,r1=select(a,v,ids,fixed,[0,1],1,objective='total')
        i2,r2=select(a,v,ids,fixed,[0,1],1,objective='role_contrast')
        self.assertTrue(torch.equal(i1,i2));self.assertEqual(r2['contrast_error'],0.)

    def test_protected_row_cannot_enter_deletion_pool(self):
        with self.assertRaisesRegex(ValueError,'prompt pool'):
            select(torch.tensor([[.5,.5]]),torch.ones(2,2),torch.tensor([0,0]),torch.tensor([True,False]),[0,1],1)


if __name__=='__main__':unittest.main()
