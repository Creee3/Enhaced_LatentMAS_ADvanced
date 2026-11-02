"""CPU-only mechanics of the EXACT historical selector; no inference claim."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'snapshot/run'))
try:
    import torch
    import stage_ops
except ImportError:
    torch=None

@unittest.skipIf(torch is None,'PyTorch required for tensor mechanics')
class ReceiverMechanics(unittest.TestCase):
    def test_three_handoffs_preserve_original_pairs_and_positions(self):
        original_factory=stage_ops.make_cache
        stage_ops.make_cache=lambda xs:tuple(xs)
        try:
            for steps in (10,40):
                with self.subTest(steps=steps):
                    cache=();mapping=None;absolute=0
                    for i,role in enumerate(('planner','critic','refiner'),1):
                        prompt=64;before=cache[0][0].shape[-2] if cache else 0
                        xs=[];scores=[]
                        for layer in range(2):
                            fresh=torch.arange(absolute,absolute+prompt+steps,dtype=torch.float64).reshape(1,1,-1,1)
                            k=fresh.expand(1,2,-1,4).clone()+layer*10000
                            v=k+50000
                            if cache:k=torch.cat([cache[layer][0],k],2);v=torch.cat([cache[layer][1],v],2)
                            xs.append((k,v));score=torch.zeros(2,k.shape[-2])
                            for head in range(2):
                                # Different layers/heads need different top rows.
                                start=4+4*layer+head*3
                                score[head,before+start:before+start+24]=10
                            scores.append(score)
                        previous=mapping.clone() if mapping is not None else None
                        cache,mapping,receipt=stage_ops.compress(tuple(xs),mapping,role=role,steps=steps,prompt_len=prompt,original_before=absolute,method='receiver_attention_sink_recent',attention=dict(kind='receiver_probe',layers=scores))
                        self.assertEqual(receipt['retained'],4+i*(32+steps))
                        self.assertFalse(receipt['value_backfill'])
                        self.assertFalse(receipt['source_v_equal']) # historical label only
                        self.assertTrue(torch.all(mapping.diff(dim=-1)>0))
                        if previous is not None:self.assertTrue(torch.equal(mapping[:,:,:before],previous))
                        self.assertTrue(torch.all(torch.isin(torch.arange(4),mapping[0,0])))
                        for layer,(k,v) in enumerate(cache):
                            self.assertTrue(torch.equal(k[0,:,:,0]-layer*10000,mapping[layer].to(k.dtype)))
                            self.assertTrue(torch.equal(v,k+50000))
                            for head in range(2):
                                required=torch.arange(absolute+prompt,absolute+prompt+steps)
                                self.assertTrue(torch.all(torch.isin(required,mapping[layer,head])))
                                recent=torch.arange(absolute+prompt-8,absolute+prompt)
                                self.assertTrue(torch.all(torch.isin(recent,mapping[layer,head])))
                        self.assertFalse(torch.equal(mapping[0,0],mapping[1,0]))
                        self.assertEqual(receipt['retained_tensor_bytes'],sum(t.numel()*t.element_size() for pair in cache for t in pair))
                        absolute+=prompt+steps
        finally:stage_ops.make_cache=original_factory

if __name__=='__main__':unittest.main()
