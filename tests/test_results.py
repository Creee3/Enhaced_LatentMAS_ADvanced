import csv
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from reproduce_results import rebuild,check

class ResultChecks(unittest.TestCase):
    def test_exact_coverage_and_sixteen_cells(self):
        cells=rebuild();check(cells)
        self.assertEqual(len(cells),16)
        self.assertEqual(next(x['correct'] for x in cells if (x['scope'],x['latent_steps'],x['method'])==('main448',10,'receiver_attention_sink_recent')),228)

    def mutated(self,duplicate):
        source=ROOT/'results/gpqa_main448/case_metrics.csv'
        with source.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        changed=rows+[rows[0]] if duplicate else rows[1:]
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'metrics.csv'
            with p.open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(changed)
            with self.assertRaisesRegex(ValueError,'Duplicate' if duplicate else 'Missing'):rebuild(p)

    def test_duplicate_rejected(self):self.mutated(True)
    def test_missing_rejected(self):self.mutated(False)

if __name__=='__main__':unittest.main()
