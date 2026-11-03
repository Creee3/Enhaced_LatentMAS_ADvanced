"""CPU-only: a broken GPU must not crash or contaminate admission of another."""
import unittest
from types import SimpleNamespace
from gpu_probe import collect


class ProbeTests(unittest.TestCase):
    def test_per_card_failure_isolation(self):
        def run(argv,**kw):
            return SimpleNamespace(returncode=1,stdout='Unknown Error',stderr='') if argv[2]=='0' else SimpleNamespace(returncode=0,stdout='2, GPU-abc, 22, 20252, 24564\n',stderr='')
        rows,errors=collect([0,2],run)
        self.assertEqual([r['gpu'] for r in rows],[2]);self.assertEqual(list(errors),[0])

    def test_wrong_index_or_missing_telemetry_rejected(self):
        rows,errors=collect([7],lambda *a,**kw:SimpleNamespace(returncode=0,stdout='0, GPU-abc, N/A, 21000, 24564',stderr=''))
        self.assertEqual(rows,[]);self.assertIn(7,errors)


if __name__=='__main__':unittest.main()
