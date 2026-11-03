import unittest
from management_identity import guarded_identity

class IdentityGuard(unittest.TestCase):
    def test_success_preserves_identity(self):
        expected=dict(pid=42,uid=1033,start_ticks='123',argv=['python'],CUDA_VISIBLE_DEVICES='1')
        self.assertIs(guarded_identity(lambda p:expected,42,lambda p:1033,1033),expected)
    def test_permission_denied_is_not_owned(self):
        def denied(pid):raise PermissionError(13,'denied')
        self.assertIsNone(guarded_identity(denied,42,lambda p:1033,1033))
    def test_other_uid_does_not_read_environment(self):
        def forbidden(pid):raise AssertionError('must not inspect foreign environment')
        self.assertIsNone(guarded_identity(forbidden,42,lambda p:999,1033))
    def test_unrelated_error_not_hidden(self):
        def broken(pid):raise RuntimeError('unexpected failure')
        with self.assertRaises(RuntimeError):guarded_identity(broken,42,lambda p:1033,1033)

if __name__=='__main__':unittest.main()
