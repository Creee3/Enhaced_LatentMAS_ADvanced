import unittest
import torch

from controls import pseudo_source_labels, r_shuffle_labels, random_matched_keep


class ControlMechanics(unittest.TestCase):
    def test_shuffle_preserves_counts_and_sink(self):
        x = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 3, 3]])
        y = r_shuffle_labels(x)
        self.assertTrue(torch.equal(torch.bincount(x.reshape(-1), minlength=4), torch.bincount(y.reshape(-1), minlength=4)))
        self.assertTrue(torch.equal(x[x == 0], y[x == 0]))

    def test_pseudo_shuffle_is_bin_local(self):
        x = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0]])
        bins = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 3]])
        y = pseudo_source_labels(x, bins)
        self.assertTrue(torch.equal(x[bins == 0], y[bins == 0]))
        self.assertTrue(torch.equal(torch.bincount(x.reshape(-1), minlength=4), torch.bincount(y.reshape(-1), minlength=4)))

    def test_random_matched_preserves_budget_and_mandatory(self):
        candidates = torch.ones((3, 12), dtype=torch.bool)
        mandatory = torch.zeros_like(candidates); mandatory[:, -2:] = True
        y = random_matched_keep(candidates, mandatory, budget=6, seed=42)
        self.assertTrue(bool((y[:, -2:]).all()))
        self.assertTrue(torch.equal(y.sum(-1), torch.full((3,), 6)))


if __name__ == "__main__":
    unittest.main()
