import unittest

import torch

from controls import pseudo_source_labels, r_shuffle_labels, random_matched_keep, stable_seed


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor([[0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
                                    [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]])

    def test_shuffle_exact_per_head_reproducible_and_not_role_rename(self):
        original = self.labels.clone()
        output, report = r_shuffle_labels(self.labels, seed=7, return_report=True)
        self.assertTrue(torch.equal(output, r_shuffle_labels(self.labels, seed=7)))
        self.assertTrue(torch.equal(self.labels, original))
        self.assertTrue(torch.equal(output[self.labels == 0], self.labels[self.labels == 0]))
        for h in range(2):
            self.assertTrue(torch.equal(torch.bincount(output[h]), torch.bincount(self.labels[h])))
            self.assertTrue(any(torch.unique(output[h, self.labels[h] == role]).numel() > 1 for role in (1, 2, 3)))
        self.assertEqual(report['degenerate_by_head'], [False, False])

    def test_shuffle_single_role_and_singletons_explicit_degenerate(self):
        for labels in (torch.tensor([[0, 1, 1, 1]]), torch.tensor([[0, 1, 2, 3]])):
            output, report = r_shuffle_labels(labels, return_report=True)
            self.assertTrue(torch.equal(output, labels))
            self.assertEqual(report['degenerate_by_head'], [True])

    def test_pseudo_exact_source_by_bin_and_no_input_change(self):
        bins = torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]]).expand(2, -1)
        output, report = pseudo_source_labels(self.labels, bins, seed=42, return_report=True)
        for h in range(2):
            for b in (0, 1):
                mask = bins[h] == b
                self.assertTrue(torch.equal(torch.bincount(output[h, mask], minlength=4),
                                            torch.bincount(self.labels[h, mask], minlength=4)))
        self.assertTrue(bool((output != self.labels).any()))
        self.assertTrue(report['per_bin_counts_exact'])

    def test_pseudo_single_role_bins_noop(self):
        labels = torch.tensor([[0, 1, 1, 2, 2]])
        bins = torch.tensor([[-1, 0, 0, 1, 1]])
        output, report = pseudo_source_labels(labels, bins, return_report=True)
        self.assertTrue(torch.equal(output, labels))
        self.assertEqual(report['degenerate_by_head'], [True])

    def test_random_exact_g_source_age_and_protected_mandatory(self):
        candidates = self.labels > 0
        protected = self.labels == 0
        mandatory = torch.zeros_like(candidates); mandatory[:, 3] = True
        bins = torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]]).expand(2, -1)
        reference = protected.clone()
        reference[:, [3, 4, 6, 8, 10, 12]] = True
        keep, report = random_matched_keep(candidates, mandatory, 6, seed=43,
            reference=reference, labels=self.labels, bins=bins, protected=protected, return_report=True)
        self.assertTrue(torch.equal(keep & ~candidates, protected))
        self.assertTrue(bool(keep[mandatory].all()))
        for h in range(2):
            for role in (1, 2, 3):
                for b in (0, 1):
                    scope = (self.labels[h] == role) & (bins[h] == b)
                    self.assertEqual(int((keep[h] & scope).sum()), int((reference[h] & scope).sum()))
        self.assertTrue(torch.equal(keep, random_matched_keep(candidates, mandatory, 6, seed=43,
            reference=reference, labels=self.labels, bins=bins, protected=protected)))
        self.assertTrue(report['source_age_counts_exact'])

    def test_random_invalid_budget_or_reference_rejected(self):
        c = self.labels > 0; p = ~c; m = torch.zeros_like(c)
        bins = torch.zeros_like(self.labels)
        with self.assertRaises(ValueError):
            random_matched_keep(c, m, 3, reference=p, labels=self.labels, bins=bins, protected=p)

    def test_stable_seed(self):
        self.assertEqual(stable_seed('q1', 2), stable_seed('q1', 2))
        self.assertNotEqual(stable_seed('q1', 1), stable_seed('q1', 2))
        self.assertNotEqual(stable_seed('q2', 2), stable_seed('q1', 2))


if __name__ == '__main__':
    unittest.main()
