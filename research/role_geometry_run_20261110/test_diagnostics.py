import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from diagnostics import (geometry, geometry_diagnostics, provenance_labels,
                         readout, readout_diagnostics, receiver_fidelity,
                         save_compact_npz)


def fixture():
    stages = [dict(role=role, original_before=i*20, prompt_len=16, original_end=(i+1)*20)
              for i, role in enumerate(('planner', 'critic', 'refiner'))]
    positions = torch.arange(60).expand(2, -1).clone()
    x = torch.zeros((2, 60, 128), dtype=torch.float64)
    # Each role has nonzero centered variation along its own orthogonal feature.
    for r in range(3):
        x[:, r*20:(r+1)*20, r] = torch.arange(20, dtype=torch.float64)
    return x, positions, stages


class DiagnosticTests(unittest.TestCase):
    def test_provenance_sink_latent_self_and_relative_bins(self):
        x, pos, stages = fixture()
        out = provenance_labels(pos, stages, observed_length=63)
        self.assertEqual(tuple(out['labels'].shape), (2, 63))
        self.assertTrue(bool((out['labels'][:, :4] == 0).all()))
        self.assertTrue(bool((out['labels'][:, 60:] == 0).all()))
        self.assertTrue(bool((out['token_class'][:, 16:20] == 1).all()))
        self.assertTrue(bool((out['labels'][:, 20:40] == 2).all()))
        self.assertTrue(bool(((out['relative_bins'][:, 4:16] >= 0) & (out['relative_bins'][:, 4:16] < 4)).all()))

    def test_rank_and_subspace_geometry_equal_count_matching(self):
        x, pos, stages = fixture()
        output = geometry_diagnostics(x, x, pos, stages, seed=42)
        self.assertEqual(output['real_metrics'].shape, (2, 2, 3, 2, 3))
        self.assertTrue(np.allclose(output['real_metrics'][..., :2], 1.0, atol=1e-10))
        self.assertTrue(np.all(output['real_sample_count'][:, :, 0] == 12))
        self.assertTrue(np.all(output['real_sample_count'][:, :, 1] == 4))
        overlap = output['real_subspace_overlap'][0, 0, 0]
        self.assertTrue(np.allclose(overlap, np.eye(3), atol=1e-10))
        self.assertTrue(np.all(output['pseudo_changed_fraction_by_head'] > 0))
        self.assertTrue(np.allclose(output['real_top_energy'][..., 0], 1))
        repeated = geometry_diagnostics(x, x, pos, stages, seed=42)
        for key in output:
            self.assertTrue(np.array_equal(output[key], repeated[key]))

    def test_zero_geometry_and_missing_role_are_explicit(self):
        x, pos, stages = fixture()
        zero = geometry_diagnostics(torch.zeros_like(x), torch.zeros_like(x), pos, stages)
        self.assertTrue(np.all(zero['real_metrics'] == 0))
        self.assertTrue(np.all(zero['real_subspace_overlap_rank'] == 0))
        mask = (pos[0] < 20) | ((pos[0] >= 36) & (pos[0] < 40))
        # Stage2 has latent rows but no prompt rows, so prompt matching is absent.
        absent = geometry_diagnostics(x[:, mask], x[:, mask], pos[:, mask], stages[:2])
        self.assertFalse(bool(absent['real_all_sources_present'][0, 0]))
        self.assertTrue(np.all(absent['real_sample_count'][:, :, 0] == 0))

    def test_readout_cross_terms_equal_direct_total_mse(self):
        rng = torch.Generator().manual_seed(4)
        a = torch.rand((2, 3, 15), generator=rng, dtype=torch.float64)
        a /= a.sum(-1, keepdim=True)
        v = torch.randn((2, 15, 128), generator=rng, dtype=torch.float64)
        labels = torch.tensor([[0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 0, 0]]).expand(2, -1)
        keep = torch.ones((2, 15), dtype=torch.bool); keep[:, [3, 7, 10]] = False
        result = readout_diagnostics(a, v, labels, keep)
        w = a * keep[:, None]
        direct = (a @ v - w @ v / w.sum(-1, keepdim=True)).square().mean((-2, -1))
        self.assertTrue(np.allclose(result['total_error_mse'], direct.numpy(), atol=1e-12))
        self.assertTrue(np.allclose(result['source_error_cross_matrix'].sum((1, 2)), direct.numpy(), atol=1e-12))
        self.assertTrue(np.all(np.abs(result['source_error_cosine_matrix']) <= 1))
        same = readout_diagnostics(a, v, labels, torch.ones_like(keep))
        self.assertTrue(np.all(same['source_error_mse'] == 0))
        self.assertTrue(np.all(same['zero_error']))

    def test_readout_obf_dense_modified_value_used(self):
        a = torch.ones((1, 1, 4), dtype=torch.float64) / 4
        v = torch.zeros((1, 4, 128), dtype=torch.float64)
        labels = torch.tensor([[0, 1, 1, 0]])
        modified = v.clone(); modified[:, 1] = 2
        out = readout_diagnostics(a, v, labels, torch.ones_like(labels, dtype=torch.bool), selected_values=modified)
        self.assertTrue(np.allclose(out['total_error_mse'], .25))

    def test_runner_adapters_and_no_raw_npz(self):
        x, pos, stages = fixture()
        xs = ((x[None], x[None]),)
        arrays, meta = geometry(xs, pos[None], stages, seed=42)
        self.assertEqual(arrays['real_metrics'].shape, (1, 2, 2, 3, 2, 3))
        a = torch.ones((2, 3, 62), dtype=torch.float64) / 62
        v = torch.cat((x, torch.ones((2, 2, 128))), 1)
        labels = provenance_labels(pos, stages, 62)['labels'][None]
        indices = torch.arange(60).expand(1, 2, -1).clone()
        read = readout([(a, v)], indices, xs, labels, 60)
        self.assertTrue(np.all(read['total_error_mse'] == 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'derived.npz'
            receipt = save_compact_npz(path, arrays, metadata=meta)
            self.assertTrue(receipt['readback_verified'])
            self.assertFalse(receipt['raw_kv_retained'])
            with np.load(path, allow_pickle=False) as stored:
                self.assertNotIn('key', stored.files)
            with self.assertRaises(FileExistsError):
                save_compact_npz(path, arrays)
            with self.assertRaises(ValueError):
                save_compact_npz(Path(directory) / 'raw.npz', {'raw_k': np.zeros((1, 2))})

    def test_fidelity_zero_and_nonnegative_kl(self):
        logits = torch.tensor([[1., 3., 2.]])
        same = receiver_fidelity(logits, logits, full_hidden=logits, selected_hidden=logits)
        self.assertTrue(np.all(same['full_to_selected_kl'] == 0))
        self.assertTrue(np.all(same['hidden_relative_l2'] == 0))
        changed = receiver_fidelity(logits, logits.flip(-1))
        self.assertTrue(np.all(changed['full_to_selected_kl'] > 0))


if __name__ == '__main__':
    unittest.main()
