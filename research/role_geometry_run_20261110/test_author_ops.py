"""CPU-only author equivalence, protection, provenance and failure-gate tests.

Synthetic tensors only. No model, tokenizer, dataset, network or cloud execution.
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

import torch

import author_ops
from author_ops import load_author, select_author, verify_author_sources


class FixtureCache:
    def __init__(self, values):
        self.values = tuple(values)

    def to_legacy_cache(self):
        return self.values

    def get_seq_length(self):
        return 0 if not self.values else int(self.values[0][0].shape[-2])


def make_cache(values):
    return FixtureCache(values)


def stage_fixture(previous, before, role, prompt_len=80, layers=36, heads=8, dimension=128):
    """Unique synthetic K, nondegenerate V, head/layer-distinct sender scores."""
    prior = tuple() if previous is None else previous.to_legacy_cache()
    history = 0 if previous is None else previous.get_seq_length()
    generator = torch.Generator().manual_seed(934 + before)
    xs = []
    for layer in range(layers):
        fresh_k = torch.randn((1, heads, prompt_len + 10, dimension), generator=generator).to(torch.bfloat16)
        fresh_v = torch.randn((1, heads, prompt_len + 10, dimension), generator=generator).to(torch.bfloat16)
        if prior:
            fresh_k = torch.cat((prior[layer][0], fresh_k), 2)
            fresh_v = torch.cat((prior[layer][1], fresh_v), 2)
        xs.append((fresh_k, fresh_v))
    attentions = []
    for step in range(10):
        by_layer = []
        for layer in range(layers):
            scores = torch.zeros((1, heads * 4, history + prompt_len + step + 1))
            j = torch.arange(prompt_len)
            for head in range(heads):
                rank = (prompt_len - ((j - head * 7 - layer * 3) % prompt_len)).float()
                rank += j.float() * 0.0013
                scores[:, head * 4:(head + 1) * 4, history:history + prompt_len] = rank / max(1, prompt_len) * 4
            by_layer.append(scores.softmax(-1))
        attentions.append(by_layer)
    record = dict(role=role, prompt_len=prompt_len, latent_steps=10,
                  original_before=before, original_end=before + prompt_len + 10)
    return tuple(xs), record, attentions


class AuthorOperators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def _chain(self, mode, dimensions=(36, 8, 128), prompts=(80, 80, 80)):
        layers, heads, dimension = dimensions
        original, receipt = load_author(mode, make_cache)
        cache = None
        positions = None
        stages = []
        before = 0
        lengths = []
        changed = 0
        head_difference = False
        for role, prompt_len in zip(("planner", "critic", "refiner"), prompts):
            xs, record, attention = stage_fixture(cache, before, role, prompt_len, layers, heads, dimension)
            snapshots = tuple(tuple(t.clone() for t in pair) for pair in xs)
            stages.append(record)
            rng_before = torch.random.get_rng_state().clone()
            raw, _, _ = original.compress(past_key_values=xs, latent_steps=10,
                                          all_steps_attentions=attention,
                                          prompt_mask=torch.ones((1, prompt_len), dtype=torch.long))
            cache, new_positions, indices, report = select_author(xs, positions, stages, mode, attention, make_cache)
            self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
            self.assertEqual(indices.shape, new_positions.shape)
            for old_pair, snapshot_pair, direct_pair, adapter_pair in zip(xs, snapshots, raw.to_legacy_cache() if hasattr(raw, "to_legacy_cache") else raw, cache.to_legacy_cache()):
                for source, snapshot in zip(old_pair, snapshot_pair):
                    self.assertTrue(torch.equal(source, snapshot), "source KV was changed")
                for direct, adapted in zip(direct_pair, adapter_pair):
                    self.assertTrue(torch.equal(direct, adapted), "adapter differs from continuous original author class")
            if positions is not None:
                self.assertTrue(torch.equal(new_positions[..., :positions.shape[-1]], positions))
            for stage in stages:
                mask = (new_positions >= stage["original_end"] - 10) & (new_positions < stage["original_end"])
                self.assertTrue(bool((mask.sum(-1) == 10).all()))
            self.assertTrue(bool(((new_positions < 4).sum(-1) == 4).all()))
            self.assertTrue(report["source_k_equal_measured"])
            self.assertTrue(report["protected_v_equal_measured"])
            self.assertTrue(report["history_preserved_equal_measured"])
            self.assertTrue(report["source_input_unmodified_measured"])
            self.assertEqual(report["source_input_integrity_check"], "tensor_versions")
            self.assertEqual(report["author"]["pca_rank"], {"H-OBF": 2, "L-OBF": 4}.get(mode, 0))
            self.assertEqual(report["author"]["commit"], author_ops.COMMIT)
            self.assertTrue(report["author"]["cache_factory_only_adapter"])
            self.assertTrue(report["independent_author_selection_equal"])
            self.assertEqual(report["actual_kv_bytes"], report["retained"] * layers * heads * dimension * 2 * 2)
            json.dumps(report, allow_nan=False)
            changed += sum(report["changed_prompt_value_rows_by_layer"])
            head_difference |= not torch.equal(new_positions, new_positions[:, :1].expand_as(new_positions))
            positions = new_positions
            lengths.append(report["retained"])
            before = record["original_end"]
        if prompts == (80, 80, 80):
            self.assertEqual(lengths, {"Full": [90, 180, 270], "Gen": [14, 24, 34]}.get(mode, [46, 88, 130]))
            if mode in {"H-OBF", "L-OBF"}:
                self.assertGreater(changed, 0, "nondegenerate discarded values did not backfill any retained V")
                self.assertEqual(receipt["inject_mode"], "uniform")
                self.assertEqual(receipt["scale_mode"], "ad_over_as")
            else:
                self.assertEqual(changed, 0)
            self.assertEqual(head_difference, mode in {"H", "H-OBF"})
        else:
            # All prompt positions fit the budget, so exact OBF has no discard pool.
            self.assertEqual(changed, 0)
        if mode == "Gen":
            expected = list(range(4))
            for stage in stages:
                expected.extend(range(stage["original_end"] - 10, stage["original_end"]))
            self.assertEqual(positions[0, 0].tolist(), expected)
        return lengths

    def test_full_three_handoff_real_qwen_kv_shape(self):
        self._chain("Full")

    def test_generation_only_three_handoff_author_semantics(self):
        self._chain("Gen")

    def test_headwise_three_handoff_real_qwen_kv_shape(self):
        self._chain("H")

    def test_layerwise_three_handoff_real_qwen_kv_shape(self):
        self._chain("L")

    def test_hobf_rank2_three_handoff_exact_author_equivalence(self):
        self._chain("H-OBF")

    def test_lobf_rank4_three_handoff_exact_author_equivalence(self):
        self._chain("L-OBF")

    def test_obf_patch_matches_independent_pseudoinverse_eigen_reference(self):
        # The author uses QR + residual SVD. This separate check uses a
        # pseudoinverse projector + symmetric eigendecomposition in float64.
        # It tests the actual value correction, not just class/metadata names.
        for mode, rank in (("H-OBF", 2), ("L-OBF", 4)):
            with self.subTest(mode=mode):
                xs, stage, attention = stage_fixture(None, 0, "planner", layers=1, heads=2, dimension=48)
                xs = tuple(tuple(t.float() for t in pair) for pair in xs)
                cache, _, ix, report = select_author(xs, None, [stage], mode, attention, make_cache)
                scores = sum(step[0][0, :, :80].double() for step in attention)
                scores = scores.reshape(2, 4, 80).sum(1)
                values = xs[0][1][0].double()
                actual = cache.to_legacy_cache()[0][1][0].double()
                for head in range(2):
                    kept = ix[0, head, 4:36]
                    candidates = torch.arange(4, 80)
                    discarded = candidates[~torch.isin(candidates, kept)]
                    x, y = values[head, kept], values[head, discarded]
                    residual = y - y @ (torch.linalg.pinv(x) @ x)
                    _, eigenvectors = torch.linalg.eigh(residual.T @ residual)
                    basis = eigenvectors[:, -rank:]
                    wd, wk = scores[head, discarded], scores[head, kept]
                    mean_residual = (wd / (wd.sum() + 1e-12)) @ residual
                    delta = mean_residual @ basis @ basis.T * (wd.sum() / (wk.sum() + 1e-12))
                    self.assertGreater(float(torch.linalg.norm(delta)), 1e-4)
                    expected = x + delta
                    torch.testing.assert_close(actual[head, 4:36], expected, rtol=2e-4, atol=2e-5)
                    self.assertLess(float((delta @ x.T).abs().max()), 1e-8)
                self.assertEqual(report["pca_rank"], rank)

    def test_short_prompts_no_discard_and_sink_only_first_handoff(self):
        for mode in ("Full", "Gen", "H", "L", "H-OBF", "L-OBF"):
            with self.subTest(mode=mode):
                self._chain(mode, dimensions=(2, 2, 48), prompts=(4, 16, 32))

    def test_source_hash_gate_rejects_modified_dependency(self):
        verify_author_sources()
        with mock.patch("author_ops._sha", return_value="0" * 64):
            with self.assertRaisesRegex(RuntimeError, "Pinned author source mismatch"):
                verify_author_sources()

    def test_tampered_cache_keys_are_rejected(self):
        xs, stage, attention = stage_fixture(None, 0, "planner", layers=1, heads=2, dimension=48)

        def corrupt_factory(values):
            changed = [(k.clone(), v.clone()) for k, v in values]
            changed[0][0][..., 0, :] += 1
            return FixtureCache(changed)

        with self.assertRaisesRegex(RuntimeError, "original K"):
            select_author(xs, None, [stage], "H", attention, corrupt_factory)

    def test_obf_protected_value_tampering_is_rejected(self):
        xs, stage, attention = stage_fixture(None, 0, "planner", layers=1, heads=2, dimension=48)

        def corrupt_factory(values):
            changed = [(k.clone(), v.clone()) for k, v in values]
            changed[0][1][..., 0, :] += 1
            return FixtureCache(changed)

        with self.assertRaisesRegex(RuntimeError, "protected history/sink/latent V"):
            select_author(xs, None, [stage], "H-OBF", attention, corrupt_factory)

    def test_missing_attention_and_mapping_are_rejected(self):
        xs, stage, attention = stage_fixture(None, 0, "planner", layers=1, heads=2, dimension=48)
        with self.assertRaisesRegex(ValueError, "ten complete"):
            select_author(xs, None, [stage], "H", attention[:1], make_cache)
        stage2 = dict(role="critic", prompt_len=80, original_before=90, original_end=180)
        with self.assertRaisesRegex(ValueError, "History mapping"):
            select_author(xs, None, [stage, stage2], "L", attention, make_cache)

    def test_inference_tensors_are_supported_without_version_counters(self):
        with torch.inference_mode():
            xs, stage, attention = stage_fixture(None, 0, "planner", layers=1, heads=2, dimension=48)
            snapshots = tuple(tuple(t.clone() for t in pair) for pair in xs)
            cache, _, _, report = select_author(xs, None, [stage], "Gen", attention, make_cache)
            self.assertEqual(cache.get_seq_length(), 14)
            self.assertEqual(report["source_input_integrity_check"], "versions_and_sampled_inference_values")
            for pair, snapshot_pair in zip(xs, snapshots):
                for tensor, snapshot in zip(pair, snapshot_pair):
                    self.assertTrue(torch.equal(tensor, snapshot))


if __name__ == "__main__":
    unittest.main(verbosity=2)
