"""Forward-free, compact role geometry and readout diagnostics.

Only derived arrays are returned/saved; raw K/V, attention, questions and tokens
are not retained. Geometry runs on detached CPU float64 tensors. Tensor layouts
are explicit in each public function's docstring.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from controls import pseudo_source_labels, stable_seed


METRIC_NAMES = ("participation_ratio", "entropy_effective_rank", "numerical_rank")
TOKEN_CLASS_NAMES = ("prompt", "latent")


def provenance_labels(positions: torch.Tensor, stages: Sequence[Mapping[str, Any]],
                      observed_length: int | None = None, n_bins: int = 4) -> dict[str, torch.Tensor]:
    """Label original positions [H,N]; optional receiver-self suffix is block 0.

    Source IDs are stage order 1..S, 0 is sink/self. `token_class` uses prompt=0,
    latent=1, sink/self=-1. `age_bins` use absolute normalized cache age, whereas
    `relative_bins` use within-role, within-token-class relative position.
    """
    if positions.ndim != 2 or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("positions must be integer [heads,N]")
    if not 1 <= len(stages) <= 3 or n_bins < 1:
        raise ValueError("one to three stages and positive n_bins required")
    pos = positions.detach().cpu().long()
    if bool((pos < 0).any()) or (pos.shape[-1] > 1 and not bool((pos.diff(dim=-1) > 0).all())):
        raise ValueError("original positions must increase strictly per head")
    n = pos.shape[-1]
    observed_length = n if observed_length is None else int(observed_length)
    if observed_length < n:
        raise ValueError("observed length cannot be less than relay length")
    shape = (pos.shape[0], observed_length)
    labels = torch.zeros(shape, dtype=torch.long)
    classes = torch.full(shape, -1, dtype=torch.long)
    relative = torch.full(shape, -1, dtype=torch.long)
    age = torch.full(shape, -1, dtype=torch.long)
    known = pos < 4
    end = int(stages[-1]["original_end"])
    if bool((pos >= end).any()):
        raise ValueError("positions exceed last original endpoint")
    age[:, :n] = (((end - 1 - pos) * n_bins) // max(1, end)).clamp(0, n_bins - 1)
    for role, stage in enumerate(stages, 1):
        before = int(stage["original_before"])
        prompt_end = before + int(stage["prompt_len"])
        role_end = int(stage["original_end"])
        if not 0 <= before <= prompt_end <= role_end <= end:
            raise ValueError("invalid stage interval")
        scope = (pos >= before) & (pos < role_end) & (pos >= 4)
        if bool((known & scope).any()):
            raise ValueError("stage intervals overlap")
        known |= scope
        labels[:, :n][scope] = role
        for kind, lo, hi in ((0, max(4, before), prompt_end), (1, prompt_end, role_end)):
            mask = scope & (pos >= lo) & (pos < hi)
            classes[:, :n][mask] = kind
            rel = ((pos - lo) * n_bins // max(1, hi - lo)).clamp(0, n_bins - 1)
            relative[:, :n][mask] = rel[mask]
    if not bool(known.all()):
        raise ValueError("unassigned relay position")
    return {"labels": labels, "token_class": classes, "age_bins": age,
            "relative_bins": relative}


def _cap_quotas(counts: torch.Tensor, cap: int) -> torch.Tensor:
    """Round-robin allocation prevents one relative-position bin dominating."""
    quotas = torch.zeros_like(counts)
    left = min(int(cap), int(counts.sum()))
    while left:
        for b in range(counts.numel()):
            if left and quotas[b] < counts[b]:
                quotas[b] += 1
                left -= 1
    return quotas


def _matched_indices(labels: torch.Tensor, token_class: torch.Tensor, relative_bins: torch.Tensor,
                     n_sources: int, cap: int, n_bins: int, seed: int):
    """Equal per-bin source sampling; missing-source strata remain explicit."""
    h_count = labels.shape[0]
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    chosen: dict[tuple[int, int, int], torch.Tensor] = {}
    sample_counts = np.zeros((h_count, n_sources, 2), dtype=np.int32)
    matched_bin_counts = np.zeros((h_count, 2, n_bins), dtype=np.int32)
    available = np.zeros((h_count, n_sources, 2, n_bins), dtype=np.int32)
    all_sources_present = np.zeros((h_count, 2), dtype=np.bool_)
    for h in range(h_count):
        for kind in range(2):
            counts = torch.tensor([[int(((labels[h] == r) & (token_class[h] == kind) &
                        (relative_bins[h] == b)).sum()) for b in range(n_bins)]
                        for r in range(1, n_sources + 1)], dtype=torch.long)
            available[h, :, kind] = counts.numpy()
            all_sources_present[h, kind] = bool((counts.sum(-1) > 0).all())
            quotas = _cap_quotas(counts.min(0).values, cap)
            matched_bin_counts[h, kind] = quotas.numpy()
            for r in range(1, n_sources + 1):
                pieces = []
                for b, quota in enumerate(quotas.tolist()):
                    idx = torch.where((labels[h] == r) & (token_class[h] == kind) & (relative_bins[h] == b))[0]
                    if quota:
                        pieces.append(idx[torch.randperm(idx.numel(), generator=rng)[:quota]])
                ix = torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.long)
                chosen[h, r - 1, kind] = ix
                sample_counts[h, r - 1, kind] = ix.numel()
    return chosen, sample_counts, matched_bin_counts, available, all_sources_present


def _spectrum(x: torch.Tensor, qrank: int) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    if x.shape[0] < 2:
        return np.zeros(3), np.zeros(qrank), torch.empty((0, x.shape[-1]), dtype=torch.float64)
    centered = x - x.mean(0, keepdim=True)
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    energies = singular.square()
    total = float(energies.sum())
    if total <= torch.finfo(torch.float64).tiny:
        return np.zeros(3), np.zeros(qrank), torch.empty((0, x.shape[-1]), dtype=torch.float64)
    tolerance = max(centered.shape) * torch.finfo(torch.float64).eps * float(singular[0])
    rank = int((singular > tolerance).sum())
    p = energies / energies.sum()
    positive = p[p > 0]
    metrics = np.array([float(energies.sum().square() / energies.square().sum()),
                        float(torch.exp(-(positive * positive.log()).sum())), rank], dtype=np.float64)
    top = np.zeros(qrank, dtype=np.float64)
    top[:min(qrank, p.numel())] = p[:qrank].numpy()
    return metrics, top, vh[:min(qrank, rank)]


def _geometry_for_labels(key: torch.Tensor, value: torch.Tensor, labels: torch.Tensor,
                         classes: torch.Tensor, rel_bins: torch.Tensor, n_sources: int,
                         cap: int, qrank: int, n_bins: int, seed: int) -> dict[str, np.ndarray]:
    chosen, counts, bins, available, present = _matched_indices(labels, classes, rel_bins, n_sources, cap, n_bins, seed)
    heads = labels.shape[0]
    metrics = np.zeros((2, heads, n_sources, 2, 3), dtype=np.float64)
    spectra = np.zeros((2, heads, n_sources, 2, qrank), dtype=np.float64)
    overlap = np.zeros((2, heads, 2, n_sources, n_sources), dtype=np.float64)
    overlap_rank = np.zeros_like(overlap, dtype=np.int16)
    for which, matrix in enumerate((key, value)):
        for h in range(heads):
            for kind in range(2):
                bases = []
                for source in range(n_sources):
                    m, top, basis = _spectrum(matrix[h, chosen[h, source, kind]], qrank)
                    metrics[which, h, source, kind] = m
                    spectra[which, h, source, kind] = top
                    bases.append(basis)
                for r in range(n_sources):
                    for s in range(n_sources):
                        rank = min(bases[r].shape[0], bases[s].shape[0], qrank)
                        overlap_rank[which, h, kind, r, s] = rank
                        if rank:
                            overlap[which, h, kind, r, s] = float((bases[r][:rank] @ bases[s][:rank].T).square().sum() / rank)
    return {"metrics": metrics, "top_energy": spectra, "subspace_overlap": overlap,
            "subspace_overlap_rank": overlap_rank, "sample_count": counts,
            "matched_relative_bin_count": bins, "available_relative_bin_count": available,
            "all_sources_present": present}


def geometry_diagnostics(key: torch.Tensor, value: torch.Tensor, positions: torch.Tensor,
                         stages: Sequence[Mapping[str, Any]], *, seed: int = 42,
                         max_samples: int = 64, qrank: int = 4, n_bins: int = 4) -> dict[str, np.ndarray]:
    """Return compact geometry, real and matched-pseudo, for one layer.

    K/V shape [heads,N,128], positions [heads,N]. Returned `real_metrics` and
    `pseudo_metrics` have axes [K_or_V,head,source,prompt_or_latent,metric].
    The first two metrics are participation/entropy effective rank; third is
    numerical rank. Feature subspaces are used transiently, never saved.
    """
    if key.ndim != 3 or value.shape != key.shape or key.shape[-1] != 128:
        raise ValueError("K/V must both be [heads,N,128]")
    if tuple(positions.shape) != tuple(key.shape[:2]):
        raise ValueError("position shape must match K/V rows")
    if not 1 <= max_samples <= 64 or not 1 <= qrank <= 4:
        raise ValueError("geometry cap is 64 rows and feature rank is at most 4")
    k, v = [x.detach().to(device="cpu", dtype=torch.float64) for x in (key, value)]
    if not bool(torch.isfinite(k).all() & torch.isfinite(v).all()):
        raise ValueError("non-finite K/V")
    prov = provenance_labels(positions, stages, n_bins=n_bins)
    labels, classes, rel = [prov[x] for x in ("labels", "token_class", "relative_bins")]
    # Match source membership within token class x within-role position stratum.
    combined_bins = classes * n_bins + rel
    pseudo, null_report = pseudo_source_labels(labels, combined_bins, seed=int(seed), return_report=True)
    output = {}
    for name, source_labels in (("real", labels), ("pseudo", pseudo)):
        result = _geometry_for_labels(k, v, source_labels, classes, rel, len(stages),
                                     max_samples, qrank, n_bins, seed)
        output.update({f"{name}_{field}": arr for field, arr in result.items()})
    output["pseudo_changed_fraction_by_head"] = np.asarray(null_report["changed_fraction_by_head"], dtype=np.float64)
    output["pseudo_degenerate_by_head"] = np.asarray(null_report["degenerate_by_head"], dtype=np.bool_)
    output["geometry_seed"] = np.asarray(int(seed), dtype=np.int64)
    output["geometry_max_samples"] = np.asarray(max_samples, dtype=np.int16)
    output["geometry_qrank"] = np.asarray(qrank, dtype=np.int16)
    output["position_sha256"] = np.asarray(hashlib.sha256(positions.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
    return output


def readout_diagnostics(attention: torch.Tensor, full_values: torch.Tensor,
                        source_labels: torch.Tensor, keep: torch.Tensor, *,
                        selected_values: torch.Tensor | None = None,
                        n_sources: int | None = None) -> dict[str, np.ndarray]:
    """Exact normalized source errors for one layer, including receiver-self.

    attention [H,Q,M], V [H,M,128], labels/keep [H,M], where M=N+self.
    `selected_values`, when given, is a dense [H,M,128] value tensor with any OBF
    modifications at retained rows; discarded rows are ignored. Error vectors
    are Full minus selected; all source components use the same normalization.
    """
    if attention.ndim != 3 or full_values.ndim != 3 or full_values.shape[-1] != 128:
        raise ValueError("expected attention [H,Q,M], V [H,M,128]")
    heads, _, length = attention.shape
    if tuple(full_values.shape[:2]) != (heads, length) or source_labels.shape != (heads, length) or keep.shape != (heads, length):
        raise ValueError("readout dimensions disagree")
    if keep.dtype != torch.bool or source_labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("keep must be bool and labels integer")
    a = attention.detach().to(device="cpu", dtype=torch.float64)
    v = full_values.detach().to(device="cpu", dtype=torch.float64)
    selected_v = v if selected_values is None else selected_values.detach().to(device="cpu", dtype=torch.float64)
    if selected_v.shape != v.shape:
        raise ValueError("selected_values must use dense original-row layout")
    lab = source_labels.detach().cpu().long()
    mask = keep.detach().cpu()
    if bool((lab < 0).any()) or bool((a < 0).any()) or not bool(torch.isfinite(a).all() & torch.isfinite(v).all() & torch.isfinite(selected_v).all()):
        raise ValueError("non-finite/negative attention or invalid source label")
    blocks = int(lab.max()) + 1 if n_sources is None else int(n_sources) + 1
    if blocks <= int(lab.max()):
        raise ValueError("n_sources excludes an observed label")
    full_z = a.sum(-1, keepdim=True)
    kept_a = a * mask[:, None, :]
    selected_z = kept_a.sum(-1, keepdim=True)
    if not bool((full_z > 0).all() & (selected_z > 0).all()):
        raise ValueError("zero attention mass")
    full_components = torch.stack([(a * (lab == r)[:, None, :]) @ v / full_z for r in range(blocks)], 1)
    selected_components = torch.stack([(kept_a * (lab == r)[:, None, :]) @ selected_v / selected_z for r in range(blocks)], 1)
    error = full_components - selected_components
    cross = torch.einsum("hrqd,hsqd->hrs", error, error) / (error.shape[-2] * error.shape[-1])
    energies = cross.diagonal(dim1=-2, dim2=-1)
    total_error = error.sum(1)
    total_mse = total_error.square().mean((-2, -1))
    algebra_error = (cross.sum((-2, -1)) - total_mse).abs()
    if not bool(torch.allclose(cross.sum((-2, -1)), total_mse, atol=1e-12, rtol=1e-10)):
        raise AssertionError("cross-term identity failed")
    denom = torch.sqrt(energies[:, :, None] * energies[:, None, :])
    cosines = torch.where(denom > 0, cross / denom.clamp_min(torch.finfo(torch.float64).tiny), torch.zeros_like(cross)).clamp(-1, 1)
    source_energy = energies.sum(-1)
    cancellation = torch.where(source_energy > 0, 1 - total_mse / source_energy, torch.zeros_like(total_mse))
    return {"source_error_mse": energies.numpy(), "total_error_mse": total_mse.numpy(),
            "source_error_cross_matrix": cross.numpy(), "source_error_cosine_matrix": cosines.numpy(),
            "source_error_mean_vector": error.mean(-2).numpy(),
            "total_error_mean_vector": total_error.mean(-2).numpy(),
            "cancellation": cancellation.numpy(), "zero_error": (source_energy == 0).numpy(),
            "cross_identity_abs_error": algebra_error.numpy(),
            "kept_attention_mass": (selected_z / full_z).squeeze(-1).numpy()}


def receiver_fidelity(full_logits: torch.Tensor, selected_logits: torch.Tensor, *,
                      full_hidden: torch.Tensor | None = None,
                      selected_hidden: torch.Tensor | None = None) -> dict[str, np.ndarray]:
    """CPU derived KL(Full||selected) and optional hidden relative L2/cosine."""
    if full_logits.shape != selected_logits.shape or full_logits.ndim < 1:
        raise ValueError("logits must share shape [...,vocabulary]")
    full, selected = [x.detach().cpu().double() for x in (full_logits, selected_logits)]
    if not bool(torch.isfinite(full).all() & torch.isfinite(selected).all()):
        raise ValueError("non-finite logits")
    lf, ls = full.log_softmax(-1), selected.log_softmax(-1)
    kl = (lf.exp() * (lf - ls)).sum(-1).clamp_min(0)
    result = {"full_to_selected_kl": kl.numpy()}
    if (full_hidden is None) != (selected_hidden is None):
        raise ValueError("both hidden states must be supplied")
    if full_hidden is not None:
        hf, hs = [x.detach().cpu().double() for x in (full_hidden, selected_hidden)]
        if hf.shape != hs.shape or not bool(torch.isfinite(hf).all() & torch.isfinite(hs).all()):
            raise ValueError("hidden state shape/non-finite mismatch")
        denom = hf.norm(dim=-1)
        result["hidden_relative_l2"] = ((hf - hs).norm(dim=-1) / denom.clamp_min(1e-30)).numpy()
        result["hidden_cosine"] = torch.nn.functional.cosine_similarity(hf, hs, dim=-1, eps=1e-30).numpy()
        result["full_hidden_zero"] = (denom == 0).numpy()
    return result


def save_compact_npz(path: str | Path, arrays: Mapping[str, np.ndarray], *,
                     metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Atomically write numeric/Unicode derived arrays and JSON metadata only."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"diagnostic already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for name, array in arrays.items():
        value = np.asarray(array)
        if value.dtype.kind not in "biufcUS":
            raise ValueError(f"non-compact/object payload {name}")
        if name in {"key", "value", "keys", "values", "attention", "raw_k", "raw_v", "tokens"}:
            raise ValueError("raw model tensors cannot be saved by diagnostic writer")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"non-finite derived array: {name}")
        payload[name] = value
    payload["metadata_json"] = np.asarray(json.dumps(dict(metadata or {}), sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    # Readback checks contain no raw K/V and verify archive integrity.
    with np.load(target, allow_pickle=False) as stored:
        if set(stored.files) != set(payload):
            raise AssertionError("diagnostic member set mismatch")
        for name, expected in payload.items():
            if not np.array_equal(stored[name], expected, equal_nan=False):
                raise AssertionError(f"diagnostic readback mismatch: {name}")
    return {"path": str(target), "sha256": digest, "bytes": target.stat().st_size,
            "arrays": sorted(payload), "raw_kv_retained": False, "readback_verified": True}


def geometry(xs, allmap: torch.Tensor, stages, seed: int = 42):
    """Runner adapter -> (derived NumPy arrays, small JSON metadata).

    xs is layer tuples (K,V), each tensor [1,H,N,128]; allmap [L,H,N].
    Layer is the first axis of every returned array. No raw tensors are saved.
    """
    if len(xs) != allmap.shape[0]:
        raise ValueError("layer/mapping mismatch")
    layers = []
    for layer, (key, value) in enumerate(xs):
        if key.ndim != 4 or key.shape[0] != 1 or value.shape != key.shape:
            raise ValueError("runner K/V must be [1,H,N,128]")
        layers.append(geometry_diagnostics(key[0], value[0], allmap[layer], stages,
                     seed=(int(seed) + layer * 1000003) % (2**63 - 1)))
    arrays = {name: np.stack([result[name] for result in layers]) for name in layers[0]}
    meta = {"geometry_version": 1, "max_samples": 64, "feature_subspace_rank": 4,
            "relative_position_bins": 4, "source_names": [str(s.get("role", r)) for r, s in enumerate(stages, 1)],
            "metric_names": list(METRIC_NAMES), "token_class_names": list(TOKEN_CLASS_NAMES),
            "metric_axes": ["layer", "K_or_V", "head", "source", "prompt_or_latent", "metric"],
            "overlap_axes": ["layer", "K_or_V", "head", "prompt_or_latent", "source", "source"],
            "matching": "equal per-source counts in within-role relative-position quartiles, separately prompt/latent",
            "zero_sample_or_rank": "recorded explicitly; zero is not evidence of identical subspace",
            "raw_kv_retained": False}
    return arrays, meta


def readout(captured, indices: torch.Tensor, new_xs, labels_by_layer, total: int):
    """Runner adapter -> derived NumPy arrays; layer is the leading axis.

    captured has per-layer (attention[H,Q,N+self], fullV[H,N+self,128]);
    indices[L,H,K] are original storage indices of retained relay rows; new_xs
    layer V[1,H,K,128] may include OBF. Receiver-self suffix stays selected.
    """
    if not len(captured) == len(new_xs) == indices.shape[0] == len(labels_by_layer):
        raise ValueError("readout layer mismatch")
    results = []
    for layer, ((attention, values), (_, retained_v)) in enumerate(zip(captured, new_xs)):
        ix = indices[layer].detach().cpu().long()
        a, v = attention.detach().cpu(), values.detach().cpu()
        lab = labels_by_layer[layer].detach().cpu().long()
        if ix.ndim != 2 or ix.shape[0] != a.shape[0] or not bool(((ix >= 0) & (ix < total)).all()):
            raise ValueError("selected storage indices outside relay")
        if retained_v.shape != (1, a.shape[0], ix.shape[-1], 128):
            raise ValueError("retained values/selected rows mismatch")
        keep = torch.zeros((a.shape[0], a.shape[-1]), dtype=torch.bool)
        keep[:, int(total):] = True
        keep.scatter_(1, ix, True)
        dense = v.clone()
        dense.scatter_(1, ix[:, :, None].expand(-1, -1, 128), retained_v[0].detach().cpu().to(v.dtype))
        results.append(readout_diagnostics(a, v, lab, keep, selected_values=dense))
    return {name: np.stack([result[name] for result in results]) for name in results[0]}
