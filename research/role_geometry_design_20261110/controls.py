"""CPU-only role-control primitives for the geometry pilot.

These controls preserve counts and mandatory rows but do not run model forwards.
"""
from __future__ import annotations

import torch


def r_shuffle_labels(labels: torch.Tensor) -> torch.Tensor:
    """Permute nonzero role labels while preserving per-row counts and block 0."""
    if labels.ndim != 2:
        raise ValueError("labels must be [heads, positions]")
    out = labels.clone()
    for h in range(labels.shape[0]):
        pos = torch.where(labels[h] != 0)[0]
        if pos.numel() < 2:
            continue
        shift = max(1, int(pos.numel() // 2))
        out[h, pos] = labels[h, pos].roll(shifts=shift, dims=0)
    assert torch.equal(torch.bincount(out.reshape(-1), minlength=int(labels.max()) + 1),
                       torch.bincount(labels.reshape(-1), minlength=int(labels.max()) + 1))
    assert torch.equal(out[labels == 0], labels[labels == 0])
    return out


def pseudo_source_labels(labels: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Shuffle source labels within matched age/position bins, preserving bin counts."""
    if labels.shape != bins.shape or labels.ndim != 2:
        raise ValueError("labels and bins must have shape [heads, positions]")
    out = labels.clone()
    for h in range(labels.shape[0]):
        for b in torch.unique(bins[h]).tolist():
            idx = torch.where((bins[h] == b) & (labels[h] != 0))[0]
            if idx.numel() < 2:
                continue
            out[h, idx] = labels[h, idx].roll(shifts=1, dims=0)
    assert torch.equal(torch.bincount(out.reshape(-1), minlength=int(labels.max()) + 1),
                       torch.bincount(labels.reshape(-1), minlength=int(labels.max()) + 1))
    assert torch.equal(out[labels == 0], labels[labels == 0])
    return out


def random_matched_keep(
    candidates: torch.Tensor,
    mandatory: torch.Tensor,
    budget: int,
    seed: int = 0,
) -> torch.Tensor:
    """Keep mandatory rows plus a deterministic random candidate sample per head."""
    if candidates.shape != mandatory.shape or candidates.ndim != 2:
        raise ValueError("candidates and mandatory must be [heads, positions]")
    if bool((mandatory & ~candidates).any()):
        raise ValueError("mandatory rows must be candidates")
    g = torch.Generator(device=candidates.device).manual_seed(int(seed))
    selected = torch.zeros_like(candidates, dtype=torch.bool)
    for h in range(candidates.shape[0]):
        mand = torch.where(mandatory[h])[0]
        free = torch.where(candidates[h] & ~mandatory[h])[0]
        need = int(budget) - int(mand.numel())
        if need < 0 or need > free.numel():
            raise ValueError("budget is incompatible with candidates")
        selected[h, mand] = True
        if need:
            pick = free[torch.randperm(free.numel(), generator=g, device=free.device)[:need]]
            selected[h, pick] = True
    assert bool(((selected & candidates).sum(-1) == int(budget)).all())
    assert bool((mandatory & ~selected).any()) is False
    return selected
