"""Deterministic, forward-free null controls for the 120-question pilot.

Membership is shuffled rather than merely renaming role IDs. Functions preserve
input tensors and use a CPU generator so their RNG is independent of model RNG.
"""
from __future__ import annotations

import hashlib
from typing import Any

import torch


def stable_seed(question_id: str, hop: int, *, seed: int = 42, salt: str = "role_geometry_v1") -> int:
    payload = f"{salt}|{int(seed)}|{question_id}|{int(hop)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _check_labels(labels: torch.Tensor) -> None:
    if labels.ndim != 2 or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("labels must be integer [heads, positions]")
    if bool((labels < 0).any()):
        raise ValueError("source IDs must be nonnegative")


def _hist(x: torch.Tensor, width: int) -> torch.Tensor:
    return torch.bincount(x.long(), minlength=width)


def _membership_shuffle(values: torch.Tensor, generator: torch.Generator) -> tuple[torch.Tensor, bool]:
    """Uniform row shuffle, rejecting identities and pure global role renames.

    With singleton classes a non-rename shuffle can be impossible. That is
    explicitly reported instead of silently substituting a role rename.
    """
    if values.numel() < 2 or torch.unique(values).numel() < 2:
        return values.clone(), True
    unique = torch.unique(values)
    for _ in range(64):
        result = values[torch.randperm(values.numel(), generator=generator)]
        if torch.equal(result, values):
            continue
        # A role's members must be split, or two source roles must share a new
        # label. A pure bijective rename leaves all memberships unchanged.
        is_rename = all(torch.unique(result[values == role]).numel() == 1 for role in unique)
        if not is_rename:
            return result, False
    # Guaranteed membership-changing fallback whenever any class has two rows:
    # swap one member of that class with one member from another class.
    counts = [(int((values == role).sum()), int(role)) for role in unique]
    eligible = [role for count, role in counts if count > 1]
    if eligible:
        role = eligible[int(torch.randint(len(eligible), (1,), generator=generator))]
        left = torch.where(values == role)[0]
        right = torch.where(values != role)[0]
        i = int(left[int(torch.randint(left.numel(), (1,), generator=generator))])
        j = int(right[int(torch.randint(right.numel(), (1,), generator=generator))])
        result = values.clone()
        result[i], result[j] = values[j], values[i]
        return result, False
    return values.clone(), True


def r_shuffle_labels(labels: torch.Tensor, seed: int = 42, *, return_report: bool = False):
    """Shuffle nonzero memberships independently per head; keep block 0 fixed.

    Exact per-head source counts are preserved. A one-role first handoff is a
    declared degenerate/no-op control rather than an invented role boundary.
    """
    _check_labels(labels)
    cpu = labels.detach().to(device="cpu", dtype=torch.long)
    result = cpu.clone()
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    degenerate = []
    fractions = []
    width = int(cpu.max()) + 1 if cpu.numel() else 1
    for h in range(cpu.shape[0]):
        eligible = torch.where(cpu[h] != 0)[0]
        shuffled, no_membership_null = _membership_shuffle(cpu[h, eligible], rng)
        result[h, eligible] = shuffled
        if not torch.equal(_hist(result[h], width), _hist(cpu[h], width)):
            raise AssertionError("per-head source counts changed")
        degenerate.append(bool(no_membership_null))
        fractions.append(float((result[h, eligible] != cpu[h, eligible]).double().mean()) if eligible.numel() else 0.0)
    if not torch.equal(result[cpu == 0], cpu[cpu == 0]):
        raise AssertionError("block 0 changed")
    report = {"control": "R-shuffle", "seed": int(seed), "changed_fraction_by_head": fractions,
              "degenerate_by_head": degenerate, "per_head_counts_exact": True,
              "block0_fixed": True, "membership_not_role_rename": True}
    output = result.to(device=labels.device, dtype=labels.dtype)
    return (output, report) if return_report else output


def pseudo_source_labels(labels: torch.Tensor, age_bins: torch.Tensor, seed: int = 42, *, return_report: bool = False):
    """Shuffle memberships within age strata, keeping source counts per stratum.

    Block 0 remains fixed. Bins with only one source or singleton-only classes
    cannot supply a membership null and are recorded as degenerate.
    """
    _check_labels(labels)
    if age_bins.shape != labels.shape or age_bins.dtype not in (torch.int32, torch.int64):
        raise ValueError("age_bins must be integer and match labels")
    cpu = labels.detach().to(device="cpu", dtype=torch.long)
    bins = age_bins.detach().to(device="cpu", dtype=torch.long)
    result = cpu.clone()
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    width = int(cpu.max()) + 1 if cpu.numel() else 1
    stratum_records = []
    for h in range(cpu.shape[0]):
        for b in torch.unique(bins[h], sorted=True).tolist():
            idx = torch.where((bins[h] == b) & (cpu[h] != 0))[0]
            shuffled, degenerate = _membership_shuffle(cpu[h, idx], rng)
            result[h, idx] = shuffled
            if not torch.equal(_hist(result[h, idx], width), _hist(cpu[h, idx], width)):
                raise AssertionError("per-bin source counts changed")
            stratum_records.append({"head": h, "bin": int(b), "n": int(idx.numel()),
                "changed": int((shuffled != cpu[h, idx]).sum()), "degenerate": bool(degenerate)})
    eligible = cpu != 0
    changed = result != cpu
    fractions = [float(changed[h, eligible[h]].double().mean()) if bool(eligible[h].any()) else 0.0 for h in range(cpu.shape[0])]
    report = {"control": "Pseudo-R", "seed": int(seed), "changed_fraction_by_head": fractions,
              "degenerate_by_head": [not bool(changed[h].any()) for h in range(cpu.shape[0])],
              "strata": stratum_records, "per_bin_counts_exact": True,
              "block0_fixed": True, "membership_not_role_rename": True}
    output = result.to(device=labels.device, dtype=labels.dtype)
    return (output, report) if return_report else output


def random_matched_keep(candidates: torch.Tensor, mandatory: torch.Tensor, budget: int, seed: int = 42, *,
                        reference: torch.Tensor, labels: torch.Tensor, bins: torch.Tensor,
                        protected: torch.Tensor | None = None, return_report: bool = False):
    """Randomize G's keep set at exact source x age-bin counts per head.

    `reference_keep` is the G keep mask on this very snapshot. Output includes
    exactly `protected`, plus mandatory candidates and a stratified random
    sample. Rows outside protected|candidates are never selected.
    """
    _check_labels(labels)
    reference_keep, age_bins = reference, bins
    shape = labels.shape
    arrays = (candidates, mandatory, reference_keep, age_bins)
    if any(x.shape != shape for x in arrays):
        raise ValueError("all masks/strata must have shape [heads, positions]")
    if any(x.dtype != torch.bool for x in (candidates, mandatory, reference_keep)):
        raise ValueError("selection masks must be bool")
    if age_bins.dtype not in (torch.int32, torch.int64):
        raise ValueError("age_bins must be integer")
    if protected is None:
        protected = reference_keep & ~candidates
    if protected.shape != shape or protected.dtype != torch.bool:
        raise ValueError("protected must be bool and match labels")
    c, m, ref, pro = [x.detach().cpu() for x in (candidates, mandatory, reference_keep, protected)]
    lab = labels.detach().cpu().long()
    bins = age_bins.detach().cpu().long()
    if bool((m & ~c).any()) or bool((pro & c).any()):
        raise ValueError("mandatory must be candidates and protected disjoint")
    if bool((m & ~ref).any()) or bool((pro & ~ref).any()) or bool((ref & ~(pro | c)).any()):
        raise ValueError("G reference violates protected/candidate/mandatory scope")
    if not bool(((ref & c).sum(-1) == int(budget)).all()):
        raise ValueError("G reference does not match requested prompt budget")
    keep = pro | m
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    records = []
    for h in range(shape[0]):
        strata = torch.stack((lab[h, c[h]], bins[h, c[h]]), -1).unique(dim=0, sorted=True)
        for role, b in strata.tolist():
            scope = c[h] & (lab[h] == role) & (bins[h] == b)
            quota = int((ref[h] & scope).sum())
            forced = int((m[h] & scope).sum())
            free = torch.where(scope & ~m[h])[0]
            need = quota - forced
            if not 0 <= need <= free.numel():
                raise ValueError("stratum quota incompatible with mandatory rows")
            if need:
                keep[h, free[torch.randperm(free.numel(), generator=rng)[:need]]] = True
            if int((keep[h] & scope).sum()) != quota:
                raise AssertionError("source-age quota changed")
            records.append({"head": h, "source": int(role), "age_bin": int(b),
                            "reference_count": quota, "mandatory_count": forced,
                            "candidate_count": int(scope.sum())})
    if not torch.equal(keep & ~c, pro) or bool((m & ~keep).any()):
        raise AssertionError("fixed rows changed")
    if not torch.equal((keep & c).sum(-1), (ref & c).sum(-1)):
        raise AssertionError("prompt budget changed")
    report = {"control": "Random-matched", "seed": int(seed), "strata": records,
              "reference": "G_same_snapshot", "source_age_counts_exact": True,
              "mandatory_exact": True, "protected_exact": True,
              "different_from_reference_by_head": [bool((keep[h] != ref[h]).any()) for h in range(shape[0])]}
    output = keep.to(device=candidates.device)
    return (output, report) if return_report else output
