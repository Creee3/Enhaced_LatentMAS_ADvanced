"""Final-only private diagnostics postprocessing; NumPy + Python standard library.

This file is outside the frozen generation source. Point --run-root at the
verified final run/archive extraction. It never imports the model or scorer.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import time
import zipfile

import numpy as np

METHODS = ("Full", "Gen", "H", "L", "H-OBF", "L-OBF", "G", "T", "R", "R-shuffle", "Pseudo-R", "Random-matched")
GROUPS = ("AIME24", "GPQA-Diamond", "GPQA-nonDiamond", "HumanEval+", "MedQA", "MMLU-Pro")
ROLES = ("planner", "critic", "refiner")
RECEIVERS = ("critic", "refiner", "judger")
RECEIVER_NAMES = ("full_to_selected_kl", "hidden_relative_l2", "hidden_cosine", "full_hidden_zero_fraction")
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 42


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _mean(values, valid=None):
    values = np.asarray(values, dtype=np.float64)
    valid = np.ones(values.shape, dtype=bool) if valid is None else np.broadcast_to(valid, values.shape)
    valid = valid & np.isfinite(values)
    return float(values[valid].mean()) if valid.any() else None


def _array(z, name, shape=None, kinds="biuf"):
    a = z[name]
    if a.dtype.kind not in kinds or (shape is not None and a.shape != shape):
        raise ValueError("Diagnostic array schema mismatch: " + name)
    if a.dtype.kind in "fc" and not np.isfinite(a).all():
        raise ValueError("Non-finite diagnostic array: " + name)
    return a


def _safe_path(base, relative):
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or ":" in relative or "\\" in relative:
        raise ValueError("Diagnostic path is not confined to its attempt")
    target = Path(base).joinpath(*pure.parts).resolve()
    if not target.is_relative_to(Path(base).resolve()):
        raise ValueError("Diagnostic path escapes its attempt")
    return target


def reduce_geometry(z, hop):
    """Mean repeated layers/heads within a question; missing/rank-zero masked."""
    real = _array(z, "real_metrics")
    if real.ndim != 6 or real.shape[1] != 2 or real.shape[3:] != (hop, 2, 3):
        raise ValueError("Geometry axes must be [layer,K_or_V,head,source,token_class,metric]")
    layers, _, heads, _, _, _ = real.shape
    pseudo = _array(z, "pseudo_metrics", real.shape)
    counts = [_array(z, p + "_sample_count", (layers, heads, hop, 2)) for p in ("real", "pseudo")]
    present = [_array(z, p + "_all_sources_present", (layers, heads, 2)) for p in ("real", "pseudo")]
    degenerate = _array(z, "pseudo_degenerate_by_head", (layers, heads)).astype(bool)
    changed = _array(z, "pseudo_changed_fraction_by_head", (layers, heads))
    if np.any((changed < 0) | (changed > 1)):
        raise ValueError("Pseudo changed fraction outside [0,1]")
    output = {"geometry.pseudo_degenerate_fraction": float(degenerate.mean()),
              "geometry.pseudo_label_changed_fraction": float(changed.mean())}
    for kv, kv_name in enumerate(("K", "V")):
        for kind, kind_name in enumerate(("prompt", "latent")):
            prefix = f"geometry.{kv_name}.{kind_name}."
            valid_real = (counts[0][..., kind] >= 2) & (real[:, kv, :, :, kind, 2] > 0) & present[0][..., kind, None]
            valid_pseudo = (counts[1][..., kind] >= 2) & (pseudo[:, kv, :, :, kind, 2] > 0) & present[1][..., kind, None] & ~degenerate[..., None]
            paired = valid_real & valid_pseudo
            for metric, metric_name in enumerate(("participation_ratio", "entropy_effective_rank")):
                r = real[:, kv, :, :, kind, metric]
                p = pseudo[:, kv, :, :, kind, metric]
                for label, values, valid in (("real", r, valid_real), ("pseudo", p, valid_pseudo), ("real_minus_pseudo", r - p, paired)):
                    # Equal weighting of source means, after repeated layer/head reduction.
                    by_source = [_mean(values[..., s], valid[..., s]) for s in range(hop)]
                    output[prefix + label + "." + metric_name] = _mean([v for v in by_source if v is not None])
                    for source, value in enumerate(by_source):
                        output[prefix + label + "." + ROLES[source] + "." + metric_name] = value
            output[prefix + "real_valid_component_fraction"] = float(valid_real.mean())
            output[prefix + "pseudo_valid_component_fraction"] = float(valid_pseudo.mean())
    overlaps = [_array(z, p + "_subspace_overlap", (layers, 2, heads, 2, hop, hop)) for p in ("real", "pseudo")]
    ranks = [_array(z, p + "_subspace_overlap_rank", overlaps[0].shape) for p in ("real", "pseudo")]
    if any(np.any((r < 0) | (r > 4)) for r in ranks):
        raise ValueError("Overlap is not capped at top4")
    for kv, kv_name in enumerate(("K", "V")):
        for kind, kind_name in enumerate(("prompt", "latent")):
            prefix = f"geometry.{kv_name}.{kind_name}."
            by_label = {name: [] for name in ("real", "pseudo", "real_minus_pseudo")}
            for left in range(hop):
                for right in range(left + 1, hop):
                    r, p = [a[:, kv, :, kind, left, right] for a in overlaps]
                    vr = (ranks[0][:, kv, :, kind, left, right] > 0) & present[0][..., kind]
                    vp = (ranks[1][:, kv, :, kind, left, right] > 0) & present[1][..., kind] & ~degenerate
                    for label, values, mask in (("real", r, vr), ("pseudo", p, vp), ("real_minus_pseudo", r - p, vr & vp)):
                        by_label[label].append(_mean(values, mask))
            for label, pairs in by_label.items():
                output[prefix + label + ".top4_subspace_overlap_offdiag"] = _mean([v for v in pairs if v is not None])
    return output


def reduce_readout(z, hop):
    """True-source errors; block0 sink/self stays explicit and separate."""
    energy = _array(z, "source_error_mse")
    if energy.ndim != 3 or energy.shape[-1] != hop + 1 or np.any(energy < -1e-14):
        raise ValueError("Readout source shape/energy mismatch")
    lh = energy.shape[:2]
    total = _array(z, "total_error_mse", lh)
    cross = _array(z, "source_error_cross_matrix", lh + (hop + 1, hop + 1))
    cos = _array(z, "source_error_cosine_matrix", cross.shape)
    cancellation = _array(z, "cancellation", lh)
    zero = _array(z, "zero_error", lh).astype(bool)
    if np.any(total < -1e-14) or np.any(np.abs(cos) > 1 + 1e-10):
        raise ValueError("Invalid readout MSE/cosine")
    if not np.allclose(np.diagonal(cross, axis1=-2, axis2=-1), energy, rtol=1e-9, atol=1e-12) or not np.allclose(cross.sum((-2, -1)), total, rtol=1e-9, atol=1e-12):
        raise ValueError("Readout cross-term identity mismatch")
    source_energy = energy[..., 1:].sum(-1)
    true_total = cross[..., 1:, 1:].sum((-2, -1))
    valid_true = source_energy > 0
    true_cancel = np.zeros_like(source_energy)
    np.divide(true_total, source_energy, out=true_cancel, where=valid_true)
    true_cancel = 1 - true_cancel
    output = {"readout.total_error_mse": _mean(total),
              "readout.true_source_error_mse": _mean(energy[..., 1:]),
              "readout.sink_self_error_mse": _mean(energy[..., 0]),
              "readout.block_error_mse_including_sink_self": _mean(energy),
              "readout.true_source_error_energy_sum": _mean(source_energy),
              "readout.true_source_cross_term_sum": _mean(true_total - source_energy),
              "readout.cancellation_including_sink_self": _mean(cancellation, ~zero),
              "readout.cancellation_true_sources": _mean(true_cancel, valid_true),
              "readout.zero_error_fraction": float(zero.mean())}
    for source in range(hop):
        output["readout." + ROLES[source] + ".error_mse"] = _mean(energy[..., source + 1])
    pair_means = []
    for left in range(1, hop + 1):
        for right in range(left + 1, hop + 1):
            valid = (energy[..., left] > 0) & (energy[..., right] > 0)
            value = _mean(cos[..., left, right], valid)
            output[f"readout.{ROLES[left-1]}_vs_{ROLES[right-1]}.error_cosine"] = value
            pair_means.append(value)
    output["readout.true_source_error_cosine_offdiag"] = _mean([v for v in pair_means if v is not None])
    return output


def missing_receiver():
    return {"receiver." + name: None for name in RECEIVER_NAMES}


def reduce_receiver(z, prefix=""):
    kl = _array(z, prefix + "full_to_selected_kl")
    hidden = _array(z, prefix + "hidden_relative_l2", kl.shape)
    cosine = _array(z, prefix + "hidden_cosine", kl.shape)
    zero = _array(z, prefix + "full_hidden_zero", kl.shape).astype(bool)
    if np.any(kl < 0) or np.any(hidden < 0) or np.any(np.abs(cosine) > 1 + 1e-10):
        raise ValueError("Invalid receiver fidelity")
    return {"receiver.full_to_selected_kl": _mean(kl),
            "receiver.hidden_relative_l2": _mean(hidden, ~zero),
            "receiver.hidden_cosine": _mean(cosine, ~zero),
            "receiver.full_hidden_zero_fraction": float(zero.mean())}


def read_diagnostic(attempt, receipt, *, method, hop, kind, source_sha=None, position_sha=None, request_ids=()):
    """Verify one file, load only required NPZ members, immediately reduce it."""
    if not receipt.get("readback_verified") or receipt.get("raw_kv_retained") is not False:
        raise ValueError("Diagnostic writer proof is missing")
    path = _safe_path(attempt, receipt["file"])
    if file_sha(path) != receipt["sha256"] or path.stat().st_size != receipt["bytes"]:
        raise ValueError("Diagnostic SHA/size mismatch")
    with zipfile.ZipFile(path) as archive:
        if any(i.file_size > 64 * 1024 * 1024 for i in archive.infolist()):
            raise ValueError("Diagnostic member exceeds compact-writer size bound")
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(receipt["arrays"]) or "metadata_json" not in z.files:
            raise ValueError("Diagnostic member receipt mismatch")
        raw = z["metadata_json"]
        if raw.ndim != 0 or raw.dtype.kind not in "US":
            raise ValueError("Diagnostic metadata must be a scalar JSON string")
        metadata = json.loads(str(raw.item()))
        if metadata.get("method") != method or metadata.get("hop") != hop:
            raise ValueError("Diagnostic metadata method/hop mismatch")
        if source_sha is not None and metadata.get("source_cache_sha") != source_sha:
            raise ValueError("Diagnostic source-cache provenance mismatch")
        if position_sha is not None and metadata.get("position_sha") != position_sha:
            raise ValueError("Diagnostic selected-position provenance mismatch")
        if kind == "geometry":
            if metadata.get("geometry_version") != 1 or metadata.get("feature_subspace_rank") != 4 or metadata.get("token_class_names") != ["prompt", "latent"] or metadata.get("metric_names") != ["participation_ratio", "entropy_effective_rank", "numerical_rank"] or metadata.get("source_names") != list(ROLES[:hop]):
                raise ValueError("Geometry metadata schema mismatch")
            metrics = reduce_geometry(z, hop)
        elif kind == "readout":
            if metadata.get("analysis") != "own_trajectory_same_state":
                raise ValueError("Own-trajectory readout scope mismatch")
            metrics = reduce_readout(z, hop)
        elif kind == "receiver":
            if metadata.get("analysis") != "own_trajectory_same_state" or metadata.get("receiver") != RECEIVERS[hop-1] or metadata.get("decode_position") != 0 or bool(metadata.get("first_hop_common_prefix")) != (hop == 1):
                raise ValueError("Receiver metadata scope mismatch")
            if any(metadata.get(name) not in request_ids for name in ("reference_probe_rid", "native_prefill_rid")):
                raise ValueError("Receiver native/probe request provenance mismatch")
            metrics = reduce_receiver(z)
        elif kind == "common":
            if metadata.get("analysis") != "same_full_state":
                raise ValueError("Common Full-state intervention scope mismatch")
            metrics = reduce_readout(z, hop)
            if hop > 1 and method != "Full":
                if metadata.get("receiver_probe_rid") not in request_ids:
                    raise ValueError("Common-state receiver probe provenance mismatch")
                metrics.update(reduce_receiver(z, "receiver_"))
            else:
                metrics.update(missing_receiver())
        else:
            raise ValueError("Unknown diagnostic kind")
    return metrics, metadata


def bootstrap_questions(records, *, iterations=1000, seed=42):
    """Percentile intervals resample questions; missing stays missing per metric."""
    if not records or iterations < 1:
        raise ValueError("Question bootstrap requires records and a positive repeat count")
    names = sorted({name for row in records for name in row})
    values = np.asarray([[row.get(name) if row.get(name) is not None else np.nan for name in names] for row in records], dtype=np.float64)
    if np.isinf(values).any():
        raise ValueError("Non-finite question scalar")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(records), size=(iterations, len(records)))
    counts = np.zeros((iterations, len(records)), dtype=np.int16)
    for question in range(len(records)):
        counts[:, question] = np.count_nonzero(draws == question, axis=1)
    result = {}
    for start in range(0, len(names), 32):
        block = values[:, start:start+32]
        valid = np.isfinite(block)
        denominator = counts @ valid.astype(np.float64)
        numerator = counts @ np.nan_to_num(block)
        boot = np.full(numerator.shape, np.nan)
        np.divide(numerator, denominator, out=boot, where=denominator > 0)
        for j, name in enumerate(names[start:start+32]):
            available = block[:, j][valid[:, j]]
            finite = boot[:, j][np.isfinite(boot[:, j])]
            n = len(available)
            result[name] = dict(mean=float(available.mean()) if n else None,
                ci95_lower=float(np.quantile(finite, .025)) if n >= 2 and len(finite) else None,
                ci95_upper=float(np.quantile(finite, .975)) if n >= 2 and len(finite) else None,
                n_questions=n, missing_questions=len(records)-n, bootstrap_valid_replicates=len(finite),
                status="available" if n >= 2 else ("one_question_no_ci" if n else "unavailable"))
    return result


def require_final_inputs(run_root):
    """Gates are checked before any diagnostic NPZ is opened or reduced."""
    root = Path(run_root).resolve()
    lock_path = root / "run_lock.json"
    source_sha = file_sha(lock_path)
    lock = load_json(lock_path)
    for relative, expected in lock["files"].items():
        path = _safe_path(root, relative)
        if file_sha(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("Frozen run source/input lock mismatch")
    audit = load_json(root / "formal_audit_verified.json")
    if not audit.get("passed") or audit.get("phase") != "formal" or audit.get("keys") != 1440 or audit.get("cases") != 120 or audit.get("own_reserved") != 0 or not audit.get("all_owned_services_stopped") or audit.get("accuracy_computed") or audit.get("run_lock_sha256") != source_sha:
        raise ValueError("Successful final1440-key structural audit required")
    archive = load_json(root / "formal_archive_verified.json")
    flags = ("passed", "remote_stream_sha", "gzip_crc", "exact_members", "per_member_sha", "safe_full_extract_readback_sha")
    if archive.get("tag") != "formal" or not all(archive.get(k) for k in flags) or len(archive.get("archive_sha256", "")) != 64:
        raise ValueError("Verified full raw formal archive required")
    items = load_json(root / "questions.json")
    manifest = load_json(root / "data_manifest.json")
    plan = load_json(root / "plan.json")
    if plan.get("target_keys") != 1440 or tuple(plan.get("methods", ())) != METHODS or manifest.get("questions_sha256") != file_sha(root / "questions.json"):
        raise ValueError("Final diagnostics scope/input SHA mismatch")
    if len(items) != 120 or [x["index"] for x in items] != list(range(120)) or len({x["question_id"] for x in items}) != 120 or Counter(x["dataset_group"] for x in items) != Counter({g:20 for g in GROUPS}):
        raise ValueError("Final diagnostics requires six independent20-question groups")
    queue = load_json(root / "queue.json")
    if len(queue["tasks"]) != 120 or any(t["state"] != "complete" or t["methods"] for t in queue["tasks"]):
        raise ValueError("Complete120-question queue required")
    wanted = {(x["dataset"], x["question_id"], 10, m) for x in items for m in METHODS}
    descriptors = {}
    common_references = {}
    # Only private minimal references survive this streaming read; text/tokens are dropped.
    fields = ("index", "global_index", "dataset", "dataset_group", "question_id", "category", "is_diamond", "method", "source_lock_sha", "diagnostic_files", "common_state_diagnostic_files", "common_state_diagnostic_request_ids", "request_ids", "transfers", "shared_planner_kv_sha", "fork_rng_sha", "shared_request_ids")
    identity_fields = ("index", "global_index", "dataset", "dataset_group", "question_id", "category", "is_diamond")
    for path in sorted((root / "attempts").glob("*/results.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                index = row["index"]
                key = (row["dataset"], row["question_id"], row["latent_steps"], row["method"])
                if key not in wanted or key in descriptors or not isinstance(index, int) or not 0 <= index < 120:
                    raise ValueError("Unexpected/duplicate result key")
                item = items[index]
                if row.get("seed") != 42 or row.get("status") != "ok" or row.get("source_lock_sha") != source_sha or any(row[k] != item[k] for k in identity_fields):
                    raise ValueError("Result identity/protocol/source mismatch")
                descriptor = {name:row[name] for name in fields}
                # Transfer selector reports and repeated common receipts can be
                # large. Keep only needed provenance; share one canonical list.
                descriptor["transfers"] = [{"position_sha":t["position_sha"]} for t in row["transfers"]]
                signature = _receipt_signature(row["common_state_diagnostic_files"])
                references = common_references.get(index)
                if references is None:
                    references = (signature,row["common_state_diagnostic_files"],row["common_state_diagnostic_request_ids"],path.parent)
                    common_references[index] = references
                elif references[0] != signature or references[2] != row["common_state_diagnostic_request_ids"] or references[3] != path.parent:
                    raise ValueError("Common-state references differ across question methods")
                descriptor["common_state_diagnostic_files"] = references[1]
                descriptor["common_state_diagnostic_request_ids"] = references[2]
                descriptor["_attempt"] = path.parent
                descriptors[key] = descriptor
    if set(descriptors) != wanted or len(descriptors) != 1440:
        raise ValueError("All1440 exact unique keys required before diagnostics")
    minimal_items = [{k:x[k] for k in identity_fields} for x in items]
    return root, minimal_items, descriptors, dict(run_lock_sha256=source_sha,
        structural_audit_sha256=file_sha(root / "formal_audit_verified.json"),
        raw_archive_proof_sha256=file_sha(root / "formal_archive_verified.json"),
        raw_archive_sha256=archive["archive_sha256"], data_manifest_sha256=file_sha(root / "data_manifest.json"))


def _receipt_signature(receipts):
    return tuple(sorted((f["file"], f["sha256"], f["bytes"]) for f in receipts))


def reduce_question(item, descriptors):
    """Consume one question's files; retain only scalar reductions and small hashes."""
    rows = {m: descriptors[(item["dataset"], item["question_id"], 10, m)] for m in METHODS}
    full = rows["Full"]
    common_signature = _receipt_signature(full["common_state_diagnostic_files"])
    if len(common_signature) != 36 or len(set(common_signature)) != 36:
        raise ValueError("Expected36 unique common-state diagnostic files per question")
    own, geometry, geo_meta, receivers = {}, {}, {}, {}
    verified = 0
    for method, row in rows.items():
        if _receipt_signature(row["common_state_diagnostic_files"]) != common_signature or row["common_state_diagnostic_request_ids"] != full["common_state_diagnostic_request_ids"] or row["_attempt"] != full["_attempt"]:
            raise ValueError("Common-state diagnostic references differ across methods")
        if any(row[k] != full[k] for k in ("shared_planner_kv_sha", "fork_rng_sha", "shared_request_ids")):
            raise ValueError("Methods did not start from the same prefix/RNG")
        if len(row["diagnostic_files"]) != 9 or len(row["transfers"]) != 3:
            raise ValueError("Each trajectory requires9 diagnostic files and3 transfers")
        by_name = {Path(f["file"]).name:f for f in row["diagnostic_files"]}
        if len(by_name) != 9:
            raise ValueError("Duplicate own-trajectory diagnostic filename")
        own[method], geometry[method], geo_meta[method], receivers[method] = {}, {}, {}, {}
        for hop in (1, 2, 3):
            combined = {}
            for kind in ("geometry", "readout", "receiver"):
                filename = f"{method}_hop{hop}_{kind}.npz"
                if filename not in by_name:
                    raise ValueError("Missing expected own-trajectory diagnostic file")
                metrics, metadata = read_diagnostic(row["_attempt"], by_name[filename], method=method, hop=hop, kind=kind,
                    source_sha=geo_meta[method].get(hop, {}).get("source_cache_sha") if kind == "readout" else None,
                    position_sha=row["transfers"][hop-1]["position_sha"] if kind == "readout" else None,
                    request_ids=row["request_ids"])
                if kind == "geometry":
                    geometry[method][hop], geo_meta[method][hop] = metrics, metadata
                if kind == "receiver":
                    receivers[method][hop] = metrics
                combined.update(metrics)
                verified += 1
            own[method][hop] = combined
    if any(geo_meta[m][1]["source_cache_sha"] != geo_meta["Full"][1]["source_cache_sha"] for m in METHODS):
        raise ValueError("Hop1 geometry snapshots differ across shared-prefix methods")
    common = {m:{} for m in METHODS}
    by_name = {Path(f["file"]).name:f for f in full["common_state_diagnostic_files"]}
    for method in METHODS:
        for hop in (1, 2, 3):
            name = f"common_hop{hop}_{method}.npz"
            if name not in by_name:
                raise ValueError("Missing Full-state intervention file")
            metrics, _ = read_diagnostic(full["_attempt"], by_name[name], method=method, hop=hop, kind="common",
                source_sha=geo_meta["Full"][hop]["source_cache_sha"], request_ids=full["common_state_diagnostic_request_ids"])
            if hop == 1:
                # These actual prefills are precisely the shared Full-prefix interventions.
                metrics.update(receivers[method][hop])
            common[method][hop] = dict(geometry["Full"][hop], **metrics)
            verified += 1
    return own, common, verified


def build_summary(run_root, *, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED):
    root, items, rows, provenance = require_final_inputs(run_root)
    collected = {group:{scope:{m:{hop:[] for hop in (1, 2, 3)} for m in METHODS}
                       for scope in ("own_trajectory", "common_full_state")} for group in GROUPS}
    verified_files = 0
    for item in items:
        own, common, count = reduce_question(item, rows)
        verified_files += count
        for scope, results in (("own_trajectory", own), ("common_full_state", common)):
            for method in METHODS:
                for hop in (1, 2, 3):
                    collected[item["dataset_group"]][scope][method][hop].append(results[method][hop])
    datasets = {}
    for group in GROUPS:
        datasets[group] = {"n_questions":20}
        for scope in ("own_trajectory", "common_full_state"):
            datasets[group][scope] = {m:{"hops":{str(h):dict(n_questions=20,metrics=bootstrap_questions(
                collected[group][scope][m][h],iterations=iterations,seed=seed)) for h in (1,2,3)}} for m in METHODS}
    return dict(complete=True, questions=120, keys=1440, groups=list(GROUPS), methods=list(METHODS),
        hops=3, verified_unique_diagnostic_files=verified_files, model_forward_calls=0,
        postprocessing_source_sha256=file_sha(Path(__file__)), provenance=provenance,
        exploratory=True, no_pooled_accuracy=True, no_pooled_primary_effect=True, no_significance_tests=True,
        independent_unit="question within dataset_group; layers, heads, sources and decode positions are repeated measurements",
        bootstrap=dict(iterations=iterations,seed=seed,interval="95% percentile, descriptive",resampling="question indices within each of the six groups; same draws across metrics",missing="undefined components excluded before question reduction; missing question scalars stay missing; fewer than2 questions has no CI"),
        interpretation=dict(own_trajectory="Each method's own pre-deletion cache and its receiver prefill; later hops are not a common Full-chain state.",
            common_full_state="All12 selectors replay the same Full pre-deletion cache at each hop. Input geometry is shared Full geometry, not geometry after selection or value backfill.",
            common_receiver="Hop1 uses each method's actual prefill from the common prefix. Hops2/3 use explicit same-Full-state compact-cache probes for11 non-Full methods; Full self-reference was not separately recorded and remains unavailable.",
            geometry="Centered participation/entropy ranks reduced across layers/heads per source, then equally across valid sources; prompt and latent, K and V remain separate.",
            pseudo="Real-minus-pseudo is paired within question using matching valid components; degenerate/unchanged pseudo labels excluded. First-hop pseudo comparisons are normally unavailable.",
            overlap="Off-diagonal unique source pairs only, normalized overlap at min(numerical ranks,4); rank-zero or missing pairs excluded; no single-source overlap.",
            readout="True-source metrics exclude block0 sink/self. Full total MSE/cancellation is separately retained including block0. Cosines omit zero-energy source pairs; cancellation omits zero total source energy.",
            receiver="KL(Full-like pre-deletion reference || selected) and hidden relative L2 at receiver prefill/decode_position0 only; hidden relative/cosine omitted for zero reference norm."),
        datasets=datasets)


def markdown_report(summary):
    def number(value):
        return "NA" if value is None else f"{value:.5g}"
    def cell(metrics, name):
        stat = metrics[name]
        if stat["mean"] is None:
            return "NA (0/20)"
        interval = "no CI" if stat["ci95_lower"] is None else f"{number(stat['ci95_lower'])}, {number(stat['ci95_upper'])}"
        return f"{number(stat['mean'])} [{interval}] (n={stat['n_questions']})"
    lines = ["# Final role-geometry diagnostics", "", "Exploratory descriptive summaries. Six dataset groups remain independent; no pooled effect, accuracy, or significance test is computed.", "",
             f"Each estimate first reduces repeated layer/head/source measurements within a question. Intervals resample questions {summary['bootstrap']['iterations']} times with seed {summary['bootstrap']['seed']} and are descriptive 95% percentile intervals. NA means the endpoint was undefined or unrecorded, not zero.", "",
             "The JSON companion contains K/V participation ratio and entropy effective rank, paired real-minus-pseudo changes, valid top4 source overlap, per-source MSE, cancellation, source-error cosine, and receiver fidelity for every method and handoff."]
    for group in GROUPS:
        lines += ["", "## " + group, ""]
        for scope, title in (("own_trajectory", "Own trajectories"), ("common_full_state", "Common Full-state interventions")):
            lines += ["### " + title, "", "| Method | Hop | True-source MSE | Total MSE | Cancellation incl. sink/self | Source cosine | Receiver KL | Hidden relative L2 |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for method in METHODS:
                for hop in (1,2,3):
                    metrics = summary["datasets"][group][scope][method]["hops"][str(hop)]["metrics"]
                    values = [cell(metrics,k) for k in ("readout.true_source_error_mse","readout.total_error_mse","readout.cancellation_including_sink_self","readout.true_source_error_cosine_offdiag","receiver.full_to_selected_kl","receiver.hidden_relative_l2")]
                    lines.append("| " + method + " | " + str(hop) + " | " + " | ".join(values) + " |")
        lines += ["", "Own-trajectory hop2/3 references differ by method; common Full-state interventions hold the prefix state fixed. Common Full hop2/3 receiver identity probes were not recorded and remain NA."]
    lines += ["", "No raw vectors, question text, answers, or generated tokens are included in this report. Original private NPZ files remain the source for vector-valued diagnostics.", ""]
    return "\n".join(lines)


def summarize_run(run_root, *, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED):
    """API: verify final gates, summarize, write JSON/MD; return compact proof."""
    root = Path(run_root).resolve()
    started = time.time()
    summary = build_summary(root, iterations=iterations, seed=seed)
    summary.update(started_unix=started, finished_unix=time.time())
    targets = {root/"final_diagnostics_summary.json":json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",
               root/"final_diagnostics_summary.md":markdown_report(summary)}
    for target,text in targets.items():
        temporary = target.with_name(target.name+".tmp")
        temporary.write_text(text,encoding="utf-8",newline="\n")
        temporary.replace(target)
    return dict(complete=True,questions=120,keys=1440,dataset_groups=6,methods=12,hops=3,
        verified_diagnostic_files=summary["verified_unique_diagnostic_files"],
        summary_sha256=file_sha(root/"final_diagnostics_summary.json"),
        report_sha256=file_sha(root/"final_diagnostics_summary.md"),model_forward_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root",type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_run(args.run_root),sort_keys=True))


if __name__ == "__main__":
    main()
