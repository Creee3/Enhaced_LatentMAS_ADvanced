"""Pinned author relay operators with cache-constructor-only AST adaptation.

No model, tokenizer, cloud, or dataset imports. The public select_author API is
stateless: len(stages) reconstructs the author's one-time sink flag at a handoff.
The external source is immutable and SHA-checked before every invocation.
"""
from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

COMMIT = "36f43183e84d2fbf3b84c9bf4e3e06b7c442a523"
_PINNED = {
    "BaseKVCompressor.py": "b441059985f3355e3eb4977fda2da544dbbfabb1ebd2b5c3e48cc7479e195907",
    "Full.py": "5c3f871e6c26f6060fb79f08029eb2c5591bb8100059318361e4dc14d4d99ed3",
    "GenerationOnly.py": "821fdfcca8174cdfc84ecd649587b9986b23c0df2e452aa256b8e1fe1c543cc2",
    "Headwise.py": "fce536dcade194019efb14aa65829f4369a0b5ba83aef1d5dc1efd1fa4901ece",
    "Layerwise.py": "dfb69af748ef476f2561ce77ca3f34e378f6ef8abe2587a9df5792853530a1f7",
    "HOBF.py": "d4e5965eb0ba1e5ac03c9131f710e252b9dc6d2dfd55debabdd6c86109663d03",
    "LOBF.py": "bd38896ac76344a85b1d61ef6f9b6bfb584c50134806a61c5fa2ae0dbef6c410",
}
_NAMES = {"Full": "Full", "Gen": "GenerationOnly", "H": "Headwise", "L": "Layerwise", "H-OBF": "HOBF", "L-OBF": "LOBF"}
_ALIASES = {v: k for k, v in _NAMES.items()}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def author_directory():
    """Use adjacent staged dependency, or the private local development copy."""
    explicit = os.environ.get("ROLE_GEOMETRY_AUTHOR_DIR")
    if explicit:
        return Path(explicit).resolve()
    here = Path(__file__).resolve().parent
    if (here / "author").is_dir():
        return here / "author"
    local = here.parents[2] / "advanced_MOC/experiments/latentmas_role_geometry120_20261110/author"
    if local.is_dir():
        return local
    raise FileNotFoundError("Stage author/ and author_source_lock.json beside author_ops.py")


def verify_author_sources(directory=None):
    directory = Path(directory) if directory is not None else author_directory()
    lock_path = directory.parent / "author_source_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock["commit"] != COMMIT:
        raise RuntimeError("Author dependency commit differs from the locked pilot revision")
    for name, expected in _PINNED.items():
        path = directory / name
        entry = lock["files"][name]
        if _sha(path) != expected or entry["sha256"] != expected or path.stat().st_size != entry["bytes"]:
            raise RuntimeError("Pinned author source mismatch: " + name)
    for name, entry in {**lock["files"], **lock.get("notices", {})}.items():
        path = directory / name
        if _sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            raise RuntimeError("Author source/notice lock mismatch: " + name)
    return directory, lock_path


def _canonical_mode(mode):
    result = _ALIASES.get(mode, mode)
    if result not in _NAMES:
        raise ValueError("Unsupported author method: " + str(mode))
    return result


@functools.lru_cache(maxsize=32)
def _author_class(directory, name, factory):
    directory = Path(directory)
    ns = {"__name__": "_pinned_author_" + name}
    base = directory / "BaseKVCompressor.py"
    exec(compile(base.read_text(encoding="utf-8"), str(base), "exec", dont_inherit=True), ns)

    class CacheFactory:
        def __new__(cls):
            return factory(tuple())

        @staticmethod
        def from_legacy_cache(values):
            return factory(tuple(values))

    ns["DynamicCache"] = CacheFactory
    path = directory / (name + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # No function or algorithm node is changed. Only these imports are replaced
    # by the exact original Base class and caller's cache constructor above.
    removed = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module in {"transformers", "compression_methods.BaseKVCompressor"}]
    if {n.module for n in removed} != {"transformers", "compression_methods.BaseKVCompressor"}:
        raise RuntimeError("Unexpected upstream import layout")
    tree.body = [n for n in tree.body if n not in removed]
    code_sha = hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec", dont_inherit=True), ns)
    return ns[name], code_sha


def load_author(mode, factory, directory=None):
    """Instantiate the unchanged author class with the locked paper parameters."""
    mode = _canonical_mode(mode)
    directory, lock_path = verify_author_sources(directory)
    cls, code_sha = _author_class(str(directory.resolve()), _NAMES[mode], factory)
    kwargs = {} if mode == "Full" else {"sink_size": 4, "kv_budget": 32}
    rank = {"H-OBF": 2, "L-OBF": 4}.get(mode, 0)
    if rank:
        kwargs.update(pca_rank=rank, inject_mode="uniform", scale_mode="ad_over_as", center=False)
    operator = cls(**kwargs)
    receipt = dict(commit=COMMIT, file_sha256=_PINNED[_NAMES[mode] + ".py"],
                   base_sha256=_PINNED["BaseKVCompressor.py"], source_lock_sha256=_sha(lock_path),
                   algorithm_ast_sha256=code_sha, cache_factory_only_adapter=True,
                   class_name=_NAMES[mode], sink_size=int(operator.sink_size), kv_budget=int(operator.kv_budget), pca_rank=rank,
                   inject_mode="uniform" if rank else None,
                   scale_mode="ad_over_as" if rank else None, center=False if rank else None)
    return operator, receipt


def _pairs(cache):
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    if hasattr(cache, "layers"):
        return tuple((x.keys, x.values) for x in cache.layers)
    if hasattr(cache, "key_cache"):
        return tuple(zip(cache.key_cache, cache.value_cache))
    return tuple(cache)


def _scores(attentions, layer, heads, history, prompt_len, total, device):
    """Independent sender-attention accumulation in native GQA head order."""
    query_heads = int(attentions[0][layer].shape[1])
    if query_heads % heads:
        raise ValueError("Pilot requires an integral number of query heads per KV head")
    max_len = max(int(step[layer].shape[-1]) for step in attentions)
    aggregate = torch.zeros((1, query_heads, max_len), device=device, dtype=torch.float32)
    for step in attentions:
        a = step[layer].to(device=device, dtype=torch.float32)
        if a.ndim != 3 or a.shape[:2] != (1, query_heads):
            raise ValueError("Author sender attention must be [1, query_heads, key_positions]")
        aggregate += F.pad(a, (0, max_len - a.shape[-1]))
    aggregate = F.pad(aggregate, (0, max(0, total - max_len)))[..., :total]
    prompt_scores = aggregate[..., history:history + prompt_len]
    if query_heads != heads:
        prompt_scores = prompt_scores.reshape(1, heads, query_heads // heads, prompt_len).sum(2)
    if not bool(torch.isfinite(prompt_scores).all()):
        raise RuntimeError("Non-finite sender attention")
    return prompt_scores[0]


def _expected_indices(xs, history, prompt_len, steps, sink, mode, attentions):
    layers, heads, total = len(xs), xs[0][0].shape[1], xs[0][0].shape[-2]
    device = xs[0][0].device
    k = min(32, prompt_len - sink)
    if mode == "Full":
        return torch.arange(total, device=device).view(1, 1, -1).expand(layers, heads, -1), prompt_len - sink
    left = torch.arange(history + sink, device=device).view(1, -1).expand(heads, -1)
    tail = torch.arange(total - steps, total, device=device).view(1, -1).expand(heads, -1)
    selected = []
    for layer in range(layers):
        if mode == "Gen":
            keep = torch.empty((heads, 0), device=device, dtype=torch.long)
        else:
            scores = _scores(attentions, layer, heads, history, prompt_len, total, device)
            if mode in {"H", "H-OBF"}:
                keep = torch.topk(scores[:, sink:], k, dim=-1).indices.sort(-1).values + sink
            else:
                scores = scores.sum(0)
                scores[:sink] = -torch.inf
                keep = torch.topk(scores, k, dim=-1).indices.sort(-1).values[None].expand(heads, -1)
            keep = keep + history
        selected.append(torch.cat((left, keep, tail), -1))
    return torch.stack(selected), 0 if mode == "Gen" else k


def _selection_metadata_equal(metadata, indices, history, sink, prompt_kept, device):
    measured = torch.tensor(metadata["selected_prompt_positions_matrix"][0], device=device, dtype=torch.long)
    expected = indices[..., history + sink:history + sink + prompt_kept] - history
    if not torch.equal(measured, expected):
        raise RuntimeError("Author selection metadata differs from independent attention selection")


def _source_guards(xs):
    guards = []
    for pair in xs:
        for tensor in pair:
            if tensor.is_inference():
                flat = tensor.reshape(-1)
                step = max(1, flat.numel() // 256)
                guards.append((tensor, None, flat[::step].clone(), step))
            else:
                guards.append((tensor, tensor._version, None, None))
    return guards


def _check_source_guards(guards):
    for tensor, version, sample, step in guards:
        if version is not None:
            if tensor._version != version:
                raise RuntimeError("Author compression mutated source KV")
        elif not torch.equal(tensor.reshape(-1)[::step], sample):
            raise RuntimeError("Author compression changed sampled inference-tensor KV")
    return "tensor_versions" if all(x[1] is not None for x in guards) else "versions_and_sampled_inference_values"


def _compact_metadata(metadata):
    result = {}
    for name, value in (metadata or {}).items():
        if not isinstance(value, torch.Tensor):
            continue
        finite = value.detach().double().reshape(-1)
        finite = finite[torch.isfinite(finite)]
        result[name] = dict(finite=int(finite.numel()), total=int(value.numel()),
                            mean=float(finite.mean()) if finite.numel() else None,
                            minimum=float(finite.min()) if finite.numel() else None,
                            maximum=float(finite.max()) if finite.numel() else None)
    return result


@torch.no_grad()
def select_author(xs, mapping, stages, mode, sender_attention, make_cache):
    """Run one author handoff, verify original KV/RoPE, return cache/map/ix/report.

    xs contains [batch=1, KV heads, storage positions, head dimension] pairs.
    mapping describes retained history before stages[-1] was appended. Stages
    holds the contiguous original-position provenance for this question's chain.
    Attention is the unchanged sender's ten native latent-step attention lists.
    """
    mode = _canonical_mode(mode)
    xs = _pairs(xs)
    if not xs or not (1 <= len(stages) <= 3):
        raise ValueError("Author handoff requires a nonempty cache and one to three stages")
    last = stages[-1]
    steps = int(last.get("latent_steps", 10))
    prompt_len = int(last["prompt_len"])
    if steps != 10 or prompt_len < 1 or int(stages[0]["original_before"]) != 0:
        raise ValueError("Pilot requires latent10, a nonempty prompt, and original position zero")
    for number, stage in enumerate(stages):
        if int(stage["original_end"]) != int(stage["original_before"]) + int(stage["prompt_len"]) + steps:
            raise ValueError("Original stage boundaries do not match prompt and latent lengths")
        if number and int(stage["original_before"]) != int(stages[number - 1]["original_end"]):
            raise ValueError("Original stage positions are not contiguous")
    shape = xs[0][0].shape
    if len(shape) != 4 or shape[0] != 1 or any(t.shape != shape for pair in xs for t in pair):
        raise ValueError("All pilot KV layers must have matching [1, heads, length, dimension] shapes")
    layers, heads, total, device = len(xs), int(shape[1]), int(shape[2]), xs[0][0].device
    history = total - prompt_len - steps
    if history < 0 or (mapping is None) != (len(stages) == 1):
        raise ValueError("History mapping and current handoff disagree")
    if mapping is None:
        if history:
            raise ValueError("First handoff cannot contain unaccounted history")
    elif mapping.shape != (layers, heads, history) or mapping.dtype != torch.long or mapping.device != device:
        raise ValueError("History mapping shape/dtype/device differs from retained KV")
    sink = min(4, prompt_len) if len(stages) == 1 else 0
    if mode not in {"Full", "Gen"}:
        if sender_attention is None or len(sender_attention) != steps or any(len(x) != layers for x in sender_attention):
            raise ValueError("Author attention selectors require ten complete sender attention steps")
    attentions = sender_attention if sender_attention is not None else []
    indices, prompt_kept = _expected_indices(xs, history, prompt_len, steps, sink, mode, attentions)
    fresh = torch.arange(int(last["original_before"]), int(last["original_end"]), device=device)
    fresh = fresh.view(1, 1, -1).expand(layers, heads, -1)
    all_positions = fresh if mapping is None else torch.cat((mapping, fresh), -1)
    if all_positions.shape != (layers, heads, total) or not bool((all_positions.diff(dim=-1) > 0).all()):
        raise ValueError("Original-position provenance is not ordered or complete")
    guards = _source_guards(xs)
    operator, author = load_author(mode, make_cache)
    # This is exactly the state that reset()+the preceding chain calls leaves.
    # Reconstruct it per invocation so no mutable compressor state crosses keys.
    operator._has_kept_sink = len(stages) > 1
    kwargs = dict(past_key_values=xs, latent_steps=steps, all_steps_attentions=attentions,
                  prompt_mask=torch.ones((1, prompt_len), dtype=torch.long, device=device))
    result, author_seconds, metadata = operator.compress(**kwargs)
    actual = _pairs(result)
    if len(actual) != layers:
        raise RuntimeError("Author output lost KV layers")
    if mode == "Full":
        result = make_cache(actual)
        actual = _pairs(result)
    if mode in {"H", "L"}:
        _selection_metadata_equal(metadata, indices, history, sink, prompt_kept, device)
    if mode in {"H-OBF", "L-OBF"}:
        reference_mode = "H" if mode == "H-OBF" else "L"
        reference, _ = load_author(reference_mode, make_cache)
        reference._has_kept_sink = len(stages) > 1
        reference_cache, _, reference_metadata = reference.compress(**kwargs)
        _selection_metadata_equal(reference_metadata, indices, history, sink, prompt_kept, device)
        for expected_pair, actual_pair in zip(_pairs(reference_cache), actual):
            if not torch.equal(expected_pair[0], actual_pair[0]):
                raise RuntimeError(mode + " selected positions differ from author " + reference_mode + " (including possible top-k tie differences)")
    positions = torch.gather(all_positions, 2, indices)
    changed_by_layer = []
    expected_length = indices.shape[-1]
    for layer, ((old_k, old_v), (new_k, new_v)) in enumerate(zip(xs, actual)):
        expected_shape = (1, heads, expected_length, old_k.shape[-1])
        if new_k.shape != expected_shape or new_v.shape != expected_shape or new_k.dtype != old_k.dtype or new_v.dtype != old_v.dtype:
            raise RuntimeError("Author output shape or dtype drift")
        gather_index = indices[layer][None, :, :, None].expand(expected_shape)
        if not torch.equal(torch.gather(old_k, 2, gather_index), new_k):
            raise RuntimeError("Author changed selected original K or position order")
        expected_v = torch.gather(old_v, 2, gather_index)
        changed = (new_v != expected_v).any(-1)
        allowed = torch.zeros_like(changed)
        if mode in {"H-OBF", "L-OBF"}:
            allowed[..., history + sink:history + sink + prompt_kept] = True
        if bool((changed & ~allowed).any()):
            raise RuntimeError("Author changed protected history/sink/latent V or eviction-only V")
        if not bool(torch.isfinite(new_k).all()) or not bool(torch.isfinite(new_v).all()):
            raise RuntimeError("Non-finite author KV output")
        changed_by_layer.append(int(changed.sum()))
    integrity_check = _check_source_guards(guards)
    if mapping is not None and not torch.equal(positions[..., :history], mapping):
        raise RuntimeError("Author failed to preserve history provenance")
    for stage in stages:
        latent = (positions >= int(stage["original_end"]) - steps) & (positions < int(stage["original_end"]))
        if not bool((latent.sum(-1) == steps).all()):
            raise RuntimeError("Author dropped protected latent positions")
    if not bool(((positions < min(4, int(stages[0]["prompt_len"]))).sum(-1) == min(4, int(stages[0]["prompt_len"]))).all()):
        raise RuntimeError("Author dropped the original global sink")
    if not bool((positions.diff(dim=-1) > 0).all()):
        raise RuntimeError("Author output original RoPE positions are not ordered")
    actual_bytes = sum(t.numel() * t.element_size() for pair in actual for t in pair)
    bytes_per_position = sum(t.shape[0] * t.shape[1] * t.shape[-1] * t.element_size() for pair in xs for t in pair)
    if actual_bytes != expected_length * bytes_per_position:
        raise RuntimeError("Author KV byte count drift")
    report = dict(method=mode, sender=last.get("role"), handoff=len(stages), retained=expected_length,
                  original_end=int(last["original_end"]), original_RoPE=True,
                  source_k_equal_measured=True, source_v_equal_measured=not any(changed_by_layer),
                  protected_v_equal_measured=True, history_preserved_equal_measured=True,
                  source_input_unmodified_measured=True, source_input_integrity_check=integrity_check, history_kept=history,
                  source_prompt_len=prompt_len, prompt_kept=prompt_kept, sink_kept=sink,
                  prompt_budget=32 if mode not in {"Full", "Gen"} else 0,
                  mandatory_recent=0, protected_latent=steps * len(stages), protected_sink=min(4, int(stages[0]["prompt_len"])),
                  sender_attention_steps=len(attentions), independent_author_selection_equal=True,
                  layer_heads_share_positions=torch.equal(positions, positions[:, :1].expand_as(positions)),
                  value_backfill=mode in {"H-OBF", "L-OBF"}, pca_rank=author["pca_rank"],
                  changed_prompt_value_rows_by_layer=changed_by_layer,
                  actual_kv_bytes=actual_bytes, full_bytes_from_original_positions=int(last["original_end"]) * bytes_per_position,
                  position_sha=hashlib.sha256(positions.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                  author_seconds=float(author_seconds), author=author,
                  author_metrics=_compact_metadata(metadata))
    return result, positions, indices, report
