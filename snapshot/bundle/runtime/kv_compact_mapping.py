"""Physically select rows from a full-coordinate cache without re-rotating K.

This extends suffix slicing to ordered, possibly disjoint, stage blocks.
The source cache must still use its original full coordinates.
"""


def build_mapping(history_length, retained_positions):
    positions = list(retained_positions)
    assert history_length > 0 and positions
    assert positions == sorted(set(positions)), 'require unique increasing positions'
    assert 0 <= positions[0] <= positions[-1] < history_length
    return dict(original_history_length=history_length,
        storage_to_original=positions,
        compact_history_length=len(positions),
        next_new_original_position=history_length,
        continuation_rope_offset=history_length-len(positions))


def compact_cache_with_mapping(full_coordinate_cache, retained_positions):
    import torch
    from transformers.cache_utils import DynamicCache
    from kv_packet import legacy
    mapping = build_mapping(int(full_coordinate_cache.get_seq_length()), retained_positions)
    pairs = []
    for pair in legacy(full_coordinate_cache):
        selected = []
        for tensor in pair:
            assert tensor.shape[-2] == mapping['original_history_length']
            index = torch.tensor(mapping['storage_to_original'], dtype=torch.long, device=tensor.device)
            selected.append(tensor.index_select(-2, index).detach().contiguous())
        pairs.append(tuple(selected))
    return DynamicCache.from_legacy_cache(tuple(pairs)), mapping
