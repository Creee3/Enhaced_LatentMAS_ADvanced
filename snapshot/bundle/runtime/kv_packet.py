"""Self-contained lossless BF16 KV packets; no network-speed claim.

The decoder accepts only packet bytes, independently of the sender cache.
The inference caller must use the returned new cache for its next request.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import time

MAGIC = b'LMKV0001'


def raw_tensor(tensor):
    import torch
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def legacy(cache):
    return cache.to_legacy_cache() if hasattr(cache, 'to_legacy_cache') else cache


def signature(cache, length=None):
    h = hashlib.sha256()
    for pair in legacy(cache):
        for tensor in pair:
            h.update(raw_tensor(tensor if length is None else tensor[..., :length, :]))
    return h.hexdigest()


def encode(cache, route):
    import torch
    layers = legacy(cache)
    assert layers and len(layers) <= 128
    shapes, parts, digests = [], [], []
    positions = layers[0][0].shape[-2]
    for pair in layers:
        assert len(pair) == 2
        shape_pair = []
        for tensor in pair:
            assert tensor.ndim == 4 and tensor.shape[0] == 1 and tensor.shape[-2] == positions
            assert tensor.dtype == torch.bfloat16 and torch.isfinite(tensor).all()
            data = raw_tensor(tensor)
            parts.append(data)
            digests.append(hashlib.sha256(data).hexdigest())
            shape_pair.append(list(tensor.shape))
        shapes.append(shape_pair)
    header = json.dumps(dict(version=1, dtype='bfloat16', shapes=shapes, tensor_sha256=digests,
                             route=route, external_reference=False), sort_keys=True, separators=(',', ':')).encode()
    assert len(header) < 65536
    packet = MAGIC+struct.pack('<I', len(header))+header+b''.join(parts)
    return packet, sum(map(len, parts))


def decode(packet, device='cpu'):
    import torch
    assert packet[:8] == MAGIC and len(packet) >= 12
    size = struct.unpack('<I', packet[8:12])[0]
    assert 0 < size < 65536 and 12+size < len(packet)
    header = json.loads(packet[12:12+size])
    assert header['version'] == 1 and header['dtype'] == 'bfloat16' and header['external_reference'] is False
    offset, pairs, part_index = 12+size, [], 0
    assert 0 < len(header['shapes']) <= 128
    expected_position = header['shapes'][0][0][-2]
    for shapes in header['shapes']:
        assert len(shapes) == 2
        pair = []
        for shape in shapes:
            assert len(shape) == 4 and shape[0] == 1 and all(type(n) is int and n > 0 for n in shape)
            assert shape[-2] == expected_position
            elements = 1
            for n in shape:
                elements *= n
            size = elements*2
            assert offset+size <= len(packet)
            data = packet[offset:offset+size]
            assert hashlib.sha256(data).hexdigest() == header['tensor_sha256'][part_index]
            tensor = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(shape).clone().to(device)
            assert torch.isfinite(tensor).all()
            pair.append(tensor)
            offset += size
            part_index += 1
        pairs.append(tuple(pair))
    assert offset == len(packet), 'truncated or trailing bytes'
    from transformers.cache_utils import DynamicCache
    return DynamicCache.from_legacy_cache(tuple(pairs)), header


def roundtrip(cache, route, evidence_path=None):
    import torch
    device = legacy(cache)[0][0].device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.monotonic()
    packet, payload_bytes = encode(cache, route)
    encoded = time.monotonic()
    received, header = decode(packet, device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    finished = time.monotonic()
    assert received is not cache and signature(received) == signature(cache)
    packet_sha = hashlib.sha256(packet).hexdigest()
    ack = json.dumps(dict(accepted=True, packet_sha256=packet_sha, receiver=route['receiver']), sort_keys=True).encode()
    if evidence_path is not None:
        path = Path(evidence_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(packet)
    info = dict(route=route, positions=received.get_seq_length(), packet_bytes=len(packet), raw_kv_bytes=payload_bytes,
                acknowledgment_bytes=len(ack), communication_bytes=len(packet)+len(ack), packet_sha256=packet_sha,
                tensor_signature=signature(received), encode_including_D2H_seconds=encoded-started,
                decode_including_H2D_seconds=finished-encoded, boundary_seconds=finished-started,
                receiver_uses_new_decoded_cache=True, payload_only_decoder=True,
                transport='logical-agent serialization boundary on one GPU; not measured network', network_latency_measured=False)
    return received, info


def sparse_roundtrip(cache, keep_positions, route):
    """Serialize only selected sequence rows and reconstruct original coordinates.

    The receiver gets a full-coordinate cache with zero-filled dropped rows and
    an explicit position mask. This keeps RoPE/cache coordinates unchanged
    while making the wire payload proportional to the selected rows.
    """
    import torch
    layers = legacy(cache)
    original_position_count = layers[0][0].shape[-2]
    keep = sorted({int(i) for i in keep_positions if 0 <= int(i) < original_position_count})
    shapes, parts, digests = [], [], []
    for pair in layers:
        shape_pair = []
        for tensor in pair:
            assert tensor.ndim == 4 and tensor.shape[0] == 1
            assert tensor.dtype == torch.bfloat16 and torch.isfinite(tensor).all()
            selected = tensor[..., keep, :]
            data = raw_tensor(selected)
            parts.append(data)
            digests.append(hashlib.sha256(data).hexdigest())
            shape_pair.append(list(selected.shape))
        shapes.append(shape_pair)
    header = json.dumps(dict(version=2, dtype='bfloat16', original_positions=original_position_count,
                             keep_positions=keep, shapes=shapes, tensor_sha256=digests,
                             route=route, external_reference=False), sort_keys=True,
                        separators=(',', ':')).encode()
    packet = MAGIC+struct.pack('<I', len(header))+header+b''.join(parts)
    offset, pairs, part_index = 12+len(header), [], 0
    for shape_pair in shapes:
        pair = []
        for shape in shape_pair:
            elements = 1
            for n in shape:
                elements *= n
            size = elements*2
            data = packet[offset:offset+size]
            assert hashlib.sha256(data).hexdigest() == digests[part_index]
            selected = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(shape).clone().to(layers[0][0].device)
            full_shape = list(shape)
            full_shape[-2] = original_position_count
            full = torch.zeros(full_shape, dtype=selected.dtype, device=selected.device)
            if keep:
                full[..., torch.tensor(keep, device=full.device), :] = selected
            pair.append(full)
            offset += size
            part_index += 1
        pairs.append(tuple(pair))
    assert offset == len(packet)
    from transformers.cache_utils import DynamicCache
    received = DynamicCache.from_legacy_cache(tuple(pairs))
    assert received is not cache
    # Dropped rows are intentionally zeroed, so compare only kept rows.
    for old_pair, new_pair in zip(layers, legacy(received)):
        for old, new in zip(old_pair, new_pair):
            if keep:
                assert torch.equal(old[..., keep, :], new[..., keep, :])
            if len(keep) < original_position_count:
                dropped = [i for i in range(original_position_count) if i not in set(keep)]
                assert torch.count_nonzero(new[..., dropped, :]) == 0
    packet_sha = hashlib.sha256(packet).hexdigest()
    ack = json.dumps(dict(accepted=True, packet_sha256=packet_sha, receiver=route['receiver']), sort_keys=True).encode()
    info = dict(route=route, positions=original_position_count, selected_positions=len(keep),
                packet_bytes=len(packet), raw_kv_bytes=sum(len(x) for x in parts),
                acknowledgment_bytes=len(ack), communication_bytes=len(packet)+len(ack),
                packet_sha256=packet_sha, tensor_signature=signature(received),
                receiver_uses_new_decoded_cache=True, payload_only_decoder=True,
                transport='logical-agent serialization boundary on one GPU; selected rows only; not measured network',
                network_latency_measured=False)
    return received, [1 if i in set(keep) else 0 for i in range(original_position_count)], info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    packet = Path(args.decode).read_bytes()
    cache, header = decode(packet)
    from runtime_common import save
    save(args.receipt, dict(packet_sha256=hashlib.sha256(packet).hexdigest(), tensor_signature=signature(cache),
                            positions=cache.get_seq_length(), route=header['route'], independent_decoder_process=True))
