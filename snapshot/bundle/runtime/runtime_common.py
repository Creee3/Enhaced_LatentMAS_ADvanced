"""Durable evidence and the existing shared Qwen ledger. No call quota."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
LOCK_ROOT = ROOT.parent if ROOT.name == 'runtime' else ROOT.parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    with temporary.open('wb') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('ab') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())


def verify_source():
    lock = json.loads((LOCK_ROOT/'source_lock.json').read_text())
    for name, entry in lock['files'].items():
        digest = entry['sha256'] if isinstance(entry, dict) else entry
        expected_bytes = entry.get('bytes') if isinstance(entry, dict) else None
        path = LOCK_ROOT/name
        assert path.exists(), 'source missing: '+name
        if expected_bytes is not None:
            assert path.stat().st_size == expected_bytes, 'source size mismatch: '+name
        assert sha(path) == digest, 'source mismatch: '+name
    return lock


class Ledger:
    def __init__(self, config, trace, phase):
        self.config, self.trace, self.phase = config, trace, phase
        self.path = Path(config['budget']['ledger_path'])

    def check(self):
        if (ROOT/'STOP').exists():
            raise RuntimeError('STOP requested')

    def update(self, fn):
        with self.path.with_name('budget.lock').open('a') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            value = json.loads(self.path.read_text())
            result = fn(value)
            save(self.path, value)
            return result

    def begin(self, bound, context):
        self.check()
        def change(value):
            charged = value['actual_tokens']+value['reserved_tokens']+value['unknown_cost_upper_bound_tokens']
            assert bound > 0 and charged+bound <= self.config['budget']['cumulative_position_limit'], 'position budget exhausted'
            rid = str(os.getpid())+'-'+str(time.time_ns())
            value['new_calls'] += 1
            value['reserved_tokens'] += bound
            value['active_reservations'][rid] = dict(bound=bound, pid=os.getpid(), phase=self.phase,
                metadata=dict(experiment=self.config['experiment_id'], **context))
            return rid
        rid = self.update(change)
        self.trace('request_started', reservation_id=rid, bound=bound, **context)
        return rid

    def settle(self, rid, actual, forwards, failed=False):
        def change(value):
            entry = value['active_reservations'].pop(rid)
            assert 0 <= actual <= entry['bound']
            value['reserved_tokens'] -= entry['bound']
            value['actual_tokens'] += actual
            value['native_forward_calls'] += forwards
            unknown = entry['bound']-actual if failed else 0
            value['unknown_cost_upper_bound_tokens'] += unknown
            return unknown
        unknown = self.update(change)
        self.trace('request_settled', reservation_id=rid, actual_positions=actual, forward_calls=forwards,
                   independent_unknown=unknown, failed=failed)


def seed_all(seed):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


