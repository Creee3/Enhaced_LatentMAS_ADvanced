"""Explicit SSH transport; never invoked on module import."""
import subprocess

ROOT = '/home/liuxin/ICML2027/experiments/latentmas_cumulative_history_20261025'
PYTHON = '/home/liuxin/.conda/envs/kv-graph/bin/python'

def call(source, timeout=90):
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
        'DevContainer', PYTHON+' -'], input=source.encode('utf-8'),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', errors='replace')[-6000:])
    return result.stdout

if __name__ == '__main__':
    import sys
    from pathlib import Path
    sys.stdout.buffer.write(call(Path(sys.argv[1]).read_text(encoding='utf-8'),180))
