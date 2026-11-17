"""Operational per-question scheduling; imports unchanged frozen science."""
import argparse
import fcntl
import os
from pathlib import Path
import time
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, required=True)
    args = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    import run as science
    import torch
    root = science.HERE
    # Shared lock excludes the old exclusive single-worker implementation.
    guard = (root / 'worker.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_SH | fcntl.LOCK_NB)
    claims = root / 'claims_v3'
    claims.mkdir(exist_ok=True)
    states = root / 'workers_v3'
    states.mkdir(exist_ok=True)
    state = dict(pid=os.getpid(), gpu=args.gpu, started=time.time(),
                 process_start_ticks=Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()[19],
                 kind='offline_matrix_worker_no_model', scheduler='v3', completed_by_worker=0)
    def update(**kw):
        state.update(kw, updated=time.time())
        science.save(states / (str(os.getpid()) + '.json'), state)
    try:
        science.verify()
        assert science.read(root / 'checks.json')['passed']
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.set_device(0)
        torch.cuda.init()
        torch.cuda.set_per_process_memory_fraction(.05, 0)
        device = torch.device('cuda:0')
        torch.cuda.reset_peak_memory_stats(device)
        update(status='running')
        for entry in science.read(root / 'manifest.json'):
            if (root / 'STOP').exists():
                break
            target = root / 'results' / (entry['id'] + '.json')
            with (claims / (entry['id'] + '.lock')).open('a') as claim:
                try:
                    fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if target.exists():
                    continue
                result = science.question(entry, device, update)
                result['peak_cuda_bytes'] = torch.cuda.max_memory_allocated(device)
                assert not target.exists()
                science.save(target, result)
                update(completed_by_worker=state['completed_by_worker'] + 1,
                       phase='committed', last_question=entry['id'])
                print(entry['id'], result['seconds'], flush=True)
        with (root / 'summary_v3.lock').open('a') as summary_guard:
            fcntl.flock(summary_guard, fcntl.LOCK_EX)
            if len(list((root / 'results').glob('*.json'))) == 40 and not (root / 'COMPLETE.json').exists():
                science.summary()
        update(status='exited')
    except BaseException:
        update(status='failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
