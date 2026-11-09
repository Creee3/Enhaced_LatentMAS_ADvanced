"""One stopped-case recovery; delegates unchanged scientific worker.

Requires verified full stopped archive and fresh empty A100. No retry loop.
"""
import argparse, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path

def matches(row, ref):
    return (row['index'] == ref['index'] == 15
            and row['shared_planner_kv_sha'] == ref['shared_planner_kv_sha']
            and row['fork_rng_sha'] == ref['fork_rng_sha'])

def main(run, child=False):
    sys.path.insert(0, str(run))
    import common as c
    folder = run/'management/recovery015'
    if child:
        import worker
        ref = c.load(folder/'reference.json')
        original_kv, original_rng, original_commit = worker.kv_sha, worker.rng_sha, worker.commit
        checked = [False, False]
        def kv(torch, xs):
            result = original_kv(torch, xs)
            assert result == ref['shared_planner_kv_sha'], 'Recovery Planner differs; refuse partial pairing'
            checked[0] = True
            return result
        def rng(torch):
            result = original_rng(torch)
            if not checked[1]:
                assert result == ref['fork_rng_sha'], 'Recovery fork RNG differs'
                checked[1] = True
                c.save(folder/'prefix_verified.json', dict(passed=True, planner=checked[0], rng=True,
                       shared_RIDs_recomputed=True, cross_attempt_RIDs_identical=False, unix=time.time()))
            return result
        def commit(task, row, out):
            assert all(checked) and matches(row, ref) and row['method'] in ['RRR', 'TTT']
            original_commit(task, row, out)
        worker.kv_sha, worker.rng_sha, worker.commit = kv, rng, commit
        worker.main('formal', 0, run/'attempts/recovery015_once')
        return
    import fcntl
    guard = (run/'manager.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    c.verify()
    archive = c.load(run/'formal_failure_archive_verified.json')
    assert archive['passed'] and archive['failure_archive']
    assert archive['archive_sha256'] == 'a2758800c4766f27b8c64037ea5fec957ccec010b0238268794cbf82b8c57610'
    assert not (run/'STOP').exists() and not folder.exists(), 'No automatic repeat'
    for phase in ['engineering', 'formal']:
        f = run/'management'/phase
        for owner in [c.load(f/'identity.json')['identity']] + [w['identity'] for w in c.load(f/'workers.json')]:
            actual = c.identity(owner['pid'])
            assert actual is None or actual['start_ticks'] != owner['start_ticks']
        assert all(c.load(p)['returncode'] == 0 and not c.load(p)['owned_after'] for p in f.glob('mps_*_stop.json'))
    budget = c.load(c.ROOT/'cumulative_budget.json')
    assert not [v for v in budget['active_reservations'].values() if v.get('metadata', {}).get('experiment') == run.name]
    required = next(p['upper_positions'] for p in c.load(run/'preflight.json')['proofs'] if p['index'] == 15)
    assert sum(budget[k] for k in ['actual_tokens', 'reserved_tokens', 'unknown_cost_upper_bound_tokens']) + required <= 115000000
    refs = [r for r in c.records() if r['index'] == 15]
    assert len(refs) == 1 and refs[0]['method'] == 'GGG'
    sample = subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=uuid,utilization.gpu,memory.free,memory.total', '--format=csv,noheader,nounits'], text=True).strip().split(', ')
    uuid, util, free, total = sample[0], *map(int, sample[1:])
    assert uuid == 'GPU-250104ae-64db-3b01-4660-a580491ebc0d' and util <= 30 and free/total >= .7
    from manager import foreign_mps
    assert not foreign_mps(dict(index=1, uuid=uuid), [])
    folder.mkdir()
    c.save(folder/'reference.json', {k:refs[0][k] for k in ['index', 'method', 'shared_planner_kv_sha', 'fork_rng_sha', 'shared_request_ids']})
    c.save(folder/'identity.json', dict(identity=c.identity(os.getpid()), unix=time.time()))
    c.save(folder/'resource_improvement.json', dict(uuid=uuid, util=util, free_MiB=free, total_MiB=total,
           live_experiment_workers_before=0, planned_workers=1, previous_failed_case_elapsed_seconds=1800,
           previous_failed_forwards=5088, full_stopped_archive=archive, unix=time.time()))
    pipe = Path('/tmp')/f'rwrecover{os.getpid()}'
    pipe.mkdir(mode=0o700)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, CUDA_MPS_PIPE_DIRECTORY=str(pipe),
        CUDA_MPS_LOG_DIRECTORY=str(folder/'mpslog'), LATENTMAS_SHARED_LEDGER=str(c.ROOT/'cumulative_budget.json'),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    (folder/'mpslog').mkdir(mode=0o700)
    process = None
    try:
        subprocess.run(['/usr/bin/nvidia-cuda-mps-control', '-d'], env=env, check=True, capture_output=True, timeout=30)
        with c.locked():
            queue = c.load(run/'queue.json'); task = queue['tasks'][15]
            assert task['state'] == 'held' and task['committed_methods'] == ['GGG'] and task['methods'] == ['RRR', 'TTT']
            assert sum(t['state']=='complete' for t in queue['tasks']) == 575
            assert not any(t['state'] in ['pending', 'running'] for t in queue['tasks'])
            c.save(folder/'queue_before.json', queue)
            task.update(state='pending', recovery_attempt=1, recovery_evidence=str(folder/'resource_improvement.json'))
            c.save(run/'queue.json', queue)
        with (folder/'worker.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--run', str(run), '--child'],
                      env=env, cwd=run, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        time.sleep(.1)
        c.save(folder/'worker.json', dict(pid=process.pid, identity=c.identity(process.pid), out=str(run/'attempts/recovery015_once')))
        rc = process.wait()
        c.save(folder/'finished.json', dict(returncode=rc, task=c.load(run/'queue.json')['tasks'][15], unix=time.time()))
    except BaseException as e:
        c.save(folder/'failure.json', dict(error=repr(e), traceback=traceback.format_exc(), unix=time.time()))
        raise
    finally:
        if process is not None: process.wait()
        stop = subprocess.run(['/usr/bin/nvidia-cuda-mps-control'], input='quit\n', text=True, env=env, capture_output=True, timeout=30)
        time.sleep(2)
        from manager import own_pipe
        c.save(folder/'mps_stop.json', dict(returncode=stop.returncode, owned_after=own_pipe(pipe), unix=time.time()))

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--child',action='store_true');a=p.parse_args();main(a.run,a.child)
