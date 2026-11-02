"""Management-only dynamic scheduling; frozen parent science is unchanged."""
import fcntl
import os
import subprocess
import time
from cloud_common import *


def stable_identity(pid, gpu, argv):
    for _ in range(100):
        found = identity(pid)
        if found and found['argv'] == argv and found['CUDA_VISIBLE_DEVICES'] == str(gpu):
            return found
        time.sleep(.02)
    raise RuntimeError('worker exec identity did not stabilize')


def main():
    guard = (RUN/'manager.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = verify()
    policy = load(RUN/'runtime_policy.json')
    reuse = load(RUN/'reuse_index.json')
    parent = Path(reuse['parent_run'])
    assert sha(parent/'run_lock.json') == reuse['parent_source_lock_sha256']
    assert load(parent/'manager_exit.json')['completed']
    for name in ['cloud_worker.py','stage_ops.py','cloud_common.py','current_jive.py','author_parser.py']:
        assert sha(RUN/name) == sha(parent/name)
    assert sha(Path(reuse['diamond_full_run'])/'run_lock.json') == reuse['diamond_full_source_lock_sha256']
    assert sha(RUN/'reused_results.jsonl') == reuse['reused_snapshot_sha256']
    reused = [__import__('json').loads(x) for x in (RUN/'reused_results.jsonl').read_bytes().splitlines() if x]
    assert len(reused) == len({key(x) for x in reused}) == reuse['reused'] == 1420
    assert {key(x) for x in reused} == {tuple(k) for k in reuse['reused_keys']}
    workers=[]
    attempted=set()
    save(RUN/'manager.json', dict(identity=identity(os.getpid()),entry=str(RUN/'manager_main448.py'),started=time.time(),source=source,reused_answers=1420))
    try:
        while True:
            q=load(RUN/'queue.json')
            live=[w for w in workers if identity(w['pid'])]
            if (RUN/'STOP').exists() or (terminal(q['tasks']) and not live):
                break
            slots=max(0,min(policy['max_workers']-len(live),sum(t['state']=='pending' for t in q['tasks'])))
            resources=sorted(gpu_rows(),key=lambda g:(-g['free_mib'],g['gpu']))
            for candidate in resources:
                if slots<=0 or choose(load(RUN/'queue.json')['tasks']) is None:
                    break
                gpu=candidate['gpu']
                if gpu not in policy['allowed_gpus'] or gpu in attempted:
                    continue
                # Refresh immediately before each individual admission.
                fresh=next(g for g in gpu_rows() if g['gpu']==gpu)
                if fresh['utilization']>30 or fresh['free_mib']/fresh['total_mib']<.70:
                    continue
                attempted.add(gpu)
                out=RUN/'attempts'/('gpu'+str(gpu)+'_'+str(time.time_ns()))
                log=out.with_suffix('.log')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),LATENTMAS_SHARED_LEDGER=str(ROOT/'cumulative_budget.json'),
                         PYTHONUNBUFFERED='1',PYTHONHASHSEED='42',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                argv=[policy['python'],str(RUN/'cloud_worker.py'),'--out',str(out)]
                with log.open('ab') as stream:
                    proc=subprocess.Popen(argv,cwd=RUN,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                ident=stable_identity(proc.pid,gpu,argv)
                workers.append(dict(pid=proc.pid,gpu=gpu,out=str(out),log=str(log),admission=fresh,admitted=time.time(),initial_identity=ident))
                save(RUN/'workers.json',workers)
                slots-=1
            with qlock():
                q=load(RUN/'queue.json')
                changed=False
                for task in q['tasks']:
                    if task['state']=='running' and identity(task['owner_pid']) is None:
                        task.update(state='failed_held',failure='owner exited; retain scene and fees; no automatic retry',ended=time.time())
                        changed=True
                if changed:
                    save(RUN/'queue.json',q)
            save(RUN/'manager_status.json',dict(unix=time.time(),identity=identity(os.getpid()),attempted_gpus=sorted(attempted),
                 live_pids=[w['pid'] for w in workers if identity(w['pid'])],tasks={s:sum(t['state']==s for t in q['tasks']) for s in ['pending','running','complete','failed_held']},
                 reused=1420,total_cohort_target=3584,primary_metrics=['KV byte ratio','frozen-parser accuracy'],generation_time_not_selection_criterion=True))
            time.sleep(10)
        q=load(RUN/'queue.json')
        save(RUN/'manager_exit.json',dict(unix=time.time(),completed=all(t['state']=='complete' for t in q['tasks']),user_stop=(RUN/'STOP').exists(),reused=1420))
    except BaseException as exc:
        save(RUN/'manager_failure.json',dict(error=repr(exc),unix=time.time()))
        raise


if __name__=='__main__':
    main()
