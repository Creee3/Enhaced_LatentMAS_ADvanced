"""Only manager entry: fresh GPU0-7 admission, at most six owned workers."""
import fcntl,json,os,subprocess,time
from cloud_common import *

def owned(worker):
    actual=identity(worker['pid']);initial=worker['initial_identity']
    return actual if actual and all(actual[k]==initial[k] for k in ['uid','start_ticks','argv','CUDA_VISIBLE_DEVICES']) else None

def stable(pid,gpu,argv):
    for _ in range(100):
        actual=identity(pid)
        if actual and actual['argv']==argv and actual['CUDA_VISIBLE_DEVICES']==str(gpu):return actual
        time.sleep(.02)
    raise RuntimeError('exec identity not stable')

def main():
    guard=(RUN/'manager.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source=verify();policy=load(RUN/'runtime_policy.json');plan=load(RUN/'plan.json')
    for n,e in load(RUN/'run_lock.json')['snapshots'].items():assert sha(RUN/n)==e['sha256'], n
    workers=load(RUN/'workers.json',[]);attempted={w['gpu'] for w in workers}
    save(RUN/'manager.json',dict(identity=identity(os.getpid()),source=source,entry='manager.py',started=time.time()))
    try:
        while True:
            queue=load(RUN/'queue.json');live=[w for w in workers if owned(w)]
            pending=sum(t['state']=='pending' for t in queue['tasks'])
            if (RUN/'STOP').exists():
                if not live:break
            elif not pending and not live:break
            slots=min(max(0,policy['max_workers']-len(live)),pending)
            if not (RUN/'STOP').exists():
                for candidate in sorted(gpu_rows(),key=lambda g:(-g['free_mib'],g['gpu'])):
                    if slots<=0 or choose(load(RUN/'queue.json')['tasks']) is None:break
                    gpu=candidate['gpu']
                    if gpu not in policy['allowed_gpus'] or gpu in attempted:continue
                    fresh=next(g for g in gpu_rows() if g['gpu']==gpu)
                    if fresh['utilization']>30 or fresh['free_mib']/fresh['total_mib']<.70:continue
                    # Failure-specific evidence, not a new global resource threshold.
                    block=policy.get('historical_loading_failure',{}).get(str(gpu))
                    if block:
                        if block['mode']=='manual_clearance':
                            clearance=load(RUN/('gpu'+str(gpu)+'_historical_clearance.json'))
                            if not clearance or not clearance.get('verified_repair_or_resource_improvement'):continue
                        elif block['mode']=='previously_demonstrated_resource_improvement':
                            if fresh['free_mib']<block['demonstrated_recovery_free_mib']:continue
                            save(RUN/('gpu'+str(gpu)+'_historical_clearance.json'),dict(
                                verified_repair_or_resource_improvement=True,failed_free_mib=block['failed_free_mib'],fresh=fresh,
                                failure_archive_verified=block['failure_archive_verified'],unknown_resettled=False))
                    attempted.add(gpu)
                    out=RUN/'attempts'/('gpu'+str(gpu)+'_'+str(time.time_ns()));log=out.with_suffix('.log')
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),LATENTMAS_SHARED_LEDGER=str(ROOT/'cumulative_budget.json'),
                        PYTHONUNBUFFERED='1',PYTHONHASHSEED='42',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
                    argv=[policy['python'],str(RUN/'cloud_worker.py'),'--out',str(out)]
                    with log.open('ab') as stream:
                        process=subprocess.Popen(argv,cwd=RUN,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    ident=stable(process.pid,gpu,argv)
                    workers.append(dict(pid=process.pid,gpu=gpu,out=str(out),log=str(log),admission=fresh,initial_identity=ident))
                    save(RUN/'workers.json',workers);slots-=1
            with qlock():
                queue=load(RUN/'queue.json');changed=False
                for task in queue['tasks']:
                    if task['state']=='running':
                        worker=next(w for w in workers if w['pid']==task['owner_pid'])
                        if not owned(worker):
                            task.update(state='failed_held',failure='owner exited; preserve scene, no automatic retry',ended=time.time());changed=True
                if changed:save(RUN/'queue.json',queue)
            save(RUN/'manager_status.json',dict(unix=time.time(),identity=identity(os.getpid()),
                attempted_gpus=sorted(attempted),live_pids=[w['pid'] for w in workers if owned(w)],
                tasks={s:sum(t['state']==s for t in queue['tasks']) for s in ['pending','running','complete','failed_held']},
                target=plan['total_target'],reused=0,new_target=plan['new_target']))
            time.sleep(10)
        queue=load(RUN/'queue.json')
        save(RUN/'manager_exit.json',dict(unix=time.time(),completed=all(t['state']=='complete' for t in queue['tasks']),user_stop=(RUN/'STOP').exists()))
    except BaseException as exc:
        save(RUN/'manager_failure.json',dict(error=repr(exc),unix=time.time()));raise

if __name__=='__main__':main()
