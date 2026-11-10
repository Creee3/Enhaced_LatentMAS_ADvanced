"""Management-only adoption of six residents; add one on GPU0 and GPU3."""
import argparse, hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path

IDENTITY_FIELDS = ('pid','uid','start_ticks','argv','CUDA_VISIBLE_DEVICES')
def same_identity(actual, expected):
    return actual is not None and all(actual.get(k) == expected.get(k) for k in IDENTITY_FIELDS)
def admit_addition(*, live_total, live_gpu, max_gpu, all_ready, blocked, util, free, total):
    if blocked or not all_ready or live_total >= 8 or live_gpu >= max_gpu:
        return False
    return free >= 21*1024 if live_gpu else util <= 30 and total > 0 and free/total >= .70

def main(root):
    r=Path(root).resolve();sys.path.insert(0,str(r))
    from common import load,save,append,verify,identity,locked,ROOT
    from manager import own_pipe,foreign_mps
    from resource_policy import cpu_set
    import fcntl
    folder=r/'management';policy=load(folder/'expansion_resource_policy.json')
    module_lock=load(folder/'expansion_management_lock.json')
    for name,expected in module_lock['files'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected,name
    assert policy['max_workers']==8 and policy['per_gpu']=={'0':3,'1':4,'3':1}
    assert policy['additions']==[0,3] and policy['run']==str(r)
    source=verify();assert source['run_lock_sha256']==policy['run_lock_sha256']
    guard=(folder/'expansion_once.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (folder/'expansion_spawn_attempted.json').exists()
    save(folder/'expansion_spawn_attempted.json',dict(identity=identity(os.getpid()),unix=time.time()))
    workers=load(folder/'workers.json');assert len(workers)==6
    for w in workers:
        assert same_identity(identity(w['pid']),w['identity']), 'resident identity changed'
        assert (Path(w['out'])/'model_ready.json').exists() and not (Path(w['out'])/'failure.json').exists()
    old=load(folder/'identity.json');assert same_identity(identity(old['identity']['pid']),policy['old_manager'])
    assert not (folder/'failure.json').exists() and not (r/'STOP').exists() and not (r/'MANAGEMENT_HOLD').exists()
    assert not (r/'engineering_gate_ready.json').exists(), 'do not take over during archive gate'
    services={}
    for card in (0,1):
        proof=load(folder/f'mps_{card}_start.json');uuid=proof['gpu_uuid'];pipe=Path(proof['pipe'])
        found=[w for w in workers if w['gpu_index']==card];assert found
        env_items=(Path('/proc')/str(found[0]['pid'])/'environ').read_bytes().split(b'\0')
        original=dict(z.split(b'=',1) for z in env_items if b'=' in z)
        assert original[b'CUDA_MPS_PIPE_DIRECTORY'].decode()==str(pipe)
        env=dict(os.environ)
        for key in ('CUDA_VISIBLE_DEVICES','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY','LATENTMAS_SHARED_LEDGER','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','TOKENIZERS_PARALLELISM','OMP_NUM_THREADS','MKL_NUM_THREADS','ROLE_GEOMETRY_AUTHOR_DIR','ROLE_GEOMETRY_DATA_ROOT'):
            if key.encode() in original:env[key]=original[key.encode()].decode()
        services[uuid]=dict(env=env,pipe=pipe,card=card)
        assert own_pipe(pipe), 'original private MPS disappeared'
    save(folder/'expansion_predecessor.json',dict(manager=old,workers=workers,source=source,unix=time.time(),action='planned management-only replacement; resident models and MPS retained'))
    # Default SIGTERM exits the old Python manager only; all scientific workers
    # were started in their own sessions. No worker or MPS signal is sent.
    assert same_identity(identity(old['identity']['pid']),policy['old_manager'])
    os.kill(old['identity']['pid'],signal.SIGTERM)
    deadline=time.monotonic()+15
    while same_identity(identity(old['identity']['pid']),policy['old_manager']):
        assert time.monotonic()<deadline,'old manager did not exit'
        time.sleep(.1)
    manager_guard=(r/'manager.lock').open('a');fcntl.flock(manager_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    scope_guard=(ROOT/'role_geometry120_scope.lock').open('a');fcntl.flock(scope_guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for w in workers:assert same_identity(identity(w['pid']),w['identity']), 'resident lost during adoption'
    save(folder/'identity.json',dict(identity=identity(os.getpid()),source=source,unix=time.time(),management_extension=module_lock,adopted_from=old['identity']))
    save(folder/'expansion_adoption_verified.json',dict(passed=True,old_manager=old['identity'],manager=identity(os.getpid()),resident_identities_preserved=[w['identity'] for w in workers],private_MPS_preserved=True,science_lock=source,unix=time.time()))
    attempted=set();blocked=set();last_affinity=0
    def live(w):return same_identity(identity(w['pid']),w['identity'])
    def living():return [w for w in workers if live(w)]
    def register():save(folder/'workers.json',workers)
    try:
        while True:
            active=living();q=load(r/'queue.json')
            for w in workers:
                if live(w) or w.get('reaped'):continue
                out=Path(w['out']);ended=load(out/'exit.json') if (out/'exit.json').exists() else None
                unfinished=any(t['state']=='running' and t.get('owner',{}).get('pid')==w['pid'] for t in q['tasks'])
                w.update(reaped=True,normal_exit=bool(ended and ended.get('normal')));register()
                if (unfinished or not w['normal_exit']) and not (r/'STOP').exists():
                    blocked.add(w['gpu_uuid']);save(out/'manager_observed_exit.json',dict(error='adopted/added worker exited abnormally',unix=time.time()))
                    with locked():
                        fresh=load(r/'queue.json')
                        for t in fresh['tasks']:
                            if t['state']=='running' and t.get('owner',{}).get('pid')==w['pid']:t.update(state='held',error='worker exited; preserve scene')
                        save(r/'queue.json',fresh)
            if time.monotonic()-last_affinity>=60:
                last_affinity=time.monotonic();allowed=cpu_set(Path('/sys/fs/cgroup/cpuset.cpus.effective').read_text())
                affinity=[]
                for w in living():
                    if not (Path(w['out'])/'model_ready.json').exists() or not allowed:continue
                    target={allowed[w['slot']%len(allowed)]}
                    for thread in Path('/proc',str(w['pid']),'task').iterdir():
                        try:os.sched_setaffinity(int(thread.name),target)
                        except (PermissionError,ProcessLookupError):pass
                    affinity.append(dict(pid=w['pid'],cpus=sorted(target)))
                save(folder/'expansion_cpu_affinity.json',dict(allowed=allowed,workers=affinity,unix=time.time()))
            q=load(r/'queue.json');eng=[q['tasks'][i] for i in q['engineering_indices']]
            if not (r/'engineering_release.json').exists() and any(t['state']=='held' for t in eng):(r/'MANAGEMENT_HOLD').touch()
            active=living();all_ready=all((Path(w['out'])/'model_ready.json').exists() for w in active)
            waiting=bool(active) and all((Path(w['out'])/'current.json').exists() and load(Path(w['out'])/'current.json')['phase']=='waiting_engineering' for w in active)
            if all(t['state']=='complete' for t in eng) and all_ready and waiting and not (r/'engineering_gate_ready.json').exists():
                save(r/'engineering_gate_ready.json',dict(keys=72,waiting_for_local_archive=True,models_retained=True,unix=time.time()))
            if not active and (all(t['state'] in ('complete','held') for t in q['tasks']) or (r/'STOP').exists() or (r/'MANAGEMENT_HOLD').exists()):break
            if (r/'STOP').exists():
                budget=load(ROOT/'cumulative_budget.json')
                for w in active:
                    if (Path(w['out'])/'model_ready.json').exists():continue
                    assert not any(v['pid']==w['pid'] for v in budget['active_reservations'].values())
                    if live(w):os.kill(w['pid'],signal.SIGTERM)
            gate_busy=(r/'engineering_gate_ready.json').exists() and not (r/'engineering_release.json').exists()
            if not gate_busy and not (r/'STOP').exists() and not (r/'MANAGEMENT_HOLD').exists() and all_ready and any(t['state']=='pending' for t in q['tasks']):
                for card in policy['additions']:
                    if card in attempted:continue
                    gpu=next(g for g in policy['gpus'] if g['index']==card);uuid=gpu['uuid'];oncard=[w for w in active if w['gpu_uuid']==uuid]
                    fresh=subprocess.run(['nvidia-smi','-i',uuid,'--query-gpu=utilization.gpu,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15).stdout.strip()
                    util,free,total=map(int,fresh.split(','));conflicts=foreign_mps(gpu,{str(s['pipe']) for s in services.values()})
                    decision=admit_addition(live_total=len(active),live_gpu=len(oncard),max_gpu=policy['per_gpu'][str(card)],all_ready=all_ready,blocked=uuid in blocked or bool(conflicts),util=util,free=free,total=total)
                    append(folder/'expansion_admission.jsonl',dict(gpu=gpu,util=util,free=free,total=total,admitted=decision,foreign_MPS_conflicts=conflicts,live_total=len(active),live_gpu=len(oncard),unix=time.time()))
                    if not decision:continue
                    if uuid not in services:
                        pipe=Path('/tmp')/f'geomexpand{os.getpid()}_{card}';pipe.mkdir(mode=0o700);log=folder/f'mps_{card}';log.mkdir(mode=0o700,exist_ok=True)
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,CUDA_MPS_PIPE_DIRECTORY=str(pipe),CUDA_MPS_LOG_DIRECTORY=str(log),LATENTMAS_SHARED_LEDGER=str(ROOT/'cumulative_budget.json'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',ROLE_GEOMETRY_AUTHOR_DIR=str(r/'author'),ROLE_GEOMETRY_DATA_ROOT=str(r))
                        start=subprocess.run(['/usr/bin/nvidia-cuda-mps-control','-d'],env=env,capture_output=True,text=True,timeout=30);assert start.returncode==0,start.stderr
                        services[uuid]=dict(env=env,pipe=pipe,card=card);save(folder/f'mps_{card}_start.json',dict(gpu_uuid=uuid,pipe=str(pipe),owned=own_pipe(pipe),unix=time.time()))
                    slot=6+policy['additions'].index(card);out=r/'attempts'/f'expansion_w{slot}_{time.time_ns()}';attempted.add(card)
                    append(folder/'expansion_attempts.jsonl',dict(gpu=card,slot=slot,out=str(out),unix=time.time()))
                    with (folder/f'expansion_worker{slot}.log').open('ab') as log:
                        proc=subprocess.Popen([sys.executable,str(r/'worker.py'),'--slot',str(slot),'--out',str(out)],cwd=r,env=services[uuid]['env'],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                    time.sleep(.1);owner=identity(proc.pid);assert owner is not None
                    workers.append(dict(pid=proc.pid,identity=owner,gpu_uuid=uuid,gpu_index=card,slot=slot,out=str(out),management_extension=True));register()
                    break
            time.sleep(2)
        q=load(r/'queue.json');save(folder/'finished.json',dict(all_complete=all(t['state']=='complete' for t in q['tasks']),counts={s:sum(t['state']==s for t in q['tasks']) for s in ('pending','running','held','complete')},keys=sum(len(t['committed_methods']) for t in q['tasks']),unix=time.time(),expanded_management=True))
    except BaseException as exc:
        save(folder/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()));(r/'MANAGEMENT_HOLD').touch();raise
    finally:
        while living():time.sleep(2)
        for uuid,service in services.items():
            before=own_pipe(service['pipe']);stop=subprocess.run(['/usr/bin/nvidia-cuda-mps-control'],input='quit\n',env=service['env'],capture_output=True,text=True,timeout=30);time.sleep(1);after=own_pipe(service['pipe'])
            save(folder/f"mps_{service['card']}_stop.json",dict(returncode=stop.returncode,owned_before=before,owned_after=after,unix=time.time()))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',required=True);a=parser.parse_args();main(a.run_root)
