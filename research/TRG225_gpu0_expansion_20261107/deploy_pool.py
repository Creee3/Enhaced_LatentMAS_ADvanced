"""Explicit GPU0 expansion: preserve/adopt live GPU1 models, management only."""
import base64,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'TRG225_a100_20261106'))
from capture_current import PRIVATE,call,active

def main():
 a=active();run=a['run'];raw=(HERE/'pool_manager.py').read_bytes()
 script=r'''
import base64,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
r=Path(RUN_PATH);sys.path.insert(0,str(r))
from cloud_common import identity,load,save,verify,sha,qlock,gpu_rows
from manager import owned,stable
m=r/'management';m.mkdir(exist_ok=True)
source=verify();assert source['run_lock_sha256']==LOCK_SHA
workers=load(r/'workers.json');assert len(workers)==4 and all(w['gpu']==1 and owned(w) for w in workers)
assert not (r/'STOP').exists() and not load(r/'manager_failure.json')
old=load(r/'manager.json');ident=old['identity'];actual_old=identity(ident['pid'])
assert ident['pid']==28169 and ident['uid']==os.getuid()
assert actual_old==ident or (actual_old is None and (m/'manager28169_before_gpu0.json').exists())
assert ident['argv']==[sys.executable,str(r/'pair_manager.py')]
assert any(t['state']=='pending' for t in load(r/'queue.json')['tasks'])
fresh=gpu_rows([0])[0];assert fresh['utilization']<=30 and fresh['free_mib']/fresh['total_mib']>=.70
module=base64.b64decode(MODULE_B64);target=m/'pool_manager.py'
assert not target.exists() or target.read_bytes()==module,'Never replace a different manager'
target.write_bytes(module)
import importlib.util
spec=importlib.util.spec_from_file_location('pool_candidate',target);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
row=lambda util,free,total=81920:dict(utilization=util,free_mib=free,total_mib=total)
assert mod.eligible(row(30,57344),False)
assert not mod.eligible(row(31,81153),False)
assert not mod.eligible(row(0,57343),False)
assert mod.eligible(row(100,21504),True)
assert not mod.eligible(row(0,21503),True)
assert not mod.eligible(None,False)
policy=dict(max_workers=8,per_gpu_max={'0':4,'1':4},allowed_gpus=[0,1],user_authorized_eight=True,first_util_max=30,first_free_ratio=.70,next_free_mib=21504,load_serial_per_card=True,attempt_once_per_slot=True)
save(m/'gpu01_resource_policy.json',policy)
save(m/'gpu01_management_lock.json',dict(source_lock_sha256=source['run_lock_sha256'],files={'pool_manager.py':sha(target),'gpu01_resource_policy.json':sha(m/'gpu01_resource_policy.json')},cpu_tests=6,science_unchanged=True))
if not (m/'manager28169_before_gpu0.json').exists():
 save(m/'manager28169_before_gpu0.json',dict(manager=old,status=load(r/'manager_status.json'),workers=workers,resources=fresh,source=source,unix=time.time()))
 with qlock():save(m/'queue_before_gpu0_takeover.json',load(r/'queue.json'))
# Identity checked immediately before signaling only the management process.
if actual_old:
 assert identity(28169)==ident
 os.kill(28169,signal.SIGTERM)
for _ in range(100):
 if not identity(28169):break
 time.sleep(.05)
assert not identity(28169),'Old manager still live'
assert all(owned(w) for w in workers),'Original models must remain resident'
guard=(r/'manager.lock').open('a')
for _ in range(300):
 try:fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);break
 except BlockingIOError:time.sleep(.1)
else:raise RuntimeError('Manager lock not released; do not bypass it')
fcntl.flock(guard,fcntl.LOCK_UN);guard.close()
argv=[sys.executable,str(target)];log=m/'pool_manager.log'
env=dict(os.environ,PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
with log.open('ab') as f:p=subprocess.Popen(argv,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
 new=identity(p.pid)
 if new and new['argv']==argv:break
 time.sleep(.05)
assert new and new['argv']==argv
for _ in range(100):
 if load(r/'manager.json')['identity']['pid']==p.pid:break
 time.sleep(.1)
assert load(r/'manager.json')['identity']['pid']==p.pid and identity(p.pid)==new
assert all(owned(w) for w in workers)
locator=Path('/home/ubuntu/cloud_migration_20261106/TRG225/cloud_active.json');a=load(locator)
a.update(manager_pid=p.pid,max_workers=8,gpus=[0,1],gpu=[0,1],manager_entry='management/pool_manager.py',per_gpu_max={'0':4,'1':4})
save(locator,a)
record=dict(passed=True,unix=time.time(),old_manager_identity=ident,new_manager_identity=new,adopted_worker_identities=[identity(w['pid']) for w in workers],resource_policy=policy,source=verify(),models_restarted=0,manual_settlements=0,accuracy_computed=False)
save(m/'gpu0_takeover_verified.json',record)
print(json.dumps(record))
'''.replace('RUN_PATH',repr(run)).replace('LOCK_SHA',repr(a['source_lock_sha256'])).replace('MODULE_B64',repr(base64.b64encode(raw).decode()))
 out=json.loads(call(script));dest=PRIVATE/'management';dest.mkdir(exist_ok=True)
 (dest/'gpu0_takeover_verified.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 active()
 print(json.dumps(dict(takeover_verified=out['passed'],manager_pid=out['new_manager_identity']['pid'],adopted_pids=[w['pid'] for w in out['adopted_worker_identities']],max_workers=8,source_lock_sha256=out['source']['run_lock_sha256'])))
if __name__=='__main__':main()
