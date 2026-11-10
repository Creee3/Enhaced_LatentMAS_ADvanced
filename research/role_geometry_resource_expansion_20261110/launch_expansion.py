"""Explicit management expansion action, never imported by tests."""
import base64,datetime,hashlib,json,sys
from pathlib import Path
CODE=Path(__file__).resolve().parent
sys.path.insert(0,str(CODE.parent/'role_geometry_run_20261110'))
from remote import call,active,PRIVATE

def main():
 run=active()['run'];before=json.loads((PRIVATE/'resource_expansion_precheck.json').read_text(encoding='utf-8'))
 policy=dict(run=run,max_workers=8,per_gpu={'0':3,'1':4,'3':1},additions=[0,3],old_manager=before['manager'],run_lock_sha256=before['source']['run_lock_sha256'],gpus=[dict(index=i,uuid=line.split(',')[1].strip()) for line in before['gpus'].splitlines() for i in [int(line.split(',')[0])] if i in (0,1,3)],user_authorization='2026-11-10 用户：你在gpu0和gpu3上在开一个worker吧；在原六基础上各加一个，总八',unchanged_scientific_scope='120 questions,12 methods,1440 keys; same frozen worker, model,queue and ledger',gpu_first_admission='util<=30%, free>=70%; existing-own-card free>=21GiB',created=datetime.datetime.now(datetime.timezone.utc).isoformat())
 files={'expansion_manager.py':(CODE/'expansion_manager.py').read_bytes(),'expansion_resource_policy.json':(json.dumps(policy,ensure_ascii=False,indent=2)+'\n').encode('utf-8')}
 lock=dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()},scientific_run_lock_unchanged=policy['run_lock_sha256'],management_only=True)
 files['expansion_management_lock.json']=(json.dumps(lock,indent=2)+'\n').encode()
 for n,b in files.items():(PRIVATE/n).write_bytes(b)
 script="""from pathlib import Path
import base64,json,sys,subprocess,time,os
r=Path(RUN);sys.path.insert(0,str(r));from common import load,save,verify,identity
f=r/'management';old=OLD
assert identity(old['pid'])==old and not (f/'failure.json').exists() and not (r/'engineering_gate_ready.json').exists()
assert not (f/'expansion_management_lock.json').exists() and not (f/'expansion_spawn_attempted.json').exists()
assert verify()['run_lock_sha256']==SHA
for name,content in FILES.items():
 b=base64.b64decode(content);(f/name).write_bytes(b)
compile((f/'expansion_manager.py').read_text(encoding='utf-8'),str(f/'expansion_manager.py'),'exec')
with (f/'expansion_manager.log').open('ab') as log:
 p=subprocess.Popen([sys.executable,str(f/'expansion_manager.py'),'--run-root',str(r)],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
time.sleep(.2);owner=identity(p.pid);assert owner is not None
save(f/'expansion_manager_spawned.json',owner);print(json.dumps(dict(spawned=owner,science=verify(),old_residents=len(load(f/'workers.json')))))
""".replace('RUN',repr(run)).replace('OLD',repr(before['manager'])).replace('SHA',repr(policy['run_lock_sha256'])).replace('FILES',repr({n:base64.b64encode(b).decode() for n,b in files.items()}))
 raw=call(script,60);(PRIVATE/'expansion_launch_receipt.json').write_bytes(raw);print(raw.decode('utf-8'))
if __name__=='__main__':main()
