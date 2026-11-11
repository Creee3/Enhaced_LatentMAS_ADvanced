"""One authorized execution controller, resident engineering gate and final archive."""
import base64,datetime,json,sys,time,traceback
from remote import HERE,PRIVATE,active,call
from inspect_current import capture
from collect import collect
def state(name,**kw):
 a=active();a.update(status=name,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw);(PRIVATE/'cloud_active.json').write_text(json.dumps(a,indent=2)+'\n',encoding='utf-8')
def remote_run(name,args=()):
 run=active()['run'];return call('import subprocess,sys\nfrom pathlib import Path\nr=Path('+repr(run)+')\np=subprocess.run([sys.executable,str(r/'+repr(name)+')]+'+repr(list(args))+',cwd=r,capture_output=True,text=True)\nassert p.returncode==0,p.stderr\nprint(p.stdout)',600)
def main():
 assert json.loads((PRIVATE/'source_sealed.json').read_text())['passed'];run=active()['run'];state('LAUNCHING_ENGINEERING')
 script=r'''
import os,sys,time,subprocess,json
from pathlib import Path
r=Path(RUN);sys.path.insert(0,str(r));from common import verify,save,identity
verify();folder=r/'management';assert not (folder/'manager_spawned.json').exists() and not (folder/'identity.json').exists()
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OPENBLAS_NUM_THREADS='1')
with (folder/'manager.log').open('ab') as f:p=subprocess.Popen([sys.executable,str(r/'manager.py'),'--execute'],cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
time.sleep(.1);owner=identity(p.pid);assert owner is not None;save(folder/'manager_spawned.json',owner);print(json.dumps(owner))
'''.replace('RUN',repr(run))
 (PRIVATE/'manager_spawned.json').write_bytes(call(script,30));state('ENGINEERING_RUNNING_RESIDENT')
 archived_failure=False
 while True:
  x=capture();failures=x['management_failure'] or [w['failure'] for w in x['workers'] if w['failure']]
  if failures and not archived_failure:
   state('FAILURE_PRESERVE_HEALTHY_WORKERS');collect('failure_scene');archived_failure=True
  if x['engineering_gate_ready'] and not x['engineering_released'] and not failures:
   raw=remote_run('audit.py',['engineering']);(PRIVATE/'engineering_audit_verified.json').write_bytes(raw);collect('engineering')
   call('import sys\nfrom pathlib import Path\nr=Path('+repr(run)+');sys.path.insert(0,str(r));from common import load,save,verify\na=load(r/"engineering_audit_verified.json");z=load(r/"engineering_archive_verified.json");assert a["passed"] and a["cases"]==6 and a["unique_rids"]==126 and z["passed"] and a["run_lock_sha256"]==verify()["run_lock_sha256"]\nsave(r/"engineering_release.json",dict(passed=True,engineering_reused=True,archive_sha256=z["archive_sha256"]))')
   state('FORMAL_RUNNING_RESIDENT')
  if x['committed']==120 and x['manager'] is None and not any(w['identity'] for w in x['workers']):
   assert not x['own_reservations'];raw=remote_run('audit.py',['formal']);(PRIVATE/'formal_audit_verified.json').write_bytes(raw);collect('formal');remote_run('summarize.py');collect('formal_final');state('COMPLETE_ARCHIVED_SUMMARIZED');break
  if x['manager'] is None and x['finished'] and x['committed']<120:
   if not archived_failure:collect('failure_scene')
   raise RuntimeError('Manager exited before exact coverage; no blind recovery')
  time.sleep(30)
if __name__=='__main__':
 try:main()
 except BaseException as e:state('CONTROLLER_ERROR_PRESERVE_SCENE',controller_error=repr(e));traceback.print_exc();raise
