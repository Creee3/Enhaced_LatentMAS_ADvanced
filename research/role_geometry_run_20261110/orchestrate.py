"""One local controller: inspect, gate archive, release residents, final audit/archive/score."""
import base64,datetime,json,sys,time,traceback
from pathlib import Path
from remote import HERE,PRIVATE,active,call
from inspect_current import capture
from collect import collect

def state(name,**kw):
 a=active();a.update(status=name,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw);(PRIVATE/'cloud_active.json').write_text(json.dumps(a,indent=2)+'\n',encoding='utf-8')
def remote_run(filename,args=''):
 run=active()['run'];return call('from pathlib import Path\nimport subprocess,sys\nr=Path('+repr(run)+')\np=subprocess.run([sys.executable,str(r/'+repr(filename)+')]+'+repr(args.split())+',cwd=r,capture_output=True,text=True)\nassert p.returncode==0,p.stderr\nprint(p.stdout)',600)
def release():
 proof=json.loads((PRIVATE/'engineering_archive_verified.json').read_text());run=active()['run'];raw=json.loads((PRIVATE/'engineering_audit_verified.json').read_text())
 call('from pathlib import Path\nimport json,sys\nr=Path('+repr(run)+');sys.path.insert(0,str(r));from common import load,save,verify\na=load(r/"engineering_audit_verified.json");z=load(r/"engineering_archive_verified.json");assert a["passed"] and a["keys"]==72 and z["passed"] and a["run_lock_sha256"]==verify()["run_lock_sha256"]\nsave(r/"engineering_release.json",dict(passed=True,engineering_keys_reused=True,archive_sha256=z["archive_sha256"]))')
def launch_manager():
 run=active()['run'];script=r"""
from pathlib import Path
import sys,os,subprocess,json,time
r=Path(RUN);sys.path.insert(0,str(r));from common import load,save,verify,identity
verify();folder=r/'management';folder.mkdir(exist_ok=True);assert not (folder/'identity.json').exists() and not (folder/'manager_spawned.json').exists()
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',ROLE_GEOMETRY_AUTHOR_DIR=str(r/'author'),ROLE_GEOMETRY_DATA_ROOT=str(r))
with (folder/'manager.log').open('ab') as f:p=subprocess.Popen([sys.executable,str(r/'manager.py'),'--execute'],cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
time.sleep(.1);owner=identity(p.pid);assert owner is not None;save(folder/'manager_spawned.json',owner);print(json.dumps(owner))
""".replace('RUN',repr(run))
 raw=call(script,30);(PRIVATE/'manager_spawned.json').write_bytes(raw)
def main():
 state('LAUNCHING_ENGINEERING');launch_manager();state('ENGINEERING_RUNNING_RESIDENT')
 while True:
  x=capture()
  failures=x['management_failure'] or [w['failure'] for w in x['workers'] if w['failure']]
  if failures:
   state('FAILURE_PRESERVE_SCENE',failure_summary=True)
   # Preserve stopped affected-worker evidence while healthy workers continue.
   collect('failure_scene');raise RuntimeError('Worker/manager failure; scene archived; no automatic recovery')
  if x['engineering_gate_ready'] and not x['engineering_released']:
   b=remote_run('audit.py','engineering');(PRIVATE/'engineering_audit_verified.json').write_bytes(b);collect('engineering');release();state('FORMAL_RUNNING_RESIDENT')
  if x['committed']==1440 and x['manager'] is None and not any(w['identity'] for w in x['workers']):
   assert not x['own_reservations'];b=remote_run('audit.py','formal');(PRIVATE/'formal_audit_verified.json').write_bytes(b);collect('formal');remote_run('score.py');remote_run('summarize_diagnostics.py');collect('formal_final');state('COMPLETE_ARCHIVED_SCORED');break
  if x['manager'] is None and x['finished'] and x['committed']<1440:
   collect('failure_scene');raise RuntimeError('Manager ended before exact coverage; scene archived')
  time.sleep(30)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  state('CONTROLLER_ERROR_PRESERVE_SCENE',controller_error=repr(e));traceback.print_exc();raise
