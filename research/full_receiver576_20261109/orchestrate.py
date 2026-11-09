"""Explicit one-run execution through engineering gate, rollout and archive.

This is the requested experiment's controller, not a recurring automation.
"""
import argparse,json,sys,time,traceback,os
from pathlib import Path
from deploy import start,PRIVATE
from inspect_current import capture
from collect import collect
from capture_current import call

def state(status,**kw):
 a=json.loads((PRIVATE/'cloud_active.json').read_text());a.update(status=status,updated_unix=time.time(),**kw);(PRIVATE/'cloud_active.json').write_text(json.dumps(a,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(controller_status=status,**kw)),flush=True)
def gated_archive(phase):
 try:return collect(phase)
 except BaseException as e:
  (PRIVATE/(phase+'_gate_failure.json')).write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2),encoding='utf-8')
  collect(phase,failure=True);state('VALIDATION_OR_ARCHIVE_FAILED',phase=phase);raise
def wait_phase(phase,poll):
 seen=set()
 while True:
  x=capture();p=next((p for p in x['phases'] if p['phase']==phase),None)
  for e in x['errors']:
   s=json.dumps(e,sort_keys=True)
   if s not in seen:seen.add(s);print(json.dumps(dict(new_error=e)),flush=True)
  if p and p['manager'] is None and all(w['identity'] is None for w in p['workers']) and p['stop_proofs']:
   if p['finished'] and p['finished']['all_eligible_complete'] and not seen:return x
   collect(phase,failure=True);state('STOPPED_WITH_ARCHIVED_FAILURE',phase=phase);raise RuntimeError('Phase did not meet structural completion; no retry or next phase')
  active=json.loads((PRIVATE/'cloud_active.json').read_text());remote=active['run']
  stop=call('from pathlib import Path\nprint((Path('+repr(remote)+')/"STOP").exists())').decode().strip()=='True'
  if stop and p and p['manager'] is None and all(w['identity'] is None for w in p['workers']):
   collect(phase,failure=True);state('USER_STOPPED',phase=phase);raise RuntimeError('User STOP; no automatic resume')
  time.sleep(poll)
def main(poll):
 guard=(PRIVATE/'controller.lock').open('a+b');guard.seek(0)
 if guard.read(1)==b'':guard.write(b'0');guard.flush()
 guard.seek(0)
 if os.name=='nt':
  import msvcrt
  msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
 else:
  import fcntl
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if not (PRIVATE/'engineering_archive_verified.json').exists():
  if not (PRIVATE/'engineering_launch.json').exists():start('engineering')
  state('ENGINEERING_RUNNING');wait_phase('engineering',poll);gated_archive('engineering');state('ENGINEERING_ARCHIVED_VERIFIED')
 if not (PRIVATE/'formal_archive_verified.json').exists():
  if not (PRIVATE/'formal_launch.json').exists():start('formal')
  state('FORMAL_RUNNING');wait_phase('formal',poll);gated_archive('formal');state('FORMAL_RAW_ARCHIVE_VERIFIED')
 remote=json.loads((PRIVATE/'cloud_active.json').read_text())['run']
 script='import subprocess,sys\nfrom pathlib import Path\nr=Path('+repr(remote)+')\np=subprocess.run([sys.executable,str(r/"score.py")],cwd=r,capture_output=True,text=True)\nassert p.returncode==0,p.stderr\nprint(p.stdout)'
 if not (PRIVATE/'final_summary.json').exists():print(call(script,600).decode(),flush=True)
 if not (PRIVATE/'formal_final_archive_verified.json').exists():collect('formal',final=True)
 state('COMPLETED_ARCHIVED_SCORED',complete=True,target_keys=1152)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--poll-seconds',type=int,default=60);a=p.parse_args()
 try:main(a.poll_seconds)
 except BaseException as e:
  (PRIVATE/'controller_failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()),indent=2),encoding='utf-8');print(json.dumps(dict(controller_error=repr(e))),flush=True);raise
