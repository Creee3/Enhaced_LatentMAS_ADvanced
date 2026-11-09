"""One authorized serial coordinator, engineering gates then both raw archives before scoring."""
import json,os,sys,time,traceback
from pathlib import Path
from deploy import PRIVATE,call
from inspect_current import capture
from collect import collect

def state(status,**kw):
 p=PRIVATE/'scope_active.json';x=json.loads(p.read_text(encoding='utf-8'));x.update(status=status,updated_unix=time.time(),**kw);p.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=status,**kw)),flush=True)

def start(kind,phase):
 run=json.loads((PRIVATE/kind/'cloud_active.json').read_text(encoding='utf-8'))['run']
 script=r'''
from pathlib import Path
import subprocess,sys,json,time,os
r=Path(__RUN__);sys.path.insert(0,str(r));from common import verify,identity,save
source=verify();phase=__PHASE__;assert not (r/'management'/phase).exists()
check=subprocess.run([sys.executable,str(r/'manager.py'),'--phase',phase],cwd=r,capture_output=True,text=True);assert check.returncode==0,check.stderr
folder=r/'launches';folder.mkdir(exist_ok=True)
with (folder/(phase+'.log')).open('ab') as log:p=subprocess.Popen([sys.executable,str(r/'manager.py'),'--phase',phase,'--execute'],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
time.sleep(.3);assert p.poll() is None,(folder/(phase+'.log')).read_text()[-1000:]
out=dict(run=str(r),phase=phase,pid=p.pid,identity=identity(p.pid),source=source,launched_ns=time.time_ns());save(folder/(phase+'.json'),out);print(json.dumps(out))
'''.replace('__RUN__',repr(run)).replace('__PHASE__',repr(phase))
 raw=call(script);(PRIVATE/kind/(phase+'_launch.json')).write_bytes(raw)
 a=json.loads((PRIVATE/kind/'cloud_active.json').read_text());a['status']=phase.upper()+'_RUNNING';(PRIVATE/kind/'cloud_active.json').write_text(json.dumps(a,indent=2),encoding='utf-8')
 print(json.dumps(dict(kind=kind,phase=phase,launched=json.loads(raw)['pid'])),flush=True)

def wait_phase(kind,phase):
 run=json.loads((PRIVATE/kind/'cloud_active.json').read_text())['run'];seen=set()
 while True:
  x=capture(kind);p=next((p for p in x['phases'] if p['phase']==phase),None)
  for e in x['errors']:
   text=json.dumps(e,sort_keys=True)
   if text not in seen:seen.add(text);print(json.dumps(dict(new_error=e)),flush=True)
  if p:
   proof=call('import subprocess,sys\nfrom pathlib import Path\nr=Path('+repr(run)+')\np=subprocess.run([sys.executable,str(r/"verify_live.py"),'+repr(phase)+'],cwd=r,capture_output=True,text=True)\nassert p.returncode==0,p.stderr\nprint(p.stdout)')
   (PRIVATE/kind/(phase+'_live_verify_summary.json')).write_bytes(proof)
  if p and p['manager'] is None and all(w['identity'] is None for w in p['workers']) and p['stop_proofs']:
   if p['finished'] and p['finished']['all_eligible_complete'] and not seen:return
   collect(kind,phase,failure=True);raise RuntimeError('Stopped with archived failure; no automatic retry')
  time.sleep(60)

def main():
 guard=(PRIVATE/'controller.lock').open('a+b');guard.seek(0)
 if guard.read(1)==b'':guard.write(b'0');guard.flush()
 guard.seek(0)
 if os.name=='nt':
  import msvcrt;msvcrt.locking(guard.fileno(),msvcrt.LK_NBLCK,1)
 else:
  import fcntl;fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for kind in ['HL','extension']:
  for phase in ['engineering','formal']:
   if (PRIVATE/kind/(phase+'_archive_verified.json')).exists():continue
   if not (PRIVATE/kind/(phase+'_launch.json')).exists():start(kind,phase)
   state(kind.upper()+'_'+phase.upper()+'_RUNNING',active_kind=kind,active_phase=phase)
   wait_phase(kind,phase);collect(kind,phase);state(kind.upper()+'_'+phase.upper()+'_ARCHIVED')
 scope=json.loads((PRIVATE/'scope_active.json').read_text());run=scope['runs']['HL']
 raw=call('import subprocess,sys\nfrom pathlib import Path\nr=Path('+repr(run)+')\np=subprocess.run([sys.executable,str(r/"score.py")],cwd=r,capture_output=True,text=True)\nassert p.returncode==0,p.stderr\nprint(p.stdout)',600);print(raw.decode(),flush=True)
 for kind in ['HL','extension']:collect(kind,'formal',final=True)
 (PRIVATE/'final_summary.json').write_bytes((PRIVATE/'HL/final_summary.json').read_bytes());(PRIVATE/'final_report.md').write_bytes((PRIVATE/'HL/final_report.md').read_bytes())
 state('COMPLETED_ARCHIVED_SCORED',new_keys=2055,combined_keys=4935)

if __name__=='__main__':
 try:main()
 except BaseException as e:
  (PRIVATE/'controller_failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc(),unix=time.time()),indent=2),encoding='utf-8');state('CONTROLLER_ERROR_PRESERVE_SCENE');raise
