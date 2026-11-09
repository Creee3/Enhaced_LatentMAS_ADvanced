"""Resume the stopped one-run controller through a corrected manager entry."""
import json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'research/full_receiver576_20261109'),str(REPO/'research/TRG225_a100_20261106')]
import orchestrate
from deploy import PRIVATE,call,start as original_start

def start(phase):
    if phase!='formal':return original_start(phase)
    a=json.loads((PRIVATE/'cloud_active.json').read_text());run=a['run']
    script=r'''
from pathlib import Path
import subprocess,sys,json,time
r=Path(__RUN__);sys.path.insert(0,str(r))
from common import verify,identity,save
source=verify();entry=r/'management/manager_gate6.py';assert not (r/'management/formal').exists()
check=subprocess.run([sys.executable,str(entry),'--phase','formal'],cwd=r,capture_output=True,text=True);assert check.returncode==0,check.stderr
folder=r/'launches';folder.mkdir(exist_ok=True)
with (folder/'formal.log').open('ab') as log:p=subprocess.Popen([sys.executable,str(entry),'--phase','formal','--execute'],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
time.sleep(.3);assert p.poll() is None,(folder/'formal.log').read_text()[-2000:]
out=dict(run=str(r),phase='formal',pid=p.pid,identity=identity(p.pid),source=source,entry=str(entry),launched_ns=time.time_ns());save(folder/'formal.json',out);print(json.dumps(out))
'''.replace('__RUN__',repr(run))
    raw=call(script);(PRIVATE/'formal_launch.json').write_bytes(raw);a.update(status='FORMAL_RUNNING');(PRIVATE/'cloud_active.json').write_text(json.dumps(a,indent=2),encoding='utf-8');print(raw.decode(),flush=True)

if __name__=='__main__':
    orchestrate.start=start
    orchestrate.main(60)
