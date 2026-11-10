"""Bounded isolated execution using the existing NumPy-enabled environment."""
import json,os,signal,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def run(code,test='',timeout=10,expression=None):
 payload=json.dumps(dict(code=code,test=test,timeout=timeout,expression=expression));read_fd,write_fd=os.pipe()
 with tempfile.TemporaryFile() as output:
  env={'PATH':'/usr/bin','HOME':'/nonexistent','PYTHONDONTWRITEBYTECODE':'1','MOC_RESULT_FD':str(write_fd),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
  p=subprocess.Popen([sys.executable,'-I',str(ROOT/'executor_child.py')],stdin=subprocess.PIPE,stdout=output,stderr=output,text=True,env=env,cwd='/tmp',pass_fds=(write_fd,),start_new_session=True);os.close(write_fd)
  try:
   p.communicate(payload,timeout=timeout+3);data=os.read(read_fd,65536);output.seek(0);log=output.read(6000).decode(errors='replace')
   if p.returncode or not data:return dict(passed=False,error_type='ResourceLimit' if p.returncode in (-9,-24,-25) else 'SandboxFailure',error=log[-2000:],returncode=p.returncode)
   return json.loads(data)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGKILL);p.communicate();return dict(passed=False,error_type='TimeoutError',error='isolated limit exceeded')
  finally:os.close(read_fd)
