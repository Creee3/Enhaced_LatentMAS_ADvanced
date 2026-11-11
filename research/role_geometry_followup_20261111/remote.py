"""Existing SSH alias, two bounded retries, no experiment mutations on import."""
import json,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
PRIVATE=HERE.parents[2]/'advanced_MOC/experiments/latentmas_role_intervention120_20261111'
PYTHON='/home/ubuntu/.conda/envs/kv-graph/bin/python'
def call(script,timeout=180):
 errors=[]
 for attempt in range(2):
  p=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1','DevContainer',PYTHON+' -'],input=script.encode('utf-8'),capture_output=True,timeout=timeout)
  if p.returncode==0:return p.stdout
  error=p.stderr.decode('utf-8',errors='replace')[-4000:];errors.append(error)
  # Retry connectivity errors only; never repeat a possibly executed mutation.
  if not any(x in error for x in ['Connection timed out','Connection refused','No route to host','Could not resolve hostname','Connection closed by UNKNOWN']):break
 raise RuntimeError('SSH/remote failed: '+str(errors))
def active():return json.loads((PRIVATE/'cloud_active.json').read_text(encoding='utf-8'))
