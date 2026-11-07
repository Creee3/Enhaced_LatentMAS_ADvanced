"""Spread owned, model-ready workers across the container's existing CPU set."""
import datetime,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'TRG225_a100_20261106'))
from capture_current import call,PRIVATE
def main():
 a=json.loads((PRIVATE/'cloud_active.json').read_text())
 script='''
import json,sys,os,time
from pathlib import Path
r=Path(RUN_PATH);sys.path.insert(0,str(r))
from cloud_common import load,save,verify,identity
from manager import owned
workers=load(r/'workers.json');cpus=sorted(os.sched_getaffinity(0));assert len(cpus)==8
assert {w['gpu'] for w in workers}<={0,1} and len(workers)<=8
report=[]
for w in sorted(workers,key=lambda x:(-x['gpu'],x['slot'])):
 assert owned(w) and w['initial_identity']['uid']==os.getuid()
 if not load(Path(w['out'])/'model_ready.json'):continue
 core=cpus[(0 if w['gpu']==1 else 4)+w['slot']-1];p=Path('/proc',str(w['pid']))
 tids=[int(t.name) for t in (p/'task').iterdir()]
 before={str(t):sorted(os.sched_getaffinity(t)) for t in tids}
 assert owned(w)
 for t in tids:
  assert (p/'task'/str(t)).exists()
  os.sched_setaffinity(t,{core})
 after={str(t):sorted(os.sched_getaffinity(t)) for t in tids}
 assert all(v==[core] for v in after.values()) and owned(w)
 report.append(dict(pid=w['pid'],gpu=w['gpu'],slot=w['slot'],cpu=core,identity=identity(w['pid']),before=before,after=after))
record=dict(passed=True,unix=time.time(),allowed_cpus=cpus,workers=report,source=verify(),model_restarts=0,science_or_thread_count_changed=False)
save(r/'management/cpu_distribution_verified.json',record);print(json.dumps(record))
'''.replace('RUN_PATH',repr(a['run']))
 raw=call(script);x=json.loads(raw)
 (PRIVATE/'management/cpu_distribution_verified.json').write_bytes(raw)
 print(json.dumps(dict(passed=True,assignments=[dict(pid=w['pid'],gpu=w['gpu'],cpu=w['cpu']) for w in x['workers']])))
if __name__=='__main__':main()
