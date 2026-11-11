"""Single bounded read-only cloud snapshot; private complete UTF8, compact stdout."""
import datetime,json
from remote import PRIVATE,call,active

def capture():
 run=active()['run']
 script=r"""
from pathlib import Path
import json,sys,time,subprocess
r=Path(RUN);sys.path.insert(0,str(r));from common import load,identity,verify,ROOT
source=verify();q=load(r/'queue.json');budget=load(ROOT/'cumulative_budget.json');folder=r/'management';workers=[]
for w in load(folder/'workers.json') if (folder/'workers.json').exists() else []:
 out=Path(w['out']);x=identity(w['pid']);live=x is not None and x['start_ticks']==w['identity']['start_ticks'];current=load(out/'current.json') if (out/'current.json').exists() else None;failure=load(out/'failure.json') if (out/'failure.json').exists() else (dict(error='Abrupt worker exit',**load(out/'manager_observed_exit.json')) if (out/'manager_observed_exit.json').exists() else None);last=None
 if (out/'trace.jsonl').exists():
  with (out/'trace.jsonl').open('rb') as f:f.seek(max(0,(out/'trace.jsonl').stat().st_size-65536));tail=f.read()
  for line in reversed(tail.split(b'\n')):
   try:z=json.loads(line)
   except (json.JSONDecodeError,UnicodeDecodeError):continue
   if z.get('event')=='native_forward_progress':last={k:z[k] for k in ['unix','completed_forwards','completed_positions']};break
 workers.append(dict(pid=w['pid'],identity=x if live else None,gpu=w['gpu_index'],ready=(out/'model_ready.json').exists(),current=current,last_forward=last,failure=failure,last_commit=load(out/'last_commit.json') if (out/'last_commit.json').exists() else None))
manager=None
if (folder/'identity.json').exists():
 owner=load(folder/'identity.json')['identity'];x=identity(owner['pid']);manager=x if x is not None and x['start_ticks']==owner['start_ticks'] else None
result=dict(unix=time.time(),run=str(r),source=source,target=120,committed=sum(len(t['committed_methods']) for t in q['tasks']),queue={state:sum(t['state']==state for t in q['tasks']) for state in ['pending','running','held','complete']},groups={group:dict(keys=sum(len(t['committed_methods']) for t in q['tasks'] if t['dataset_group']==group),complete=sum(t['state']=='complete' for t in q['tasks'] if t['dataset_group']==group)) for group in load(r/'data_manifest.json')['groups']},manager=manager,workers=workers,engineering_gate_ready=(r/'engineering_gate_ready.json').exists(),engineering_released=(r/'engineering_release.json').exists(),management_failure=load(folder/'failure.json') if (folder/'failure.json').exists() else None,finished=load(folder/'finished.json') if (folder/'finished.json').exists() else None,budget={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']},own_reservations=[dict(rid=rid,pid=x['pid'],bound=x['bound']) for rid,x in budget['active_reservations'].items() if x.get('metadata',{}).get('experiment')==r.name],gpus=subprocess.run(['nvidia-smi','-i','0,1','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout)
print(json.dumps(result))
""".replace('RUN',repr(run))
 raw=call(script);x=json.loads(raw);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');(PRIVATE/('inspect_'+stamp+'.json')).write_bytes(raw);(PRIVATE/'latest_inspect.json').write_bytes(raw)
 small={k:x[k] for k in ['unix','target','committed','queue','groups','engineering_gate_ready','engineering_released','budget','gpus']};small.update(manager_pid=x['manager']['pid'] if x['manager'] else None,own_reservations=len(x['own_reservations']),workers=[dict(pid=w['pid'],uid=w['identity']['uid'] if w['identity'] else None,start_ticks=w['identity']['start_ticks'] if w['identity'] else None,gpu=w['gpu'],ready=w['ready'],current=w['current'],failure=w['failure']['error'] if w['failure'] else None) for w in x['workers']],manager_failure=x['management_failure']['error'] if x['management_failure'] else None)
 print(json.dumps(small));return x
if __name__=='__main__':capture()
