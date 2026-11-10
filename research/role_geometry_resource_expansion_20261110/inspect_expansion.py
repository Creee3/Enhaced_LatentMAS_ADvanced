"""Read-only resource proof for the explicitly authorized eight-worker pool."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'role_geometry_run_20261110'))
from remote import call,active,PRIVATE

def capture(wait_seconds=0):
 assert 0<=wait_seconds<=45
 script=r"""from pathlib import Path
import sys,json,time,subprocess
r=Path(RUN);sys.path.insert(0,str(r));from common import load,identity,verify,ROOT
f=r/'management';deadline=time.monotonic()+WAIT
while True:
 ws=load(f/'workers.json')
 if (len(ws)==8 and all((Path(w['out'])/'model_ready.json').exists() for w in ws)) or (f/'failure.json').exists() or time.monotonic()>=deadline:break
 time.sleep(2)
q=load(r/'queue.json');old=load(f/'expansion_predecessor.json')['workers'];adopted=all(any(w['pid']==v['pid'] and w['identity']==v['identity'] and identity(v['pid'])==v['identity'] for w in ws) for v in old)
rows=[]
for w in ws:
 out=Path(w['out']);owner=identity(w['pid']);ready=load(out/'model_ready.json') if (out/'model_ready.json').exists() else None
 trace=[]
 if (out/'trace.jsonl').exists():
  raw=(out/'trace.jsonl').read_bytes()
  trace=[json.loads(x) for x in raw.split(b'\n')[:-1] if x]
 row=dict(pid=w['pid'],gpu=w['gpu_index'],slot=w['slot'],identity=owner,identity_matches=owner==w['identity'],ready=bool(ready),phase=load(out/'current.json').get('phase') if (out/'current.json').exists() else 'loading',failure=load(out/'failure.json')['error'] if (out/'failure.json').exists() else None,verified_weight_files=len({x['file'] for x in trace if x.get('event')=='model_file_verified'}),submitted_methods=[json.loads(x)['method'] for x in (out/'results.jsonl').read_bytes().split(b'\n')[:-1] if x] if (out/'results.jsonl').exists() else [])
 if ready:row.update(bf16_cuda=ready['bf16_cuda'],parameter_bytes=ready['parameter_bytes'],identity_norm=ready['identity_norm'],target_norm=ready['target_norm'],ready_source=ready['source']['run_lock_sha256'])
 rows.append(row)
budget=load(ROOT/'cumulative_budget.json');result=dict(unix=time.time(),science=verify(),old_six_identities_preserved=adopted,manager=identity(load(f/'identity.json')['identity']['pid']),workers=rows,committed=sum(len(t['committed_methods']) for t in q['tasks']),queue={v:sum(t['state']==v for t in q['tasks']) for v in ['pending','running','held','complete']},engineering_released=(r/'engineering_release.json').exists(),failure=load(f/'failure.json') if (f/'failure.json').exists() else None,budget={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']},gpus=subprocess.run(['nvidia-smi','-i','0,1,3','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout)
print(json.dumps(result))
""".replace('RUN',repr(active()['run'])).replace('WAIT',str(wait_seconds))
 raw=call(script,100);(PRIVATE/'expansion_latest.json').write_bytes(raw);x=json.loads(raw)
 small={k:x[k] for k in ['unix','committed','queue','engineering_released','old_six_identities_preserved','budget','gpus','failure']}
 small['workers']=[{k:w[k] for k in ['pid','gpu','ready','phase','failure','verified_weight_files']} for w in x['workers']]
 print(json.dumps(small));return x
if __name__=='__main__':capture(40 if '--wait' in sys.argv else 0)
