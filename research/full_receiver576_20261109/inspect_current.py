"""One SSH snapshot, private UTF8 landing, stdout only counts/identity/errors."""
import datetime,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PRIVATE=Path('D:/liuxin_workspace/ICML2027/advanced_MOC/experiments/latentmas_full_receiver576_20261109')
sys.path.insert(0,str(HERE.parent/'TRG225_a100_20261106'));from capture_current import call
def capture():
 active=json.loads((PRIVATE/'cloud_active.json').read_text());run=active['run']
 script=r'''
from pathlib import Path
import json,sys,time
r=Path(RUN);sys.path[:0]=[str(r),str(r.parent/'TRG225_a100_s42_l10_20261106T155639Z')];from common import load,identity,verify
q=load(r/'queue.json');b=load(r.parents[1]/'cumulative_budget.json');phases=[];errors=[]
for phase in ['engineering','formal']:
 p=r/'management'/phase
 if not p.exists():continue
 owner=load(p/'identity.json');manager=identity(owner['identity']['pid']);workers=[]
 for w in load(p/'workers.json') if (p/'workers.json').exists() else []:
  out=Path(w['out']);actual=identity(w['pid']);live=actual is not None and actual['start_ticks']==w['identity']['start_ticks'];current=load(out/'current.json') if (out/'current.json').exists() else None;failure=load(out/'failure.json') if (out/'failure.json').exists() else None
  latest=None
  if (out/'trace.jsonl').exists():
   with (out/'trace.jsonl').open('rb') as f:
    f.seek(max(0,(out/'trace.jsonl').stat().st_size-65536));tail=f.read()
   for line in reversed(tail.split(b'\n')):
    try:x=json.loads(line)
    except (json.JSONDecodeError,UnicodeDecodeError):continue
    if x.get('event')=='native_forward_progress':latest={k:x.get(k) for k in ['unix','completed_forwards','completed_positions']};break
  if failure:errors.append(dict(phase=phase,pid=w['pid'],error=failure['error']))
  workers.append(dict(pid=w['pid'],identity=actual if live else None,gpu=w['gpu_index'],ready=(out/'model_ready.json').exists(),current=current,last_commit=load(out/'last_commit.json') if (out/'last_commit.json').exists() else None,last_forward=latest,failure=failure))
 if (p/'failure.json').exists():errors.append(dict(phase=phase,manager_error=load(p/'failure.json')['error']))
 phases.append(dict(phase=phase,manager=manager if manager and manager['start_ticks']==owner['identity']['start_ticks'] else None,workers=workers,finished=load(p/'finished.json') if (p/'finished.json').exists() else None,stop_proofs=[load(z) for z in p.glob('mps_*_stop.json')]))
summary=dict(unix=time.time(),source=verify(),target=1152,committed=sum(len(t['committed_methods']) for t in q['tasks']),queue={s:sum(t['state']==s for t in q['tasks']) for s in ['pending','running','held','complete']},dataset_counts={d:dict(complete=sum(t['dataset']==d and t['state']=='complete' for t in q['tasks']),keys=sum(len(t['committed_methods']) for t in q['tasks'] if t['dataset']==d)) for d in ['gpqa','mmlu_pro','gsm8k']},phases=phases,errors=errors,budget={k:b[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']},own_reservations=[dict(rid=rid,pid=v['pid'],bound=v['bound']) for rid,v in b['active_reservations'].items() if v.get('metadata',{}).get('experiment')==r.name])
print(json.dumps(summary))
'''.replace('RUN',repr(run))
 raw=call(script);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');(PRIVATE/('inspect_'+stamp+'.json')).write_bytes(raw);(PRIVATE/'latest_inspect.json').write_bytes(raw);x=json.loads(raw)
 output={k:x[k] for k in ['target','committed','queue','dataset_counts','errors','budget']};output['own_reservations']=len(x['own_reservations']);output['managers']=[dict(phase=p['phase'],pid=p['manager']['pid'] if p['manager'] else None,finished=p['finished']) for p in x['phases']];output['workers']=[dict(phase=p['phase'],pid=w['pid'],gpu=w['gpu'],live=w['identity'] is not None,ready=w['ready'],uid=w['identity']['uid'] if w['identity'] else None,start_ticks=w['identity']['start_ticks'] if w['identity'] else None,current_phase=w['current']['phase'] if w['current'] else None,method=w['current'].get('method') if w['current'] else None,last_forward=w['last_forward']) for p in x['phases'] for w in p['workers']]
 print(json.dumps(output));return x
if __name__=='__main__':capture()
