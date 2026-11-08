"""One compact native smoke snapshot; no answers or accuracy on stdout."""
import datetime,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'TRG225_a100_20261106'))
from capture_current import call
PRIVATE=Path('D:/liuxin_workspace/ICML2027/advanced_MOC/experiments/batch_validation_20261108')

def main():
 r=json.loads((PRIVATE/'smoke_launch.json').read_text())['prepared_run']
 source=r'''
from pathlib import Path
import json,sys,time
r=Path(RUN);sys.path.insert(0,str(r.parent/'TRG225_a100_s42_l10_20261106T155639Z'));from cloud_common import identity,load
workers=load(r/'workers.json') if (r/'workers.json').exists() else [];out=[]
def rows(p):
 result=[]
 if p.exists():
  for b in p.read_bytes().splitlines():
   if not b:continue
   try:result.append(json.loads(b))
   except json.JSONDecodeError:pass # final append can be in flight
 return result
for w in workers:
 d=r/f"case{w['case']}";trace=rows(d/'trace.jsonl');results=rows(d/'results.jsonl');started=[x for x in trace if x['event']=='request_started'];settled=[x for x in trace if x['event']=='request_settled'];pending=set(x['reservation_id'] for x in started)-set(x['reservation_id'] for x in settled)
 progress=next((x.get('completed_forwards') for x in reversed(trace) if x['event']=='native_forward_progress' and x.get('reservation_id') in pending),None)
 out.append(dict(case=w['case'],identity=identity(w['pid']),ready=(d/'model_ready.json').exists(),committed_methods=[x['method'] for x in results],started=len(started),settled=len(settled),pending_requests=len(pending),current_completed_forwards=progress,trace_events=len(trace),last_event=trace[-1]['event'] if trace else None,log_bytes=(r/f"case{w['case']}.log").stat().st_size,done=(d/'done.json').exists()))
fails={p.name:load(p) for p in r.glob('failure_case*.json')};b=load(r.parents[1]/'cumulative_budget.json');summary=dict(timestamp_ns=time.time_ns(),manager=identity(load(r/'launch_receipt.json')['pid']),workers=out,committed=sum(len(x['committed_methods']) for x in out),target=12,failures=fails,completed=(r/'completed.json').exists(),mps_stop=load(r/'mps_stop.json') if (r/'mps_stop.json').exists() else None,budget={k:b[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']})
if summary['manager'] is None and not summary['completed']:summary['manager_error_tail']=(r/'manager.log').read_text()[-2000:]
print(json.dumps(summary))
'''.replace('RUN',repr(r))
 raw=call(source);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');(PRIVATE/f'smoke_inspect_{stamp}.json').write_bytes(raw);(PRIVATE/'smoke_latest.json').write_bytes(raw);x=json.loads(raw)
 if x['mps_stop']:x['mps_stop']={k:x['mps_stop'][k] for k in ['returncode','owned_after']}
 for w in x['workers']:
  if w['identity']:w['identity']={k:w['identity'][k] for k in ['pid','uid','start_ticks','CUDA_VISIBLE_DEVICES']}
 print(json.dumps(x))
if __name__=='__main__':main()
