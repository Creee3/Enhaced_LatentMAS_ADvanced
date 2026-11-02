"""One bounded read-only snapshot, full UTF8 metadata saved privately."""
import datetime,json
from pathlib import Path
from remote import ROOT,call

HERE=Path(__file__).resolve().parent
PRIVATE=HERE.parents[1].parent/'advanced_MOC/experiments/latentmas_role_blocks_smoke_20261102'

def main():
    active=json.loads((PRIVATE/'active.json').read_text(encoding='utf-8'))
    script='''import json,os,sys,hashlib
from pathlib import Path
run=Path(RUN_VALUE);out=Path(OUT_VALUE);sys.path.insert(0,str(run))
from cloud_common import identity,load
a=load(run/'active.json');b=load(run.parents[1]/'cumulative_budget.json')
rows=[json.loads(s) for s in (out/'results.jsonl').read_text(encoding='utf-8').split('\\n') if s] if (out/'results.jsonl').exists() else []
ident=identity(a['identity']['pid'])
value=dict(run=run.name,identity=ident,current=load(out/'current.json'),model_ready=load(out/'model_ready.json'),failure=load(out/'failure.json'),exit=load(out/'exit.json'),
 committed=len(rows),unique=len({(r['question_id'],r['repeat'],r['method']) for r in rows}),
 metadata=[{k:r[k] for k in ['question_id','repeat','method','generated_tokens','token_sha','selected_position_sha','retained_kv_bytes','positions_shape','timing']} for r in rows],
 budget={k:b[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']},own_reservations={k:v for k,v in b['active_reservations'].items() if v['pid']==a['identity']['pid']})
trace=out/'trace.jsonl'
if trace.exists():
 lines=trace.read_bytes().split(b'\\n');events=[json.loads(x) for x in lines[-12:] if x]
 value['last_events']=[{k:v for k,v in x.items() if k in ['unix','event','file','actual_positions','completed_forwards','reservation_id']} for x in events]
print(json.dumps(value,ensure_ascii=False))
'''.replace('RUN_VALUE',repr(active['run'])).replace('OUT_VALUE',repr(active['attempt']))
    raw=call(script,90);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (PRIVATE/('inspect_'+stamp+'.json')).write_bytes(raw);(PRIVATE/'inspect_latest.json').write_bytes(raw)
    v=json.loads(raw)
    print(json.dumps(dict(run=v['run'],identity=v['identity'],model_ready=bool(v['model_ready']),committed=v['committed'],unique=v['unique'],remaining=18-v['unique'],current=v['current'],failure=v['failure'],exit=v['exit'],budget=v['budget'],own_reservation_count=len(v['own_reservations']),last_events=v.get('last_events')),ensure_ascii=False))

if __name__=='__main__':main()
