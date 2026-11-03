"""One capture, complete private UTF8 inspect, bounded public summary."""
import datetime,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'role_smoke_20261102'))
from remote import call
from deploy import PRIVATE

def main():
    active=json.loads((PRIVATE/'cloud_active.json').read_text(encoding='utf-8'));run=active['run']
    source='import runpy,sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(run)+')\nif Path('+repr(run+'/manager_recovery.py')+').exists():\n from manager_recovery import install\n install()\nrunpy.run_path('+repr(run+'/inspect_run.py')+',run_name="__main__")'
    raw=call(source,180);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (PRIVATE/('inspect_'+stamp+'.json')).write_bytes(raw);(PRIVATE/'inspect_latest.json').write_bytes(raw)
    v=json.loads(raw)
    details=[]
    for w in v['workers']:
        ident=w['identity']
        details.append(dict(gpu=w['gpu'],pid=w['pid'],alive=bool(ident),
            uid=ident['uid'] if ident else None,start_ticks=ident['start_ticks'] if ident else None,
            CVD=ident['CUDA_VISIBLE_DEVICES'] if ident else None,model_ready=bool(w['model_ready']),
            new_committed=w['committed'],formal_labels=w['formal_labels'],
            current={k:w['current'][k] for k in ['question_id','method','phase'] if w.get('current') and k in w['current']},
            failure=dict(error=w['failure'].get('error','')[:240],current=w['failure'].get('current'),committed=w['failure'].get('committed')) if w['failure'] else None,
            exit={k:w['exit'][k] for k in ['completed','committed','released_idle_tail'] if k in w['exit']} if w['exit'] else None))
    summary=dict(run=v['run'],unique=v['combined_unique'],target=v['target'],remaining=v['remaining'],new_unique=v['new_unique'],reused=v['reused'],duplicates=v['duplicates'],unexpected=v['unexpected'],
        snapshots_valid=all(v['snapshots'].values()),diamond_unique=v['diamond_unique'],diamond_target=v['diamond_target'],queue=v['queue'],
        manager_pid=v['manager_identity']['pid'] if v['manager_identity'] else None,
        manager_failure=v['manager_failure'],manager_exit=v['manager_exit'],workers=details,budget=v['budget'],own_reserved_count=len(v['own_reservations']))
    active.update(latest_capture=stamp,latest_summary=summary)
    (PRIVATE/'cloud_active.json').write_text(json.dumps(active,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
