"""One read-only snapshot. No unfinished accuracy, no raw answers on stdout."""
import json,time
from cloud_common import *

def main():
    lock=load(RUN/'run_lock.json')
    source=verify()
    snapshot_sha={n:sha(RUN/n)==e['sha256'] for n,e in lock['snapshots'].items()}
    cohort=load(RUN/'cohort_questions.json');byid={q['question_id']:q for q in cohort}
    expected={(q['dataset'],q['question_id'],10,m) for q in cohort for m in ['T','R','R-shuffle']}
    reused=[json.loads(s) for s in (RUN/'reused_results.jsonl').read_bytes().split(b'\n') if s]
    rows=all_rows();keys=[key(r) for r in reused+rows];unique=set(keys)
    workers=load(RUN/'workers.json',[]);details=[]
    for w in workers:
        out=Path(w['out']);actual=identity(w['pid']);initial=w['initial_identity']
        real=actual if actual and all(actual[k]==initial[k] for k in ['uid','start_ticks','argv','CUDA_VISIBLE_DEVICES']) else None
        commits=[r for r in rows if r['question_id'] in byid and r.get('judger_receipt',{}).get('reservation_id','').startswith(str(w['pid'])+'-')]
        ready=load(out/'model_ready.json')
        formal={m:any(r['method']==m and r['source_k_equal_measured'] and r['source_v_equal_measured'] and r['protected_latent_sink'] and r['positions_shape']==[36,8,130] for r in commits) for m in ['T','R','R-shuffle']}
        details.append(dict(**w,identity=real,model_ready=ready,committed=len(commits),formal_labels=formal,
            current=load(out/'current.json'),last_commit=load(out/'last_commit.json'),failure=load(out/'failure.json'),exit=load(out/'exit.json')))
    budget=load(ROOT/'cumulative_budget.json');own={rid:x for rid,x in budget['active_reservations'].items() if x.get('metadata',{}).get('experiment')==RUN.name}
    queue=load(RUN/'queue.json');manager=load(RUN/'manager.json');mi=identity(manager['identity']['pid']) if manager else None
    diamond={k for k in expected if byid[k[1]]['is_diamond']}
    value=dict(unix=time.time(),run=RUN.name,source=source,snapshots=snapshot_sha,reused=len(reused),new_unique=len({key(r) for r in rows}),
        combined_unique=len(unique),target=1344,remaining=len(expected-unique),duplicates=len(keys)-len(unique),unexpected=len(unique-expected),
        diamond_unique=len(unique&diamond),diamond_target=594,baseline_reused=1792,
        queue={s:sum(t['state']==s for t in queue['tasks']) for s in ['pending','running','complete','failed_held']},
        manager_identity=mi,manager_failure=load(RUN/'manager_failure.json'),manager_exit=load(RUN/'manager_exit.json'),workers=details,
        budget={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']},own_reservations=own)
    print(json.dumps(value,ensure_ascii=False))

if __name__=='__main__':main()
