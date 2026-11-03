"""Explicit prepare/preflight/launch for user-authorized G-only matched225 run."""
import argparse
import base64
import datetime
import hashlib
import json
import sys
from pathlib import Path
from build import build,sha,HERE,OLD,REPO
sys.path.insert(0,str(HERE.parent/'role_smoke_20261102'))
from remote import ROOT,PYTHON,call

PRIVATE=REPO.parent/'advanced_MOC/experiments/latentmas_attention_control225_20261103'
PARENT_PRIVATE=REPO.parent/'advanced_MOC/experiments/latentmas_role_blocks_main448_20261102'


def encoded(x):return (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


def prepare():
    assert not (PRIVATE/'cloud_active.json').exists(),'Existing deployment: inspect, never silently replace'
    build();PRIVATE.mkdir(parents=True,exist_ok=True)
    active=json.loads((PARENT_PRIVATE/'cloud_active.json').read_text(encoding='utf-8'))
    report=PARENT_PRIVATE/'management/user_paused_paired_accuracy_20261103.json'
    paused=json.loads(report.read_text(encoding='utf-8'))
    assert active['status']=='PAUSED_BY_USER' and active['automation_status']=='PAUSED'
    assert paused['common_question_count']==225 and paused['all_owned_processes_exited']
    common={tuple(k)+(10,) for k in paused['common_question_keys']}
    assert all(len(k)==3 for k in common)
    # Only immutable answer/plan data return over SSH; never printed to the chat.
    script='''import json,sys,base64,hashlib,os
from pathlib import Path
run=Path(PARENT);sys.path.insert(0,str(run))
from manager_recovery import install
install()
from cloud_common import load,verify,sha,identity,all_rows,key
source=verify();lock=load(run/'run_lock.json')
assert sha(run/'run_lock.json')==PARENT_LOCK
assert all(sha(run/n)==e['sha256'] for n,e in lock['snapshots'].items())
for w in load(run/'workers.json',[]):assert identity(w['pid']) is None,'Original run worker alive'
mi=load(run/'manager.json');assert not mi or identity(mi['identity']['pid']) is None,'Original manager alive'
budget=load(run.parents[1]/'cumulative_budget.json')
assert not any(x.get('metadata',{}).get('experiment')==run.name for x in budget['active_reservations'].values())
cohort=load(run/'cohort_questions.json');keys=set(map(tuple,COMMON))
questions=[q for q in cohort if (q['dataset'],q['question_id'],10) in keys]
assert len(questions)==225 and sum(q['is_diamond'] for q in questions)==106
rows=[json.loads(x) for x in (run/'baseline_results.jsonl').read_bytes().splitlines() if x]
rows+=[json.loads(x) for x in (run/'reused_results.jsonl').read_bytes().splitlines() if x]
rows+=all_rows()
reference=[r for r in rows if (r['dataset'],r['question_id'],r['latent_steps']) in keys]
methods=['full','headobf','jive','receiver_attention_sink_recent','T','R','R-shuffle']
assert len(reference)==len({key(r) for r in reference})==225*7
assert {key(r) for r in reference}=={(*k,m) for k in keys for m in methods}
proof=dict(source=source,parent_run=str(run),parent_lock_sha256=sha(run/'run_lock.json'),
 parent_snapshots=lock['snapshots'],parent_own_processes_exited=True,parent_own_reservations=0,
 methods=methods,reference_count=len(reference),question_count=225,diamond_count=106,
 old_run_stays_paused=True,model_forward_calls=0)
print(json.dumps(dict(questions=questions,reference=reference,proof=proof,policy=load(run/'runtime_policy.json'),
 parent_plan=load(run/'plan.json'),base_source_lock_sha256=lock['base_source_lock_sha256']),ensure_ascii=False))
'''.replace('PARENT_LOCK',repr(active['source_lock_sha256'])).replace('PARENT',repr(active['run'])).replace('COMMON',repr([list(x) for x in sorted(common)]))
    source=json.loads(call(script,180));questions=source['questions'];reference=source['reference']
    # Validate against local frozen cohort without exposing its contents.
    local=json.loads((PARENT_PRIVATE/'cohort_questions.json').read_text(encoding='utf-8'))
    byid={q['question_id']:q for q in local}
    assert all(q==byid[q['question_id']] for q in questions)
    source['proof'].update(local_paused_report_sha256=sha(report),local_parent_cohort_sha256=sha(PARENT_PRIVATE/'cohort_questions.json'))
    policy=source['policy']
    policy['historical_loading_failure'].update({str(g):dict(mode='manual_clearance',
        reason=reason) for g,reason in [(3,'Archived previous loading OOM; require independently verified repair/resource improvement'),
                                       (6,'Archived previous Judger OOM; require independently verified repair/resource improvement')]})
    plan=dict(source['parent_plan'])
    plan.update(kind='attention_mass_G_matched225',dataset_scope='225 previously viewed common questions, includes Diamond106 and nonDiamond119',
        methods=['G'],total_target=225,reused=0,new_target=225,case_jobs=225,baseline_readonly=1575,
        comparison_methods=8,combined_comparison_target=1800,diamond_new_methods_target=106,diamond_comparison_target=848,
        role_objective='G: sum attention over identical 64 queries; top88 free prompt plus mandatory8 in same <=192 bank',
        primary_test='G versus T, paired two-sided exact McNemar alpha .05, exploratory already-viewed225',
        evaluation='No unfinished accuracy unless explicitly requested; completed225: all8 on same225 + Diamond106/nonDiamond119 separately',
        old_run_stays_paused=True,reference_methods=source['proof']['methods'],user_authorization='2026-11-03: 你先去启动G的实验吧')
    queue=dict(tasks=[dict(id=f'case_{i:04d}',item_index=i,question_id=q['question_id'],steps=10,
             methods=['G'],state='pending',priority=i) for i,q in enumerate(questions)])
    files={n:p.read_bytes() for p in HERE.iterdir() if p.is_file() and (n:=p.name) in [
        'cloud_worker.py','block_selector.py','attention_selector.py','manager.py','manager_recovery.py','management_identity.py',
        'cloud_common.py','stage_ops.py','current_jive.py','author_parser.py','queue_logic.py','inspect_run.py','worker_compatibility.json',
        'protocol.md','README.md','task_plan.md','findings.md','progress.md','test_attention_selector.py']}
    for n,x in dict(plan=plan,runtime_policy=policy,questions=questions,cohort_questions=questions,queue_initial=queue,source_reference=source['proof']).items():files[n+'.json']=encoded(x)
    files['baseline_results.jsonl']=b''.join((json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n').encode('utf-8') for r in reference)
    files['reused_results.jsonl']=b''
    def entry(b):return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
    lock=dict(base_source_lock_sha256=source['base_source_lock_sha256'],parent_run_lock_sha256=active['source_lock_sha256'],
        files={n:entry(b) for n,b in files.items() if n not in ['baseline_results.jsonl','reused_results.jsonl']},
        snapshots={n:entry(files[n]) for n in ['baseline_results.jsonl','reused_results.jsonl']})
    files['run_lock.json']=encoded(lock)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name='attention_G_gpqa_matched225_s42_l10_'+stamp;run=ROOT+'/evidence/'+name
    for n,b in files.items():(PRIVATE/n).write_bytes(b)
    files['queue.json']=encoded(queue)
    payload={n:base64.b64encode(b).decode() for n,b in files.items()}
    receipt=json.loads(call('''import json,base64
from pathlib import Path
run=Path(RUN_VALUE);assert not run.exists();run.mkdir();(run/'attempts').mkdir()
for n,b in PAYLOAD.items():(run/n).write_bytes(base64.b64decode(b))
print(json.dumps(dict(staged=True,run=str(run),manager_launched=False)))
'''.replace('RUN_VALUE',repr(run)).replace('PAYLOAD',repr(payload)),180))
    active_g=dict(run=run,run_name=name,status='STAGED_PREFLIGHT',target=225,new_target=225,reused=0,baseline_reused=1575,
        max_workers=6,allowed_gpus=list(range(8)),local_package=str(PRIVATE),manager_entry='manager_recovery.py',
        source_lock_sha256=hashlib.sha256(files['run_lock.json']).hexdigest(),parent_run=active['run'],old_run_stays_paused=True)
    (PRIVATE/'cloud_active.json').write_bytes(encoded(active_g))
    (HERE/'run_reference.json').write_bytes(encoded({k:v for k,v in active_g.items() if k not in ['run','parent_run','local_package']}))
    print(json.dumps(dict(**receipt,questions=225,diamond=106,reference_answers=1575,source_lock_sha256=active_g['source_lock_sha256']),ensure_ascii=False))


def launch():
    active=json.loads((PRIVATE/'cloud_active.json').read_text(encoding='utf-8'));run=active['run']
    assert active['status']=='STAGED_PREFLIGHT'
    raw=call('''import json,os,sys,time,subprocess,fcntl
from pathlib import Path
from types import SimpleNamespace
run=Path(RUN_VALUE);root=run.parents[1];assert not (run/'manager.json').exists(),'already launched'
sys.path.insert(0,str(run));from cloud_common import verify,sha,identity,save,load
source=verify();lock=load(run/'run_lock.json')
assert sha(run/'run_lock.json')==LOCK_VALUE
assert all(sha(run/n)==e['sha256'] for n,e in lock['snapshots'].items())
parent=Path(PARENT_VALUE)
for w in load(parent/'workers.json',[]):assert identity(w['pid']) is None
pm=load(parent/'manager.json');assert not pm or identity(pm['identity']['pid']) is None
assert sha(parent/'run_lock.json')==lock['parent_run_lock_sha256']
base=root/'evidence/gpqa_repair_8096_20261028T161254Z/bundle';sys.path.insert(0,str(base/'runtime'))
import experiment_worker as ew
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/home/liuxin/models/Qwen3-8B',local_files_only=True)
args=SimpleNamespace(task='gpqa',model_name='/home/liuxin/models/Qwen3-8B',prompt='sequential',think=False,method='latent_mas',text_mas_context_length=-1,latent_space_realign=False)
bounds=[]
for item in load(run/'questions.json'):
 lengths={r:len(tok(tok.apply_chat_template(ew.stage_prompt(r,item['question'],args),tokenize=False,add_generation_prompt=True),add_special_tokens=False)['input_ids']) for r in ['planner','critic','refiner','judger']}
 p,c,r,j=[lengths[k] for k in ['planner','critic','refiner','judger']]
 context=p+c+r+j+30+8192;assert context<=32768
 bounds.append(dict(question_id=item['question_id'],prompt_lengths=lengths,max_positions=p+2*c+2*r+2*j+30+8192,max_original_context=context))
maximum=sum(b['max_positions'] for b in bounds)
with (root/'budget.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);budget=load(root/'cumulative_budget.json')
 before={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']}
 assert sum(before.values())+maximum<=115000000
 assert not any(x.get('metadata',{}).get('experiment')==parent.name for x in budget['active_reservations'].values())
save(run/'launch_preflight.json',dict(source=source,model_forward_calls=0,exact_tokenizer_bound=maximum,
 max_original_context=max(x['max_original_context'] for x in bounds),budget_before=before,per_question=bounds,
 old_run_exited_and_no_reservations=True,old_run_stays_paused=True))
with (run/'manager.log').open('ab') as log:
 proc=subprocess.Popen([sys.executable,str(run/'manager_recovery.py')],cwd=run,env=dict(os.environ,PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
 ident=identity(proc.pid)
 if ident and ident['argv']==[sys.executable,str(run/'manager_recovery.py')]:break
 time.sleep(.02)
assert ident and ident['uid']==os.getuid()
receipt=dict(manager=ident,budget_before=before,new_max_positions=maximum,model_forward_calls_preflight=0,
 max_original_context=max(x['max_original_context'] for x in bounds),old_run_stays_paused=True)
save(run/'launch_receipt.json',receipt)
print(json.dumps(receipt))
'''.replace('RUN_VALUE',repr(run)).replace('PARENT_VALUE',repr(active['parent_run'])).replace('LOCK_VALUE',repr(active['source_lock_sha256'])),180)
    receipt=json.loads(raw);(PRIVATE/'launch_receipt.json').write_bytes(raw)
    active.update(status='RUNNING',manager_pid=receipt['manager']['pid'],budget_before=receipt['budget_before'],max_new_positions=receipt['new_max_positions'])
    (PRIVATE/'cloud_active.json').write_bytes(encoded(active));print(raw.decode('utf-8'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','launch']);arg=ap.parse_args()
    (prepare if arg.action=='prepare' else launch)()
