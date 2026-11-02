"""Prepare private frozen payload; explicit launch after CPU/tokenizer preflight."""
import argparse,base64,datetime,hashlib,json,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'role_smoke_20261102'))
from remote import ROOT,PYTHON,call
from build_worker import build

PRIVATE=REPO.parent/'advanced_MOC/experiments/latentmas_role_blocks_main448_20261102'
OLD=REPO.parent/'advanced_MOC/experiments/latentmas_receiver_main448_20261101/deployments/receiver_gpqa_main448_s42_l10_l40_20261101T133212Z/extracted/completed/run'
SMOKE=REPO.parent/'advanced_MOC/experiments/latentmas_role_blocks_smoke_20261102'
SMOKE_EXTRACT=SMOKE/'verified_extracted_df9368018255'/'run'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rawrows(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').split('\n') if s]
def key(r):return r['dataset'],r['question_id'],r['latent_steps'],r['method']
def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()

def prepare():
    assert not (PRIVATE/'cloud_active.json').exists(), 'existing deployment: capture it, do not relaunch'
    build();PRIVATE.mkdir(parents=True,exist_ok=True)
    cohort=json.loads((OLD/'cohort_questions.json').read_text(encoding='utf-8'));byid={q['question_id']:q for q in cohort}
    assert len(cohort)==len(byid)==448 and sum(q['is_diamond'] for q in cohort)==198
    smoke_archive=json.loads((SMOKE/'archive_verification.json').read_text())
    assert smoke_archive['archive_sha256']=='df93680182551bd4f14f92dc18298e9cd34df799808d13b1ffb06f34e2d9bb8b'
    archive=next(SMOKE.glob('*_full_df9368018255.tar.gz'));assert sha(archive)==smoke_archive['archive_sha256']
    smoke=[]
    for n in ['smoke_worker.py','block_selector.py']:
        assert sha(HERE.parent/'role_smoke_20261102'/n)==sha(SMOKE_EXTRACT/n), n
    for n in ['stage_ops.py','current_jive.py','author_parser.py']:
        assert sha(REPO/'snapshot/run'/n)==sha(SMOKE_EXTRACT/n), n
    for p in (SMOKE_EXTRACT/'attempts').glob('*/results.jsonl'):smoke+=rawrows(p)
    assert len(smoke)==18
    reuse=[]
    for qid in sorted({x['question_id'] for x in smoke}):
        assert byid[qid]['question']==next(q for q in json.loads((SMOKE_EXTRACT/'questions.json').read_text()) if q['question_id']==qid)['question']
        for method in ['T','R','R-shuffle']:
            group=[x for x in smoke if x['question_id']==qid and x['method']==method]
            assert len(group)==3 and len({x['token_sha'] for x in group})==len({x['selected_position_sha'] for x in group})==1
            original=next(x for x in group if x['repeat']==1)
            r=dict(original,dataset=byid[qid]['dataset'],is_diamond=byid[qid]['is_diamond'],gold=byid[qid]['gold'],
                engineering_only=False,reused_from=dict(run=SMOKE_EXTRACT.name,source_run=json.loads((SMOKE/'active.json').read_text())['run'],repeat=1,immutable_original_row_sha256=hashlib.sha256(encoded(original)).hexdigest()))
            reuse.append(r)
    assert len(reuse)==len({key(x) for x in reuse})==6
    oldrows=rawrows(OLD/'reused_results.jsonl')
    for p in (OLD/'attempts').glob('*/results.jsonl'):oldrows+=rawrows(p)
    baseline=[x for x in oldrows if x['latent_steps']==10]
    expected_baseline={(q['dataset'],q['question_id'],10,m) for q in cohort for m in ['full','headobf','jive','receiver_attention_sink_recent']}
    assert len(baseline)==len({key(x) for x in baseline})==1792 and {key(x) for x in baseline}==expected_baseline
    used={r['question_id'] for r in reuse};items=[q for q in cohort if q['question_id'] not in used]
    assert len(items)==446
    name='role_blocks_gpqa_main448_s42_l10_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    policy=dict(allowed_gpus=list(range(8)),max_workers=6,python=PYTHON,poll_seconds=10,util_max=30,free_fraction_min=.70,
        historical_loading_failure={'4':dict(mode='manual_clearance',reason='prior full loading OOM; same ~24081MiB snapshot did not establish a repair'),
            '5':dict(mode='previously_demonstrated_resource_improvement',failed_free_mib=17253,demonstrated_recovery_free_mib=24081,
                failure_archive_verified='receiver_gpqa32 loading failure archived; historical unknown not resettled')})
    plan=dict(kind='main448_fixed_role_blocks',dataset_scope='GPQA-main448 includes Diamond198',latent_steps=10,seed=42,
        methods=['T','R','R-shuffle'],repeat=1,total_target=1344,reused=6,new_target=1338,case_jobs=446,
        baseline_readonly=1792,comparison_methods=7,combined_comparison_target=3136,
        diamond_new_methods_target=594,diamond_comparison_target=1386,
        output_cap=8192,context_cap=32768,selection_queries=16,prompt_budget=96,protected_latent=30,sink=4,
        mandatory_refiner_recent=8,bank_max=192,role_objective='four block MSE including self/sink; T matches sum; R-shuffle shifted equal-size boundaries',
        handoffs='first two frozen receiver; third all available prompt candidates; original KV gather',
        primary_test='paired McNemar R versus T; real-boundary interpretation requires R versus R-shuffle',
        evaluation='frozen parser accuracy and KV bytes; time auxiliary only, no unfinished accuracy',
        timing='CUDA sync both boundaries; diagnostics included; prefix/probe shared; reconstructed standalone label')
    queue=dict(tasks=[dict(id=f'case_{i:04d}',item_index=i,question_id=q['question_id'],steps=10,
            methods=plan['methods'],state='pending',priority=i) for i,q in enumerate(items)])
    files={}
    for name0 in ['cloud_worker.py','block_selector.py','manager.py','inspect_run.py','worker_compatibility.json']:
        files[name0]=(HERE/name0).read_bytes()
    for name0 in ['cloud_common.py','stage_ops.py','current_jive.py','author_parser.py','queue_logic.py']:
        files[name0]=(REPO/'snapshot/run'/name0).read_bytes()
    for name0 in ['task_plan.md','progress.md','findings.md','README.md']:files[name0]=(HERE/name0).read_bytes()
    for k,v in dict(plan=plan,runtime_policy=policy,questions=items,cohort_questions=cohort,queue_initial=queue).items():files[k+'.json']=encoded(v)
    files['reused_results.jsonl']=b''.join((json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n').encode() for x in reuse)
    files['baseline_results.jsonl']=b''.join((json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n').encode() for x in baseline)
    reuse_index=dict(smoke_source_run=json.loads((SMOKE/'active.json').read_text())['run'],smoke_source_lock_sha256=sha(SMOKE_EXTRACT/'run_lock.json'),smoke_archive=smoke_archive,
        parent_main_run=OLD.parents[2].name,parent_main_source_lock_sha256=sha(OLD/'run_lock.json'),parent_main_archive_sha256='11393167cf8d3789d7f6dd492027582800b223da2ed8f91fd9fa1ac4ae067eea',
        reused_keys=[list(key(r)) for r in reuse],reused=6,baseline=1792,repeated_engineering_rows_not_counted=12,
        snapshot_sha256={n:hashlib.sha256(files[n]).hexdigest() for n in ['reused_results.jsonl','baseline_results.jsonl']})
    files['reuse_index.json']=encoded(reuse_index)
    def entry(b):return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
    lock=dict(base_source_lock_sha256=json.loads((OLD/'run_lock.json').read_text())['base_source_lock_sha256'],
        files={n:entry(b) for n,b in files.items() if not n.endswith(('.md','.jsonl'))},
        snapshots={n:entry(files[n]) for n in ['reused_results.jsonl','baseline_results.jsonl']})
    files['run_lock.json']=encoded(lock)
    for n,b in files.items():(PRIVATE/n).write_bytes(b)
    files['queue.json']=encoded(queue)
    run=ROOT+'/evidence/'+name
    payload={n:base64.b64encode(b).decode() for n,b in files.items()}
    script='''import base64,json
from pathlib import Path
run=Path(RUN_VALUE);assert not run.exists();run.mkdir();(run/'attempts').mkdir()
for n,b in PAYLOAD_VALUE.items():(run/n).write_bytes(base64.b64decode(b))
print(json.dumps(dict(run=str(run),staged=True,manager_launched=False)))
'''.replace('RUN_VALUE',repr(run)).replace('PAYLOAD_VALUE',repr(payload))
    raw=call(script,180)
    active=dict(run=run,run_name=name,status='staged_preflight',local_package=str(PRIVATE),manager_entry='manager.py',target=1344,new_target=1338,reused=6,baseline_reused=1792,max_workers=6,allowed_gpus=list(range(8)),source_lock_sha256=hashlib.sha256(files['run_lock.json']).hexdigest())
    (PRIVATE/'cloud_active.json').write_bytes(encoded(active))
    (HERE/'run_reference.json').write_bytes(encoded({k:active[k] for k in ['run_name','target','new_target','reused','baseline_reused','max_workers','allowed_gpus','source_lock_sha256']}))
    print(raw.decode())

def launch():
    a=json.loads((PRIVATE/'cloud_active.json').read_text());run=a['run']
    script='''import json,os,sys,time,subprocess,fcntl
from pathlib import Path
from types import SimpleNamespace
run=Path(RUN_VALUE);root=run.parents[1];assert not (run/'manager.json').exists(), 'already launched'
sys.path.insert(0,str(run));from cloud_common import verify,sha,identity,save
source=verify()
for n,e in json.loads((run/'run_lock.json').read_text())['snapshots'].items():assert sha(run/n)==e['sha256']
base=root/'evidence/gpqa_repair_8096_20261028T161254Z/bundle';sys.path.insert(0,str(base/'runtime'))
import experiment_worker as ew
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/home/liuxin/models/Qwen3-8B',local_files_only=True)
args=SimpleNamespace(task='gpqa',model_name='/home/liuxin/models/Qwen3-8B',prompt='sequential',think=False,method='latent_mas',text_mas_context_length=-1,latent_space_realign=False)
questions=json.loads((run/'questions.json').read_text());bounds=[]
for item in questions:
 lengths={r:len(tok(tok.apply_chat_template(ew.stage_prompt(r,item['question'],args),tokenize=False,add_generation_prompt=True),add_special_tokens=False)['input_ids']) for r in ['planner','critic','refiner','judger']}
 p,c,r,j=[lengths[k] for k in ['planner','critic','refiner','judger']]
 context=p+c+r+j+30+8192;assert context<=32768
 bounds.append(dict(question_id=item['question_id'],prompt_lengths=lengths,max_positions=p+2*c+2*r+4*j+30+3*8192,max_original_context=context))
maximum=sum(x['max_positions'] for x in bounds)
with (root/'budget.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);budget=json.loads((root/'cumulative_budget.json').read_text())
 before={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']}
 assert sum(before.values())+maximum<=115000000
proof=dict(source=source,model_forward_calls=0,exact_tokenizer_bound=maximum,max_original_context=max(x['max_original_context'] for x in bounds),budget_before=before,per_question=bounds)
save(run/'launch_preflight.json',proof)
with (run/'manager.log').open('ab') as log:
 proc=subprocess.Popen([sys.executable,str(run/'manager.py')],cwd=run,env=dict(os.environ,PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
 ident=identity(proc.pid)
 if ident and ident['argv']==[sys.executable,str(run/'manager.py')]:break
 time.sleep(.02)
assert ident
save(run/'launch_receipt.json',dict(identity=ident,launched=time.time(),preflight=dict(bound=maximum,budget_before=before)))
print(json.dumps(dict(manager=ident,budget_before=before,new_max_positions=maximum,max_original_context=proof['max_original_context'])))
'''.replace('RUN_VALUE',repr(run))
    raw=call(script,180);receipt=json.loads(raw);(PRIVATE/'launch_receipt.json').write_bytes(raw)
    a.update(status='running',manager_pid=receipt['manager']['pid'],budget_before=receipt['budget_before'],max_new_positions=receipt['new_max_positions'])
    (PRIVATE/'cloud_active.json').write_bytes(encoded(a));print(raw.decode())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','launch']);a=ap.parse_args()
    (prepare if a.action=='prepare' else launch)()
