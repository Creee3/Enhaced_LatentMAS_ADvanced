"""Supersede pre-generation staging safely; replay verified token lengths, not imports."""
import base64
import datetime
import hashlib
import json
import shutil
from pathlib import Path
from deploy import PRIVATE,HERE,call,encoded,ROOT,PYTHON
from build import build,sha


def main():
    old=json.loads((PRIVATE/'cloud_active.json').read_text(encoding='utf-8'))
    assert old['status']=='STAGED_PREFLIGHT'
    check=json.loads(call('''import json,sys
from pathlib import Path
r=Path(RUN);sys.path.insert(0,str(r));from cloud_common import load,identity
assert (r/'STOP').exists()
assert not load(r/'workers.json',[])
assert not list((r/'attempts').glob('*/results.jsonl'))
budget=load(r.parents[1]/'cumulative_budget.json')
assert not any(v.get('metadata',{}).get('experiment')==r.name for v in budget['active_reservations'].values())
parent=Path(PARENT)
print(json.dumps(dict(zero_workers=True,zero_answers=True,zero_own_reservations=True,
 parent_preflight=load(parent/'launch_preflight.json'))))
'''.replace('RUN',repr(old['run'])).replace('PARENT',repr(old['parent_run'])),60))
    history=PRIVATE/'staging_history'/old['run_name'];history.mkdir(parents=True,exist_ok=False)
    for p in PRIVATE.iterdir():
        if p.is_file():shutil.copy2(p,history/p.name)
    build()
    plan=json.loads((PRIVATE/'plan.json').read_text(encoding='utf-8'))
    questions=json.loads((PRIVATE/'questions.json').read_text(encoding='utf-8'))
    refs=[json.loads(x) for x in (PRIVATE/'baseline_results.jsonl').read_bytes().split(b'\n') if x]
    ts={r['question_id']:r for r in refs if r['method']=='T'}
    old_bounds={b['question_id']:b for b in check['parent_preflight']['per_question']}
    bounds=[]
    for q in questions:
        row=ts[q['question_id']];rec=row['judger_receipt'];proof=rec['position_proof']
        assert proof['positions']==rec['actual_positions'] and proof['forwards']==rec['native_forward_calls']
        assert proof['sdpa_checks']==proof['forwards']*36
        lens={x['role']:x['prompt_len'] for x in row['prefix_stages']}
        # Native generate: first forward is full prompt; each subsequent forward is one token.
        lens['judger']=rec['actual_positions']-rec['native_forward_calls']+1
        assert lens['judger']>0
        if q['question_id'] in old_bounds:assert lens==old_bounds[q['question_id']]['prompt_lengths']
        p,c,r,j=[lens[k] for k in ['planner','critic','refiner','judger']]
        context=p+c+r+j+30+8192;assert context<=32768
        bounds.append(dict(question_id=q['question_id'],prompt_lengths=lens,max_positions=p+2*c+2*r+2*j+30+8192,
            max_original_context=context,source_judger_rid=rec['reservation_id'],
            historical_exact_tokenizer_preflight_matched=q['question_id'] in old_bounds,
            source='frozen native input-position receipt; all subsequent generation forwards one token'))
    assert len(bounds)==225 and sum(b['historical_exact_tokenizer_preflight_matched'] for b in bounds)==223
    recipe=dict(per_question=bounds,exact_input_position_bound=sum(b['max_positions'] for b in bounds),
        model_forward_calls=0,new_tokenizer_import=False,source='replayed measured tokenizer lengths plus matching frozen223 tokenizer preflight',
        reference_sha256=sha(PRIVATE/'baseline_results.jsonl'),superseded_staging_lock_sha256=old['source_lock_sha256'])
    # Source changes are frozen in a new run; superseded staging remains immutable + STOP.
    plan['startup_version']=2
    (HERE/'protocol.md').write_text((HERE/'protocol.md').read_text(encoding='utf-8')+
        '\n启动v2管理修复：JSONL严格按LF读取，防止合法U+2028分隔字符被误断行；GPU逐物理卡查询，单卡驱动异常只排除该卡。启动预算复用冻结T的实际tokenized输入位置，并逐题对照原223题tokenizer预检；另2题使用已审计native位置收据（首forward为全prompt，之后每forward为单token），零新增forward，避免重复导入阻塞的Transformers预检。算法和生成参数不变。\n',encoding='utf-8')
    files={}
    for name in json.loads((PRIVATE/'run_lock.json').read_text(encoding='utf-8'))['files']:
        p=HERE/name
        files[name]=p.read_bytes() if p.is_file() else (PRIVATE/name).read_bytes()
    files['gpu_probe.py']=(HERE/'gpu_probe.py').read_bytes()
    files['test_gpu_probe.py']=(HERE/'test_gpu_probe.py').read_bytes()
    files['preflight_recipe.json']=encoded(recipe)
    files['plan.json']=encoded(plan)
    files['superseded_staging.json']=encoded(dict(run=old['run'],source_lock_sha256=old['source_lock_sha256'],
        STOP=True,workers=0,answers=0,own_reservations=0,reason='preflight import I/O timeout; pre-generation JSONL and per-GPU telemetry fixes'))
    for name in ['baseline_results.jsonl','reused_results.jsonl']:files[name]=(PRIVATE/name).read_bytes()
    def entry(b):return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
    old_lock=json.loads((PRIVATE/'run_lock.json').read_text(encoding='utf-8'))
    lock=dict(base_source_lock_sha256=old_lock['base_source_lock_sha256'],parent_run_lock_sha256=old_lock['parent_run_lock_sha256'],
        files={n:entry(b) for n,b in files.items() if not n.endswith('.jsonl')},
        snapshots={n:entry(files[n]) for n in ['baseline_results.jsonl','reused_results.jsonl']})
    files['run_lock.json']=encoded(lock)
    files['queue.json']=files['queue_initial.json']
    name='attention_G_gpqa_matched225_s42_l10_v2_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=ROOT+'/evidence/'+name
    entry_source='''import fcntl,os,sys,time,json,subprocess
from pathlib import Path
run=Path(__file__).resolve().parent;sys.path.insert(0,str(run))
from cloud_common import load,save,verify,identity,sha
guard=(run/'bootstrap.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
save(run/'bootstrap.json',dict(identity=identity(os.getpid()),started=time.time()))
try:
 source=verify();lock=load(run/'run_lock.json');root=run.parents[1]
 assert all(sha(run/n)==e['sha256'] for n,e in lock['snapshots'].items())
 parent=Path(PARENT_RUN)
 assert sha(parent/'run_lock.json')==lock['parent_run_lock_sha256']
 for w in load(parent/'workers.json',[]):assert identity(w['pid']) is None
 pm=load(parent/'manager.json');assert not pm or identity(pm['identity']['pid']) is None
 recipe=load(run/'preflight_recipe.json');maximum=recipe['exact_input_position_bound']
 assert sha(run/'baseline_results.jsonl')==recipe['reference_sha256']
 with (root/'budget.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);budget=load(root/'cumulative_budget.json')
  before={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']}
  assert sum(before.values())+maximum<=115000000
  assert not any(v.get('metadata',{}).get('experiment')==parent.name for v in budget['active_reservations'].values())
 save(run/'launch_preflight.json',dict(source=source,budget_before=before,model_forward_calls=0,
  exact_tokenizer_bound=maximum,max_original_context=max(b['max_original_context'] for b in recipe['per_question']),
  per_question=recipe['per_question'],replayed_native_tokenized_positions=True,old_run_stays_paused=True))
 assert not (run/'STOP').exists()
 assert not (run/'manager.json').exists()
 with (run/'manager.log').open('ab') as log:
  proc=subprocess.Popen([sys.executable,str(run/'manager_recovery.py')],cwd=run,env=dict(os.environ,PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(100):
  ident=identity(proc.pid)
  if ident and ident['argv']==[sys.executable,str(run/'manager_recovery.py')]:break
  time.sleep(.02)
 assert ident
 save(run/'launch_receipt.json',dict(manager=ident,budget_before=before,new_max_positions=maximum,
  model_forward_calls_preflight=0,old_run_stays_paused=True))
 save(run/'bootstrap_exit.json',dict(passed=True,unix=time.time()))
except BaseException as exc:
 save(run/'bootstrap_failure.json',dict(error=repr(exc),unix=time.time()));raise
'''.replace('PARENT_RUN',repr(old['parent_run']))
    files['launch_entry.py']=entry_source.encode('utf-8')
    # Launch entry is code: include it in the final lock before any process starts.
    lock['files']['launch_entry.py']=entry(files['launch_entry.py']);files['run_lock.json']=encoded(lock)
    for n,b in files.items():(PRIVATE/n).write_bytes(b)
    payload={n:base64.b64encode(b).decode() for n,b in files.items()}
    raw=call('''import base64,sys,json,subprocess,os,time
from pathlib import Path
r=Path(RUN);assert not r.exists();r.mkdir();(r/'attempts').mkdir();(r/'management').mkdir()
for n,b in PAYLOAD.items():(r/n).write_bytes(base64.b64decode(b))
sys.path.insert(0,str(r));from cloud_common import identity
with (r/'bootstrap.log').open('ab') as log:
 p=subprocess.Popen([sys.executable,str(r/'launch_entry.py')],cwd=r,env=dict(os.environ,PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
 ident=identity(p.pid)
 if ident and ident['argv']==[sys.executable,str(r/'launch_entry.py')]:break
 time.sleep(.02)
print(json.dumps(dict(bootstrap_identity=ident,run=str(r))))
'''.replace('RUN',repr(run)).replace('PAYLOAD',repr(payload)),180)
    receipt=json.loads(raw)
    active=dict(old,run=run,run_name=name,status='STARTING',source_lock_sha256=hashlib.sha256(files['run_lock.json']).hexdigest(),
        superseded_staging=old['run'],bootstrap_identity=receipt['bootstrap_identity'])
    active.pop('latest_summary',None);active.pop('latest_capture',None)
    (PRIVATE/'cloud_active.json').write_bytes(encoded(active))
    (HERE/'run_reference.json').write_bytes(encoded({k:active[k] for k in ['run_name','status','target','new_target','max_workers','source_lock_sha256','old_run_stays_paused']}))
    print(json.dumps(dict(**receipt,target=225,maximum_new_positions=recipe['exact_input_position_bound'],old_run_stays_paused=True)))


if __name__=='__main__':main()
