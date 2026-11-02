"""Manual deployment command for the explicitly authorized smoke, not a test."""
import base64,datetime,hashlib,json
from pathlib import Path
from remote import ROOT,call

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
PRIVATE=REPO.parent/'advanced_MOC/experiments/latentmas_role_blocks_smoke_20261102'
OLD=REPO.parent/'advanced_MOC/experiments/latentmas_receiver_main448_20261101/deployments/receiver_gpqa_main448_s42_l10_l40_20261101T133212Z/extracted/completed/run'

def main():
    PRIVATE.mkdir(parents=True,exist_ok=True)
    assert not (PRIVATE/'active.json').exists(), 'Do not launch duplicate smoke'
    cohort=json.loads((OLD/'cohort_questions.json').read_text(encoding='utf-8'))
    items=[next(x for x in cohort if x['is_diamond']==flag) for flag in [False,True]]
    plan=dict(kind='engineering_smoke_not_effectiveness',seed=42,latent_steps=10,repeats=3,
        questions=[dict(question_id=x['question_id'],is_diamond=x['is_diamond'],question_sha256=x['question_sha256']) for x in items],
        arms=['T','R','R-shuffle'],terminal_answers=18,selection_queries=16,prompt_budget=96,
        protected_latent=30,sink=4,mandatory_refiner_recent=8,bank_max=192,
        output_cap=8192,gpu_pool=list(range(8)),max_workers=1,
        question_selection='First canonical main entry of each membership; no use of gold or historical outcomes',
        fake_roles='Rotate nonzero role block sequence by half the smallest positive block size; block0 unchanged',
        timing='CUDA sync both boundaries; shared prefix/probe once; standalone sum explicitly reconstructed')
    name='role_blocks_smoke2x3_s42_l10_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=ROOT+'/evidence/'+name
    files={}
    for filename in ['smoke_worker.py','block_selector.py']:
        files[filename]=(HERE/filename).read_bytes()
    for filename in ['stage_ops.py','current_jive.py','author_parser.py','cloud_common.py','queue_logic.py']:
        files[filename]=(REPO/'snapshot/run'/filename).read_bytes()
    for filename in ['task_plan.md','findings.md','progress.md']:
        files[filename]=(HERE/filename).read_bytes()
    files['plan.json']=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode()
    files['questions.json']=(json.dumps(items,ensure_ascii=False,indent=2)+'\n').encode()
    lock=dict(base_source_lock_sha256=json.loads((OLD/'run_lock.json').read_text())['base_source_lock_sha256'],
        files={n:dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest()) for n,b in files.items() if not n.endswith('.md')})
    files['run_lock.json']=(json.dumps(lock,indent=2)+'\n').encode()
    for n,b in files.items():(PRIVATE/n).write_bytes(b)
    payload={n:base64.b64encode(b).decode() for n,b in files.items()}
    script='''import base64,json,subprocess,os,sys,time,fcntl
from pathlib import Path
root=Path(ROOT_VALUE);run=Path(RUN_VALUE)
assert not run.exists(), 'duplicate run path'
with (root/'budget.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);b=json.loads((root/'cumulative_budget.json').read_text())
 assert b['actual_tokens']+b['reserved_tokens']+b['unknown_cost_upper_bound_tokens']+400000<=115000000
before={k:b[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']}
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.free,memory.total','--format=csv,noheader,nounits'],text=True)
admission=[]
for line in rows.splitlines():
 i,u,free,total=map(float,line.split(','));admission.append(dict(gpu=int(i),util=u,free_mib=free,total_mib=total,eligible=u<=30 and free/total>=.7))
eligible=[x for x in admission if x['eligible']]
assert eligible, 'no eligible GPU; no model launched'
gpu=max(eligible,key=lambda x:x['free_mib'])['gpu']
run.mkdir()
for n,data in PAYLOAD_VALUE.items():(run/n).write_bytes(base64.b64decode(data))
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),LATENTMAS_SHARED_LEDGER=str(root/'cumulative_budget.json'),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
out=run/'attempts'/('gpu'+str(gpu)+'_attempt1');out.parent.mkdir()
with (run/'worker.log').open('wb') as log:
 p=subprocess.Popen([sys.executable,str(run/'smoke_worker.py'),'--out',str(out)],cwd=run,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
identity=None
for _ in range(10):
 try:
  proc=Path('/proc')/str(p.pid);f=(proc/'stat').read_text().rsplit(')',1)[1].split()
  identity=dict(pid=p.pid,uid=proc.stat().st_uid,start_ticks=f[19],gpu=gpu,argv=[v.decode() for v in (proc/'cmdline').read_bytes().split(b'\\0') if v]);break
 except FileNotFoundError:time.sleep(.1)
active=dict(run=str(run),attempt=str(out),identity=identity,admission=admission,budget_before=before,launched=time.time(),source_lock_sha256=__import__('hashlib').sha256((run/'run_lock.json').read_bytes()).hexdigest())
(run/'active.json').write_text(json.dumps(active,ensure_ascii=False,indent=2)+'\\n',encoding='utf-8')
print(json.dumps(active))
'''.replace('ROOT_VALUE',repr(ROOT)).replace('RUN_VALUE',repr(run)).replace('PAYLOAD_VALUE',repr(payload))
    raw=call(script,180);(PRIVATE/'active.json').write_bytes(raw)
    active=json.loads(raw)
    (HERE/'smoke_config.json').write_text(json.dumps(dict(run_name=name,**plan),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(run=name,identity=active['identity'],budget_before=active['budget_before']),ensure_ascii=False))

if __name__=='__main__':main()
