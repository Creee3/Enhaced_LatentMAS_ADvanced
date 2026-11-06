"""Create a new two-question same-prefix experiment; preserve all old runs."""
import datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025')
OLD=ROOT/'evidence/attention_G_gpqa_matched225_s42_l10_v2_20261103T111226Z'
MANAGEMENT=Path('/home/ubuntu/cloud_migration_20261106/paired_TG')
def load(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def replace_once(text,old,new):
 assert text.count(old)==1,old
 return text.replace(old,new)
def main():
 MANAGEMENT.mkdir(exist_ok=True);assert not (MANAGEMENT/'cloud_active.json').exists()
 lock=load(OLD/'run_lock.json');assert sha(OLD/'run_lock.json')=='ad5a0edf3b9049f831a53f505ea89ea11ea3bcbfe18233ae09bb15ac86fbfc6a'
 for group in ['files','snapshots']:
  for n,e in lock[group].items():assert sha(OLD/n)==e['sha256']
 all_questions=load(OLD/'questions.json');questions=[next(q for q in all_questions if q['is_diamond']==flag) for flag in [True,False]]
 assert len({q['question_id'] for q in questions})==2
 run=ROOT/'evidence'/('paired_TG_a100_two_s42_l10_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
 run.mkdir();(run/'attempts').mkdir();(run/'management').mkdir()
 for n in set(lock['files'])|set(lock['snapshots']):shutil.copy2(OLD/n,run/n)
 shutil.copy2(Path(__file__).with_name('pair_manager.py'),run/'pair_manager.py')
 text=(run/'cloud_worker.py').read_text()
 text=replace_once(text,"assert plan['methods']==['G']","assert plan['methods']==['T','G']")
 text=replace_once(text,'assert len(reference_T)==len(items)==225','assert len(reference_T)==len(items)==2')
 old="assert [x['selected_position_sha256'] for x in prefix_transfers]==[x['selected_position_sha256'] for x in reference['prefix_transfers']], 'T prefix position mismatch; preserve scene'"
 text=replace_once(text,old,"old_prefix_match=([x['selected_position_sha256'] for x in prefix_transfers]==[x['selected_position_sha256'] for x in reference['prefix_transfers']])")
 text=text.replace('matched_T_prefix_positions=True','matched_T_prefix_positions=old_prefix_match,paired_on_new_shared_prefix=True')
 text=replace_once(text,"order=['G'];order=order[repeat-1:]+order[:repeat-1]","order=['T','G'] if task['item_index']==0 else ['G','T']")
 text=replace_once(text,'from attention_selector import select','from selector_dispatch import select')
 text=replace_once(text,"row=dict(**current,dataset=", "row=dict(pair_protocol='A100-shared-prefix-v1',**current,dataset=")
 (run/'cloud_worker.py').write_text(text)
 (run/'selector_dispatch.py').write_text("from attention_selector import select as select_g\nfrom block_selector import select as select_t\ndef select(a,v,labels,protected,candidates,mandatory,budget,mode):\n    assert mode in ('T','G')\n    return (select_g if mode=='G' else select_t)(a,v,labels,protected,candidates,mandatory,budget,mode)\n")
 (run/'cloud_common.py').write_text((run/'cloud_common.py').read_text().replace('range(8)','range(4)'))
 for n in ['questions.json','cohort_questions.json']:save(run/n,questions)
 refs=[json.loads(b) for b in (OLD/'baseline_results.jsonl').read_bytes().split(bytes([10])) if b]
 refs=[r for r in refs if r['question_id'] in {q['question_id'] for q in questions}];assert len(refs)==14
 (run/'baseline_results.jsonl').write_bytes(b''.join((json.dumps(r,ensure_ascii=False)+'\n').encode() for r in refs))
 plan=load(run/'plan.json');plan.update(kind='paired_TG_two_A100',methods=['T','G'],total_target=4,new_target=4,case_jobs=2,baseline_readonly=14,diamond_new_methods_target=2,old_run_stays_paused=True)
 save(run/'plan.json',plan)
 queue=dict(tasks=[dict(id='case_'+str(i),item_index=i,question_id=q['question_id'],steps=10,methods=['T','G'],state='pending',priority=i) for i,q in enumerate(questions)])
 for n in ['queue.json','queue_initial.json']:save(run/n,queue)
 policy=load(run/'runtime_policy.json');policy.update(allowed_gpus=[1],max_workers=2,python=sys.executable,historical_loading_failure={},same_gpu_pair_user_authorized=True)
 save(run/'runtime_policy.json',policy)
 names=set(lock['files'])|{'pair_manager.py','selector_dispatch.py'}
 entry=lambda p:dict(bytes=p.stat().st_size,sha256=sha(p))
 newlock=dict(lock,files={n:entry(run/n) for n in sorted(names)},snapshots={n:entry(run/n) for n in lock['snapshots']})
 save(run/'run_lock.json',newlock)
 pre=load(OLD/'launch_preflight.json');bounds=[b for b in pre['per_question'] if b['question_id'] in {q['question_id'] for q in questions}]
 assert len(bounds)==2;maximum=sum(2*b['max_positions'] for b in bounds)
 with (ROOT/'budget.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);budget=load(ROOT/'cumulative_budget.json');before={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']}
  assert before==dict(actual_tokens=74906188,reserved_tokens=0,unknown_cost_upper_bound_tokens=58106) and not budget['active_reservations']
  assert sum(before.values())+maximum<=115000000
 save(run/'launch_preflight.json',dict(model_forward_calls=0,conservative_native_position_upper_bound=maximum,per_question=bounds,budget_before=before,selection='first Diamond and first nonDiamond in frozen225 order, independent of labels',paired_methods=['T','G'],total_new_answers=4))
 sys.path.insert(0,str(run));from cloud_common import verify,identity
 verify()
 # Frozen selectors, parser, shared prefix/probe/judge/gather code remain identical.
 for name in ['attention_selector.py','block_selector.py','author_parser.py','stage_ops.py']:assert (run/name).read_bytes()==(OLD/name).read_bytes()
 import ast
 oldtree=ast.parse((OLD/'cloud_worker.py').read_text());newtree=ast.parse((run/'cloud_worker.py').read_text())
 functions=lambda t:{n.name:ast.dump(n,include_attributes=False) for n in ast.walk(t) if isinstance(n,ast.FunctionDef)}
 of,nf=functions(oldtree),functions(newtree)
 checked=['args_for','timed','context','stage','probe','judge','final_select','validate']
 assert all(of[n]==nf[n] for n in checked)
 import torch
 from selector_dispatch import select
 torch.manual_seed(42);a=torch.rand(2,64,12,dtype=torch.double);a/=a.sum(-1,keepdim=True)
 v=torch.rand(2,12,3,dtype=torch.double);labels=torch.zeros(2,12,dtype=torch.long);labels[:,2:]=1
 protected=torch.zeros(2,12,dtype=torch.bool);protected[:,:2]=True;mandatory=torch.zeros_like(protected);mandatory[:,-1]=True
 for mode in ['T','G']:
  chosen,report=select(a,v,labels,protected,~protected,mandatory,4,mode)
  assert (chosen.sum(-1)==6).all() and chosen[:,-1].all() and report['bank_rows']==8
 save(run/'management/compatibility.json',dict(passed=True,unchanged_functions=checked,selectors_byte_identical=True,old_G_lock=sha(OLD/'run_lock.json'),new_lock=sha(run/'run_lock.json')))
 with (run/'manager.log').open('ab') as log:p=subprocess.Popen([sys.executable,str(run/'pair_manager.py')],cwd=run,env=dict(os.environ,PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 active=dict(run=str(run),run_name=run.name,status='RUNNING',target=4,question_count=2,methods=['T','G'],manager_pid=p.pid,source_lock_sha256=sha(run/'run_lock.json'),old_runs_stay_paused=True)
 save(MANAGEMENT/'cloud_active.json',active);save(run/'management/cloud_active.json',active);print(json.dumps(active))
if __name__=='__main__':main()
