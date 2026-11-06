"""Explicit new 225x3 deployment, strict pilot reuse and bounded GPU1 slots."""
import ast,datetime,fcntl,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025')
OLD=ROOT/'evidence/attention_G_gpqa_matched225_s42_l10_v2_20261103T111226Z'
PILOT=ROOT/'evidence/paired_TG_a100_two_s42_l10_20261106T154043Z'
M=Path('/home/ubuntu/cloud_migration_20261106/TRG225')
def load(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(b) for b in p.read_bytes().split(bytes([10])) if b]
def edit(s,a,b):assert s.count(a)==1,a;return s.replace(a,b)
def main():
 M.mkdir(exist_ok=True);assert not (M/'cloud_active.json').exists()
 lock=load(PILOT/'run_lock.json');assert load(PILOT/'management/paired_verified.json')['passed']
 for g in ['files','snapshots']:
  for n,e in lock[g].items():assert sha(PILOT/n)==e['sha256']
 reused=[]
 for p in (PILOT/'attempts').glob('*/results.jsonl'):reused+=rows(p)
 assert len(reused)==4 and len({(r['question_id'],r['method']) for r in reused})==4
 questions=load(OLD/'questions.json');assert len(questions)==225 and sum(q['is_diamond'] for q in questions)==106
 run=ROOT/'evidence'/('TRG225_a100_s42_l10_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
 run.mkdir();(run/'attempts').mkdir();(run/'management').mkdir()
 for n in set(lock['files'])|set(lock['snapshots']):shutil.copy2(PILOT/n,run/n)
 text=(run/'cloud_worker.py').read_text()
 text=edit(text,"assert plan['methods']==['T','G']","assert plan['methods']==['T','R','G']")
 text=edit(text,'assert len(reference_T)==len(items)==2',"assert len(reference_T)==len(items)==225\n        pilot_ref={x['question_id']:x for x in [json.loads(b) for b in (RUN/'reused_results.jsonl').read_bytes().split(bytes([10])) if b] if x['method']=='T'}\n        reference_T.update(pilot_ref)")
 text=edit(text,"order=['T','G'] if task['item_index']==0 else ['G','T']","order=['T','R','G']; offset=task['item_index']%3; order=order[offset:]+order[:offset]")
 needle="record('prefix_match.json',dict(**current,matched_T_prefix_positions=old_prefix_match,paired_on_new_shared_prefix=True))"
 text=edit(text,needle,"if item['question_id'] in pilot_ref:assert old_prefix_match,'Pilot reuse prefix mismatch; preserve scene'\n                "+needle)
 (run/'cloud_worker.py').write_text(text)
 dispatch=(run/'selector_dispatch.py').read_text();assert "('T','G')" in dispatch
 (run/'selector_dispatch.py').write_text(dispatch.replace("('T','G')","('T','R','G')"))
 manager=(run/'pair_manager.py').read_text().replace('attempts<2','attempts<4').replace('len(live)<2','len(live)<4').replace('attempts==2','attempts==4')
 manager=edit(manager,"first_ready=not workers or load(Path(workers[0]['out'])/'model_ready.json')","first_ready=all(load(Path(w['out'])/'model_ready.json') for w in live)\n   any_failure=any((Path(w['out'])/'failure.json').exists() for w in workers)")
 manager=edit(manager,"eligible=fresh and fresh['free_mib']/fresh['total_mib']>=.70 and (live or fresh['utilization']<=30)","eligible=fresh and not any_failure and ((not live and fresh['free_mib']/fresh['total_mib']>=.70 and fresh['utilization']<=30) or (live and fresh['free_mib']>=21*1024))")
 manager=manager.replace("if not any(owned(w) for w in workers) and attempts==4 and pending:break","if not any(owned(w) for w in workers) and (attempts==4 or any((Path(w['out'])/'failure.json').exists() for w in workers)) and pending:break")
 (run/'pair_manager.py').write_text(manager)
 for n in ['questions.json','cohort_questions.json']:save(run/n,questions)
 shutil.copy2(OLD/'baseline_results.jsonl',run/'baseline_results.jsonl')
 (run/'reused_results.jsonl').write_bytes(b''.join((json.dumps(r,ensure_ascii=False)+'\n').encode() for r in reused))
 plan=dict(kind='TRG225_A100_shared_prefix',methods=['T','R','G'],total_target=675,new_target=671,reused=4,case_jobs=225,baseline_readonly=1575,
  diamond_new_methods_target=318,nonDiamond_target=357,primary_test='R versus T two-sided exact McNemar alpha .05 at completion, exploratory previously viewed225',
  secondary='G comparisons exploratory; no R-shuffle and no role-boundary-specific causal claim',scope='225=106Diamond+119nonDiamond; subset reports separate',
  reuse_protocol='two pilot cases strict same-device same-protocol prefix-position replay; RID differs and recompute cost counted',old_runs_stay_paused=True)
 save(run/'plan.json',plan)
 done={(r['question_id'],r['method']) for r in reused};queue=dict(tasks=[dict(id='case_'+str(i),item_index=i,question_id=q['question_id'],steps=10,methods=[m for m in ['T','R','G'] if (q['question_id'],m) not in done],state='pending',priority=-1 if q['question_id'] in {r['question_id'] for r in reused} else i) for i,q in enumerate(questions)])
 assert sum(len(t['methods']) for t in queue['tasks'])==671
 for n in ['queue.json','queue_initial.json']:save(run/n,queue)
 policy=dict(allowed_gpus=[1],max_workers=4,python=sys.executable,poll_seconds=10,same_gpu_user_authorized=True,initial_util_max=30,initial_free_fraction_min=.70,next_worker_free_mib=21*1024,load_one_at_a_time=True,no_automatic_retries=True)
 save(run/'runtime_policy.json',policy)
 names=set(lock['files']);entry=lambda p:dict(bytes=p.stat().st_size,sha256=sha(p))
 save(run/'run_lock.json',dict(lock,files={n:entry(run/n) for n in names},snapshots={n:entry(run/n) for n in lock['snapshots']}))
 pre=load(OLD/'launch_preflight.json');maximum=3*pre['exact_tokenizer_bound']
 with (ROOT/'budget.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);budget=load(ROOT/'cumulative_budget.json');before={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']}
  assert before==dict(actual_tokens=74917051,reserved_tokens=0,unknown_cost_upper_bound_tokens=58106) and not budget['active_reservations']
  assert sum(before.values())+maximum<=115000000
 save(run/'launch_preflight.json',dict(model_forward_calls=0,conservative_native_position_upper_bound=maximum,source_native_tokenizer_bound=pre['exact_tokenizer_bound'],budget_before=before,pilot_run=str(PILOT),pilot_lock_sha=sha(PILOT/'run_lock.json'),reused=4,new_answers=671))
 sys.path.insert(0,str(run));from cloud_common import verify
 verify()
 functions=lambda p:{n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef)}
 old,new=functions(PILOT/'cloud_worker.py'),functions(run/'cloud_worker.py');fixed=['args_for','timed','context','stage','probe','judge','final_select','validate'];assert all(old[n]==new[n] for n in fixed)
 for n in ['block_selector.py','attention_selector.py','stage_ops.py','author_parser.py']:assert (run/n).read_bytes()==(PILOT/n).read_bytes()
 import torch
 from selector_dispatch import select
 torch.manual_seed(42);a=torch.rand(2,64,12,dtype=torch.double);a/=a.sum(-1,keepdim=True);v=torch.rand(2,12,3,dtype=torch.double)
 labels=torch.zeros(2,12,dtype=torch.long);labels[:,2:]=1;protect=torch.zeros(2,12,dtype=torch.bool);protect[:,:2]=True;mandatory=torch.zeros_like(protect);mandatory[:,-1]=True
 for mode in ['T','R','G']:
  chosen,report=select(a,v,labels,protect,~protect,mandatory,4,mode);assert (chosen.sum(-1)==6).all() and chosen[:,-1].all() and report['bank_rows']==8
 save(run/'management/compatibility.json',dict(passed=True,unchanged_functions=fixed,selectors_byte_identical=True,cpu_dispatch_modes=['T','R','G']))
 with (run/'manager.log').open('ab') as log:p=subprocess.Popen([sys.executable,str(run/'pair_manager.py')],cwd=run,env=dict(os.environ,PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 active=dict(run=str(run),run_name=run.name,status='RUNNING',target=675,new_target=671,reused=4,methods=['T','R','G'],manager_pid=p.pid,max_workers=4,gpu=1,source_lock_sha256=sha(run/'run_lock.json'),old_runs_stay_paused=True)
 save(M/'cloud_active.json',active);save(run/'management/cloud_active.json',active);print(json.dumps(active))
if __name__=='__main__':main()
