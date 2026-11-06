"""Explicit authorized G-only deployment on new hardware; no old-run edits."""
import datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025')
OLD=ROOT/'evidence/attention_G_gpqa_matched225_s42_l10_v2_20261103T111226Z'
MANAGEMENT=Path('/home/ubuntu/cloud_migration_20261106/G_a100')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 MANAGEMENT.mkdir(exist_ok=True)
 assert not (MANAGEMENT/'cloud_active.json').exists(),'Already deployed; inspect existing run'
 lock=load(OLD/'run_lock.json')
 assert sha(OLD/'run_lock.json')=='ad5a0edf3b9049f831a53f505ea89ea11ea3bcbfe18233ae09bb15ac86fbfc6a'
 for group in ['files','snapshots']:
  for n,e in lock[group].items():assert sha(OLD/n)==e['sha256']
 assert (OLD/'STOP').exists() and not list((OLD/'attempts').glob('*/results.jsonl'))
 pre=load(OLD/'launch_preflight.json');assert pre['model_forward_calls']==0 and pre['replayed_native_tokenized_positions']
 assert load(Path('/home/ubuntu/cloud_migration_20261106/verification/MODEL_VERIFIED.json'))['all_frozen_shas_match']
 run=ROOT/'evidence'/('attention_G_a100_matched225_s42_l10_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
 run.mkdir();(run/'attempts').mkdir();(run/'management').mkdir()
 for n in set(lock['files'])|set(lock['snapshots']):shutil.copy2(OLD/n,run/n)
 common=(run/'cloud_common.py').read_text();assert 'range(8)' in common
 (run/'cloud_common.py').write_text(common.replace('range(8)','range(4)'))
 policy=load(run/'runtime_policy.json');policy.update(allowed_gpus=list(range(4)),max_workers=2,python=sys.executable,historical_loading_failure={})
 save(run/'runtime_policy.json',policy)
 queue=load(run/'queue_initial.json');assert len(queue['tasks'])==225 and all(t['state']=='pending' and t['methods']==['G'] for t in queue['tasks'])
 save(run/'queue.json',queue)
 newlock=dict(lock)
 newlock['files']={n:{'bytes':(run/n).stat().st_size,'sha256':sha(run/n)} for n in lock['files']}
 save(run/'run_lock.json',newlock)
 assert (run/'cloud_worker.py').read_bytes()==(OLD/'cloud_worker.py').read_bytes()
 with (ROOT/'budget.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX);budget=load(ROOT/'cumulative_budget.json')
  before={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']}
  assert before==dict(actual_tokens=74904762,reserved_tokens=0,unknown_cost_upper_bound_tokens=58106)
  assert not budget['active_reservations'] and sum(before.values())+pre['exact_tokenizer_bound']<=115000000
 sys.path.insert(0,str(run));from cloud_common import verify,identity
 source=verify()
 proof=dict(source=source,parent_run=str(OLD),parent_lock_sha256=sha(OLD/'run_lock.json'),run_lock_sha256=sha(run/'run_lock.json'),
   scientific_worker_byte_identical=True,original_snapshots_byte_identical=True,model_forward_calls=0,
   exact_tokenizer_bound=pre['exact_tokenizer_bound'],max_original_context=pre['max_original_context'],
   frozen_tokenizer_and_native_positions_preflight_reused=True,budget_before=before,
   old_runs_stay_paused=True,max_workers=2,candidate_gpus=list(range(4)),
   historical_failure_scope='Old UUID/4090 failures preserved in old run; new A100 UUID/80GiB hardware, independent admission and one try per card')
 save(run/'launch_preflight.json',proof);save(run/'management/new_host_authorization.json',proof)
 with (run/'manager.log').open('ab') as log:
  process=subprocess.Popen([sys.executable,str(run/'manager_recovery.py')],cwd=run,env=dict(os.environ,PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 for _ in range(100):
  ident=identity(process.pid)
  if ident and ident['argv']==[sys.executable,str(run/'manager_recovery.py')]:break
  time.sleep(.02)
 assert ident
 active=dict(run=str(run),run_name=run.name,status='RUNNING',manager_identity=ident,max_workers=2,target=225,
  source_lock_sha256=sha(run/'run_lock.json'),old_runs_stay_paused=True,automation_status='PAUSED')
 save(MANAGEMENT/'cloud_active.json',active);save(run/'management/cloud_active.json',active)
 print(json.dumps(active))
if __name__=='__main__':main()
