"""CPU-only staging/preflight/lock, then an explicit separate start action."""
import argparse,base64,datetime,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PRIVATE=Path('D:/liuxin_workspace/ICML2027/advanced_MOC/experiments/latentmas_rolewise_TRG576_20261108')
sys.path.insert(0,str(HERE.parent/'TRG225_a100_20261106'))
from capture_current import call
def prepare():
 active_path=PRIVATE/'cloud_active.json'
 if active_path.exists():remote=json.loads(active_path.read_text())['run']
 else:remote='/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025/evidence/rolewise_TRG576_s42_l10_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 files={p.name:base64.b64encode(p.read_bytes()).decode() for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.md','.yaml']}
 for name in ['questions.json','data_manifest.json']:files[name]=base64.b64encode((PRIVATE/name).read_bytes()).decode()
 for p in (PRIVATE/'source_data').rglob('*.parquet'):files['source_data/'+p.relative_to(PRIVATE/'source_data').as_posix()]=base64.b64encode(p.read_bytes()).decode()
 for name in ['author_parser.py','vendor_prompts.py']:files['frozen_test_refs/'+name]=base64.b64encode((PRIVATE/'frozen_source_reference'/name).read_bytes()).decode()
 plan=dict(methods=['GGG','RRR','TTT'],datasets=dict(gpqa=96,mmlu_pro=280,gsm8k=200),target_keys=1728,seed=42,latent_steps=10,cap=8192,context=32768,prompt_budget_per_handoff=96,mandatory_latest_prompt=8,temperature=.6,top_p=.95,top_k=20,max_workers=6,max_per_gpu=4,gpus=[dict(index=1,uuid='GPU-250104ae-64db-3b01-4660-a580491ebc0d'),dict(index=0,uuid='GPU-b9ef4c83-1acf-672a-8404-b464bf3cb622')],cumulative_position_limit=115000000,user_authorized=True,compute_incomplete_accuracy=False)
 files['plan.json']=base64.b64encode((json.dumps(plan,indent=2)+'\n').encode()).decode()
 script=r'''
from pathlib import Path
import base64,json,sys,os,subprocess,hashlib
r=Path(REMOTE);r.mkdir(parents=True,exist_ok=True);assert not (r/'attempts').exists() and not (r/'management').exists(),'Never hot-update a launched experiment'
for n,b in FILES.items():
 p=r/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(b))
sys.path[:0]=[str(r),str(r.parent/'TRG225_a100_s42_l10_20261106T155639Z'),str(r.parent/'gpqa_repair_8096_20261028T161254Z/bundle/runtime')]
from common import load,save,sha,ROOT,SCIENCE,BASE
from cloud_common import verify
source=verify();assert source['all_sha_match']
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',ROLEWISE_FROZEN_SOURCE=str(r/'frozen_test_refs'),PYTHONPATH=os.pathsep.join(sys.path[:3]))
tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(r),'-p','test_*.py','-v'],cwd=r,env=env,capture_output=True,text=True);save(r/'cpu_tests.json',dict(passed=tests.returncode==0,stdout=tests.stdout,stderr=tests.stderr));assert tests.returncode==0,tests.stderr
pre=subprocess.run([sys.executable,str(r/'preflight.py')],cwd=r,env=env,capture_output=True,text=True);assert pre.returncode==0,pre.stderr
items=load(r/'questions.json');manifest=load(r/'data_manifest.json');queue=dict(engineering_indices=manifest['smoke_indices'],tasks=[dict(index=x['index'],dataset=x['dataset'],question_id=x['question_id'],methods=['GGG','RRR','TTT'],committed_methods=[],state='pending') for x in items]);save(r/'queue.json',queue)
names=[p.relative_to(r).as_posix() for p in r.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['run_lock.json','queue.json','cpu_tests.json','staging_verified.json','progress.md','task_plan.md','research-state.yaml'] and not p.name.endswith('_archive_verified.json') and p.suffix in ['.py','.json','.md','.yaml','.parquet']]
lock=dict(files={n:dict(bytes=(r/n).stat().st_size,sha256=sha(r/n)) for n in names},science_run_lock_sha256=source['run_lock_sha256'],model_lock_sha256=sha(BASE/'model_lock.json'),source_run=str(SCIENCE),pretrained_weights_copied=False);save(r/'run_lock.json',lock)
check=subprocess.run([sys.executable,str(r/'manager.py'),'--phase','engineering'],cwd=r,env=env,capture_output=True,text=True);assert check.returncode==0,check.stderr
summary=dict(run=str(r),status='PREFLIGHT_VERIFIED_NOT_STARTED',source=source,new_run_lock_sha256=sha(r/'run_lock.json'),preflight=json.loads(pre.stdout),cpu_tests_passed=True,manager_check_only=json.loads(check.stdout),model_forward_calls=0);save(r/'staging_verified.json',summary);print(json.dumps(summary))
'''.replace('REMOTE',repr(remote)).replace('FILES',repr(files))
 raw=call(script,600);x=json.loads(raw);active_path.write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8');(PRIVATE/'staging_verified.json').write_bytes(raw)
 extra=call('from pathlib import Path\nimport json,base64\nr=Path('+repr(remote)+')\nprint(json.dumps({n:base64.b64encode((r/n).read_bytes()).decode() for n in ["preflight.json","run_lock.json","cpu_tests.json"]}))')
 for n,b in json.loads(extra).items():(PRIVATE/n).write_bytes(base64.b64decode(b))
 print(json.dumps(x))
def start(phase):
 a=json.loads((PRIVATE/'cloud_active.json').read_text());remote=a['run']
 script=r'''
from pathlib import Path
import sys,subprocess,time,json,os
r=Path(REMOTE);sys.path[:0]=[str(r),str(r.parent/'TRG225_a100_s42_l10_20261106T155639Z')];from common import identity,save,verify,load
source=verify();phase=PHASE;assert not (r/'management'/phase).exists()
check=subprocess.run([sys.executable,str(r/'manager.py'),'--phase',phase],cwd=r,capture_output=True,text=True);assert check.returncode==0,check.stderr
folder=r/'launches';folder.mkdir(exist_ok=True)
with (folder/(phase+'.log')).open('ab') as log:p=subprocess.Popen([sys.executable,str(r/'manager.py'),'--phase',phase,'--execute'],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
time.sleep(.3);assert p.poll() is None or p.returncode==0,(folder/(phase+'.log')).read_text()[-2000:];out=dict(run=str(r),phase=phase,pid=p.pid,identity=identity(p.pid),source=source,launched_ns=time.time_ns());save(folder/(phase+'.json'),out);print(json.dumps(out))
'''.replace('REMOTE',repr(remote)).replace('PHASE',repr(phase))
 raw=call(script);(PRIVATE/(phase+'_launch.json')).write_bytes(raw);a.update(status=phase.upper()+'_RUNNING');(PRIVATE/'cloud_active.json').write_text(json.dumps(a,indent=2)+'\n',encoding='utf-8');print(raw.decode())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','start-engineering','start-formal']);a=p.parse_args()
 if a.action=='prepare':prepare()
 else:start(a.action.split('-',1)[1])
