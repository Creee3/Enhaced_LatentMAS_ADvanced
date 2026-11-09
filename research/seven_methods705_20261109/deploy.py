"""Explicit two-run CPU staging; never update a launched source tree."""
import argparse,base64,datetime,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
PRIVATE=HERE.parents[2]/'advanced_MOC/experiments/latentmas_seven_methods705_20261109'
sys.path.insert(0,str(HERE.parent/'TRG225_a100_20261106'))
from capture_current import call

def prepare():
 assert not (PRIVATE/'scope_active.json').exists(),'Use existing scope; no duplicate staging'
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 root='/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025/evidence/'
 runs={kind:root+kind+'_705exp_s42_l10_'+stamp for kind in ['HL','extension']}
 files={p.name:base64.b64encode(p.read_bytes()).decode() for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.md','.yaml']}
 for p in (PRIVATE/'author').iterdir():files['author/'+p.name]=base64.b64encode(p.read_bytes()).decode()
 files['author_source_lock.json']=base64.b64encode((PRIVATE/'author_source_lock.json').read_bytes()).decode()
 payload=dict(files=files,questions={kind:json.loads((PRIVATE/('questions705.json' if kind=='HL' else 'questions129.json')).read_text(encoding='utf-8')) for kind in runs})
 script=r'''
from pathlib import Path
import sys,os,json,base64,hashlib,subprocess,shutil
ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025');PREV=ROOT/'evidence/full_receiver576_s42_l10_20261109T052239Z';ROLE=ROOT/'evidence/rolewise_TRG576_s42_l10_20261108T141249Z'
RUNS=__RUNS__;DATA=__PAYLOAD__
assert os.getuid()==1000
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert load(PREV/'formal_final_archive_verified.json')['archive_sha256']=='116be3feac34d96a72e857f49799b80b522053fb469c535c719c7693862a14a8'
assert load(ROLE/'formal_final_archive_verified.json')['archive_sha256']=='438ac46bb046bf7994fc01eb6354f452a22383ff4607070ef87b483256e24361'
refs=[]
for parent in [PREV,ROLE]:
 for p in [parent/'run_lock.json',parent/'questions.json',parent/'final_summary.json',parent/'formal_final_archive_verified.json']+list((parent/'attempts').glob('*/results.jsonl')):
  refs.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
preflights={}
for kind,run in RUNS.items():
 r=Path(run);assert not r.exists();r.mkdir()
 for n,b in DATA['files'].items():
  p=r/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(b))
 questions=DATA['questions'][kind];save(r/'questions.json',questions)
 manifest=load(PREV/'data_manifest.json');manifest.update(questions_sha256=sha(r/'questions.json'),counts={ds:sum(x['dataset']==ds for x in questions) for ds in ['gpqa','mmlu_pro','gsm8k']},total=len(questions));save(r/'data_manifest.json',manifest)
 shutil.copytree(PREV/'source_data',r/'source_data')
 save(r/'reference_lock.json',dict(files=refs,frozen_answers_readonly=True))
 eng=[0,96,376] if kind=='HL' else [0,next(i for i,q in enumerate(questions) if q['is_diamond']!=questions[0]['is_diamond']),next(i for i in range(len(questions)) if i not in [0,next(j for j,q in enumerate(questions) if q['is_diamond']!=questions[0]['is_diamond'])])]
 methods=['H','L'] if kind=='HL' else ['Full','receiver','GGG','RRR','TTT']
 plan=dict(kind=kind,cases=len(questions),methods=methods,target_keys=len(questions)*len(methods),seed=42,latent_steps=10,cap=8192,context=32768,temperature=.6,top_p=.95,top_k=20,max_workers=6,max_per_gpu=3,gpus=[dict(index=1,uuid='GPU-250104ae-64db-3b01-4660-a580491ebc0d'),dict(index=0,uuid='GPU-b9ef4c83-1acf-672a-8404-b464bf3cb622')],cumulative_position_limit=115000000,user_authorized=True,compute_incomplete_accuracy=False,engineering_indices=eng,other_run=str(Path(RUNS['extension' if kind=='HL' else 'HL'])))
 save(r/'plan.json',plan)
 env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=os.pathsep.join([str(r),str(ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z'),str(ROOT/'evidence/gpqa_repair_8096_20261028T161254Z/bundle/runtime')]))
 test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(r),'-p','test_*.py','-v'],cwd=r,env=env,capture_output=True,text=True);save(r/'cpu_tests.json',dict(passed=test.returncode==0,stdout=test.stdout,stderr=test.stderr));assert test.returncode==0,test.stderr
 pre=subprocess.run([sys.executable,str(r/'preflight.py')],cwd=r,env=env,capture_output=True,text=True);assert pre.returncode==0,pre.stderr;preflights[kind]=load(r/'preflight.json')
 save(r/'queue.json',dict(engineering_indices=eng,tasks=[dict(index=q['index'],dataset=q['dataset'],question_id=q['question_id'],methods=list(methods),committed_methods=[],state='pending') for q in questions]))
budget=load(ROOT/'cumulative_budget.json');charged=sum(budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']);upper=sum(x['upper_positions'] for x in preflights.values());assert charged+upper<=115000000
joint=dict(passed=True,charged_before=charged,total_upper_positions=upper,maximum_total=charged+upper,limit=115000000,model_forward_calls=0,subruns={k:{n:v[n] for n in ['total_cases','target_keys','upper_positions','max_original_context']} for k,v in preflights.items()})
for kind,run in RUNS.items():
 r=Path(run);save(r/'scope_budget_proof.json',joint);plan=load(r/'plan.json');plan['future_upper_positions']=preflights['extension']['upper_positions'] if kind=='HL' else 0;save(r/'plan.json',plan)
 # New run source only; original science and pretrained files remain read-only.
 sys.path.insert(0,str(ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z'));from cloud_common import verify
 source=verify();names=[p.relative_to(r).as_posix() for p in r.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['queue.json','cpu_tests.json','progress.md','task_plan.md','findings.md','research-state.yaml'] and p.suffix in ['.py','.json','.md','.yaml','.parquet']]
 save(r/'run_lock.json',dict(files={n:dict(bytes=(r/n).stat().st_size,sha256=sha(r/n)) for n in names},science_run_lock_sha256=source['run_lock_sha256'],source_run=str(ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z'),model_lock_sha256=sha(ROOT/'evidence/gpqa_repair_8096_20261028T161254Z/bundle/runtime/model_lock.json'),pretrained_weights_copied=False))
 check=subprocess.run([sys.executable,str(r/'manager.py'),'--phase','engineering'],cwd=r,capture_output=True,text=True);assert check.returncode==0,check.stderr
joint['runs']=RUNS
print(json.dumps(joint))
'''.replace('__RUNS__',repr(runs)).replace('__PAYLOAD__',repr(payload))
 raw=call(script,600);proof=json.loads(raw);(PRIVATE/'scope_active.json').write_text(json.dumps(dict(status='PREFLIGHT_VERIFIED_NOT_STARTED',**proof),indent=2),encoding='utf-8');(PRIVATE/'joint_preflight.json').write_bytes(raw)
 for kind,run in runs.items():
  d=PRIVATE/kind;d.mkdir(exist_ok=True);(d/'cloud_active.json').write_text(json.dumps(dict(run=run,status='PREFLIGHT_VERIFIED_NOT_STARTED'),indent=2),encoding='utf-8')
  values=json.loads(call('from pathlib import Path\nimport json,base64\nr=Path('+repr(run)+')\nprint(json.dumps({n:base64.b64encode((r/n).read_bytes()).decode() for n in ["preflight.json","run_lock.json","cpu_tests.json","plan.json"]}))'))
  for n,b in values.items():(d/n).write_bytes(base64.b64decode(b))
 print(json.dumps(proof))

if __name__=='__main__':prepare()
