"""Finalize new, never-launched run sources and joint CPU preflight, before Git/launch."""
from pathlib import Path
import base64,json
from deploy import HERE,PRIVATE,call

def main():
 scope=json.loads((PRIVATE/'scope_active.json').read_text());runs=scope['runs']
 files={p.name:base64.b64encode(p.read_bytes()).decode() for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.md','.yaml']}
 script=r'''
from pathlib import Path
import os,sys,json,base64,subprocess,hashlib
RUNS=__RUNS__;FILES=__FILES__;ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025')
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
pre={};tests={}
for kind,run in RUNS.items():
 r=Path(run);assert not (r/'attempts').exists() and not (r/'management').exists(),'Cannot hot-change launched scientific source'
 for n,b in FILES.items():(r/n).write_bytes(base64.b64decode(b))
 env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONPATH=os.pathsep.join([str(r),str(ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z'),str(ROOT/'evidence/gpqa_repair_8096_20261028T161254Z/bundle/runtime')]))
 result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(r),'-p','test_*.py','-v'],cwd=r,env=env,capture_output=True,text=True);save(r/'cpu_tests.json',dict(passed=result.returncode==0,stdout=result.stdout,stderr=result.stderr));assert result.returncode==0,result.stderr;tests[kind]=dict(passed=True)
 result=subprocess.run([sys.executable,str(r/'preflight.py')],cwd=r,env=env,capture_output=True,text=True);assert result.returncode==0,result.stderr;pre[kind]=load(r/'preflight.json')
b=load(ROOT/'cumulative_budget.json');charged=sum(b[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']);upper=sum(x['upper_positions'] for x in pre.values());assert charged+upper<=115000000
joint=dict(passed=True,charged_before=charged,total_upper_positions=upper,maximum_total=charged+upper,limit=115000000,model_forward_calls=0,subruns={k:{n:v[n] for n in ['total_cases','target_keys','upper_positions','max_original_context']} for k,v in pre.items()},runs=RUNS)
locks={};gates={}
for kind,run in RUNS.items():
 r=Path(run);save(r/'scope_budget_proof.json',joint);plan=load(r/'plan.json');plan['future_upper_positions']=pre['extension']['upper_positions'] if kind=='HL' else 0;save(r/'plan.json',plan)
 lock=load(r/'run_lock.json');names=[p.relative_to(r).as_posix() for p in r.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['run_lock.json','queue.json','cpu_tests.json','sealing_verified.json','progress.md','task_plan.md','findings.md','research-state.yaml'] and p.suffix in ['.py','.json','.md','.yaml','.parquet']]
 lock['files']={n:dict(bytes=(r/n).stat().st_size,sha256=sha(r/n)) for n in names};save(r/'run_lock.json',lock);locks[kind]=sha(r/'run_lock.json')
 check=subprocess.run([sys.executable,str(r/'manager.py'),'--phase','engineering'],cwd=r,env=env,capture_output=True,text=True);assert check.returncode==0,check.stderr
 negative=subprocess.run([sys.executable,str(r/'manager.py'),'--phase','formal'],cwd=r,env=env,capture_output=True,text=True);assert negative.returncode!=0 and not (r/'management').exists() and not (r/'attempts').exists()
 gates[kind]=dict(engineering_check=json.loads(check.stdout),formal_before_engineering_rejected=True,model_forward_calls=0)
save(Path(RUNS['HL'])/'sealing_verified.json',dict(passed=True,tests=tests,gates=gates,run_locks=locks,joint=joint))
print(json.dumps(dict(passed=True,tests=tests,gates=gates,run_locks=locks,joint=joint)))
'''.replace('__RUNS__',repr(runs)).replace('__FILES__',repr(files))
 raw=call(script,600);proof=json.loads(raw);(PRIVATE/'sealing_verified.json').write_bytes(raw);(PRIVATE/'joint_preflight.json').write_text(json.dumps(proof['joint'],indent=2),encoding='utf-8')
 for kind,run in runs.items():
  values=json.loads(call('from pathlib import Path\nimport base64,json\nr=Path('+repr(run)+')\nprint(json.dumps({n:base64.b64encode((r/n).read_bytes()).decode() for n in ["run_lock.json","cpu_tests.json","preflight.json","plan.json"]}))'))
  for n,b in values.items():(PRIVATE/kind/n).write_bytes(base64.b64decode(b))
 scope.update(status='SEALED_PREFLIGHT_VERIFIED_NOT_STARTED',run_locks=proof['run_locks'],joint_preflight=proof['joint']);(PRIVATE/'scope_active.json').write_text(json.dumps(scope,indent=2),encoding='utf-8');print(json.dumps(proof))

if __name__=='__main__':main()
