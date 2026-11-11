"""Seal source and CPU/tokenizer evidence before any model launch."""
import base64,json
from remote import HERE,PRIVATE,active,call
def seal():
 run=active()['run'];script=r'''
import sys,os,subprocess,json
from pathlib import Path
r=Path(RUN);sys.path.insert(0,str(r));from common import load,save,sha,ROOT,SCIENCE,BASE,OLD
assert not (r/'run_lock.json').exists()
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
p=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(r),'-p','test_*.py','-q'],cwd=r,env=env,capture_output=True,text=True);save(r/'cpu_tests_verified.json',dict(passed=p.returncode==0,stdout=p.stdout,stderr=p.stderr,model_forward_calls=0));assert p.returncode==0,p.stderr
assert load(r/'preflight.json')['passed'] and load(r/'preflight.json')['model_forward_calls']==0
b=load(ROOT/'cumulative_budget.json');pre=load(r/'preflight.json');assert sum(b[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens'])+pre['upper_positions']<=115000000
assert (r/'questions.json').read_bytes()==(OLD/'questions.json').read_bytes();plan=load(r/'plan.json');items=load(r/'questions.json')
save(r/'queue.json',dict(engineering_indices=plan['engineering_indices'],tasks=[dict(index=x['index'],dataset=x['dataset'],dataset_group=x['dataset_group'],question_id=x['question_id'],methods=['diagnostic'],committed_methods=[],state='pending') for x in items]))
sys.path.insert(0,str(SCIENCE));from cloud_common import verify as old_verify
old=old_verify();names=[p.name for p in r.iterdir() if p.is_file() and (p.suffix=='.py' or p.name in ['protocol.md','questions.json','data_manifest.json','plan.json','preflight.json','matched_instructions.json'])]
save(r/'run_lock.json',dict(files={n:dict(bytes=(r/n).stat().st_size,sha256=sha(r/n)) for n in names},science_run_lock_sha256=old['run_lock_sha256'],old_run_lock_sha256=sha(OLD/'run_lock.json'),model_lock_sha256=sha(BASE/'model_lock.json'),no_weights_copied=True))
p=subprocess.run([sys.executable,str(r/'manager.py')],cwd=r,env=env,capture_output=True,text=True);assert p.returncode==0,p.stderr
print(json.dumps(dict(passed=True,run=str(r),run_lock_sha256=sha(r/'run_lock.json'),files=len(names),model_forward_calls=0,cpu_tests=7,requests=2520,upper_positions=pre['upper_positions'],maximum_total=sum(b[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens'])+pre['upper_positions'])))
'''.replace('RUN',repr(run))
 raw=call(script,300);(PRIVATE/'source_sealed.json').write_bytes(raw)
 values=json.loads(call('from pathlib import Path\nimport json,base64\nr=Path('+repr(run)+')\nprint(json.dumps({n:base64.b64encode((r/n).read_bytes()).decode() for n in ["run_lock.json","plan.json","preflight.json","matched_instructions.json","cpu_tests_verified.json"]}))'))
 for n,b in values.items():(PRIVATE/n).write_bytes(base64.b64decode(b))
 print(raw.decode());return json.loads(raw)
if __name__=='__main__':seal()
