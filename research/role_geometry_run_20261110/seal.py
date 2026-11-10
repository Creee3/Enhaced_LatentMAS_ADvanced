"""Freeze new source after CPU/tokenizer/sandbox checks; never launch on import."""
import base64,json
from remote import HERE,PRIVATE,active,call

def seal():
 run=active()['run'];script=r"""
from pathlib import Path
import sys,os,json,hashlib,subprocess
r=Path(RUN);sys.path.insert(0,str(r));from common import load,save,sha,ROOT,SCIENCE,BASE
assert not (r/'run_lock.json').exists()
env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(r),str(SCIENCE),str(BASE)]),ROLE_GEOMETRY_DATA_ROOT=str(r),ROLE_GEOMETRY_AUTHOR_DIR=str(r/'author'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
for name,args in [('cpu_tests',[sys.executable,'-m','unittest','discover','-s',str(r),'-p','test_*.py','-q']),('sandbox',[sys.executable,str(r/'sandbox_preflight.py'),'--run-root',str(r)])]:
 p=subprocess.run(args,cwd=r,env=env,capture_output=True,text=True);save(r/(name+'_verified.json'),dict(passed=p.returncode==0,stdout=p.stdout,stderr=p.stderr,model_forward_calls=0));assert p.returncode==0,p.stderr+'\n'+p.stdout
b=load(ROOT/'cumulative_budget.json');p=load(r/'preflight.json');assert p['passed'] and sum(b[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens'])+p['upper_positions']<=115000000
plan=load(r/'plan.json');items=load(r/'questions.json')
assert len(items)==120 and len(plan['methods'])==12
save(r/'queue.json',dict(engineering_indices=plan['engineering_indices'],tasks=[dict(index=x['index'],dataset=x['dataset'],dataset_group=x['dataset_group'],question_id=x['question_id'],batch=x['index']//30,methods=list(plan['methods']),committed_methods=[],state='pending') for x in items]))
sys.path.insert(0,str(SCIENCE));from cloud_common import verify as old_verify
source=old_verify();names=[p.relative_to(r).as_posix() for p in r.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['queue.json','cpu_tests_verified.json','sandbox_verified.json','progress.md','findings.md','task_plan.md','research-state.yaml']]
save(r/'run_lock.json',dict(files={n:dict(bytes=(r/n).stat().st_size,sha256=sha(r/n)) for n in names},science_run_lock_sha256=source['run_lock_sha256'],source_run=str(SCIENCE),model_lock_sha256=sha(BASE/'model_lock.json'),pretrained_weights_copied=False))
p=subprocess.run([sys.executable,str(r/'manager.py')],cwd=r,env=env,capture_output=True,text=True);assert p.returncode==0,p.stderr
print(json.dumps(dict(passed=True,run=str(r),run_lock_sha256=sha(r/'run_lock.json'),files=len(names),cpu_tests=True,sandbox=True,budget=True,model_forward_calls=0)))
""".replace('RUN',repr(run))
 raw=call(script,600);proof=json.loads(raw);(PRIVATE/'source_sealed.json').write_bytes(raw)
 values=json.loads(call('from pathlib import Path\nimport json,base64\nr=Path('+repr(run)+')\nprint(json.dumps({n:base64.b64encode((r/n).read_bytes()).decode() for n in ["run_lock.json","plan.json","cpu_tests_verified.json","sandbox_verified.json","executor_preflight.json"] if (r/n).exists()}))'))
 for n,b in values.items():(PRIVATE/n).write_bytes(base64.b64decode(b))
 print(json.dumps(proof));return proof
if __name__=='__main__':seal()
