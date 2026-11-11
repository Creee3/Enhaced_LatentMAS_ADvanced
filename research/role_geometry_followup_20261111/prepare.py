"""Prepare a fresh private run, tokenizer-only validation, never launch models."""
import base64,datetime,json,subprocess
from pathlib import Path
from remote import HERE,PRIVATE,call
def main():
 PRIVATE.mkdir(parents=True,exist_ok=True)
 assert not (PRIVATE/'cloud_active.json').exists(),'A run already exists; inspect, do not recreate'
 run='/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025/evidence/role_intervention120_s42_l10_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 originals=PRIVATE.parent/'latentmas_role_geometry120_20261110/verified_formal_final_6da4c4764f70'
 files={p.name:base64.b64encode(p.read_bytes()).decode() for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.md']}
 files.update({n:base64.b64encode((originals/n).read_bytes()).decode() for n in ['questions.json','data_manifest.json']})
 script="from pathlib import Path\nimport base64,json,sys,subprocess,os\nr=Path("+repr(run)+");r.mkdir(exist_ok=False);(r/'attempts').mkdir();(r/'management').mkdir()\n"
 script+='files='+repr(files)+"\nfor n,b in files.items():(r/n).write_bytes(base64.b64decode(b))\n"
 script+="sys.path.insert(0,str(r));from common import load,save,OLD,ROOT\nitems=load(r/'questions.json');assert (r/'questions.json').read_bytes()==(OLD/'questions.json').read_bytes()\n"
 script+="groups=list(dict.fromkeys(x['dataset_group'] for x in items));eng=[next(x['index'] for x in items if x['dataset_group']==g) for g in groups]\n"
 script+="save(r/'plan.json',dict(kind='role_intervention120',max_workers=6,max_per_gpu=4,methods=['diagnostic'],engineering_indices=eng,questions=120,requests=2520,contexts=3,roles=['planner','critic','refiner'],variants=2,gpus=[dict(index=1,uuid='GPU-250104ae-64db-3b01-4660-a580491ebc0d',max_workers=3),dict(index=0,uuid='GPU-b9ef4c83-1acf-672a-8404-b464bf3cb622',max_workers=3)]))\n"
 script+="env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')\np=subprocess.run([sys.executable,str(r/'preflight.py')],cwd=r,env=env,capture_output=True,text=True);save(r/'preflight_process.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));print(json.dumps(dict(run=str(r),returncode=p.returncode,summary=p.stdout,error=p.stderr[-3000:])));assert p.returncode==0,p.stderr\n"
 raw=call(script,300);result=json.loads(raw);(PRIVATE/'preparation_verified.json').write_bytes(raw)
 (PRIVATE/'cloud_active.json').write_text(json.dumps(dict(run=run,status='PREPARED_NOT_LAUNCHED'),indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
