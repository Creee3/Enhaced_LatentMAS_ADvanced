"""Stage new experiment only; no model weights, no model construction or launch."""
import base64,datetime,hashlib,json,sys
from pathlib import Path
from remote import HERE,PRIVATE,call
METHODS=['Full','Gen','H','L','H-OBF','L-OBF','G','T','R','R-shuffle','Pseudo-R','Random-matched']
def stage():
 path=PRIVATE/'cloud_active.json'
 if path.exists():run=json.loads(path.read_text())['run']
 else:
  stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');run='/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025/evidence/role_geometry120_s42_l10_'+stamp
  path.write_text(json.dumps(dict(run=run,status='CPU_PREPARATION_NOT_STARTED'),indent=2)+'\n',encoding='utf-8')
 payload={}
 for root,prefix in [(HERE,''),(PRIVATE/'dependencies','dependencies/')]:
  for p in root.rglob('*') if prefix else root.iterdir():
   if not p.is_file() or '__pycache__' in p.parts or p.name.startswith('.') :continue
   if not prefix and p.suffix not in ['.py','.md','.json','.yaml']:continue
   n=prefix+p.relative_to(root).as_posix();payload[n]=base64.b64encode(p.read_bytes()).decode()
 for f in (PRIVATE/'author').iterdir():
  if f.is_file():payload['author/'+f.name]=base64.b64encode(f.read_bytes()).decode()
 for name in ['questions.json','data_manifest.json','author_source_lock.json']:payload[name]=base64.b64encode((PRIVATE/name).read_bytes()).decode()
 plan=dict(kind='role_geometry120',cases=120,methods=METHODS,target_keys=1440,seed=42,latent_steps=10,cap=8192,context=32768,temperature=.6,top_p=.95,top_k=20,max_workers=6,max_per_gpu=4,batch_size=30,engineering_indices=[0,20,40,60,80,100],gpus=[dict(index=1,uuid='GPU-250104ae-64db-3b01-4660-a580491ebc0d',max_workers=4),dict(index=0,uuid='GPU-b9ef4c83-1acf-672a-8404-b464bf3cb622',max_workers=2)],cumulative_position_limit=115000000,user_authorized=True,compute_incomplete_accuracy=False,common_snapshot_receiver_methods=METHODS[1:])
 payload['plan.json']=base64.b64encode((json.dumps(plan,indent=2)+'\n').encode()).decode()
 script=r"""
from pathlib import Path
import base64,json,os,subprocess,sys
r=Path(RUN);r.mkdir(exist_ok=True);assert not (r/'run_lock.json').exists(),'frozen source cannot be restaged'
for n,b in DATA.items():
 p=r/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(base64.b64decode(b))
env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
p=subprocess.run([sys.executable,str(r/'preflight.py')],cwd=r,env=env,capture_output=True,text=True)
if p.returncode:raise RuntimeError(p.stderr+'\n'+p.stdout)
print(p.stdout.strip())
""".replace('RUN',repr(run)).replace('DATA',repr(payload))
 raw=call(script,600);summary=json.loads(raw);(PRIVATE/'preflight_summary.json').write_bytes(raw)
 (PRIVATE/'preflight.json').write_bytes(call('from pathlib import Path\nprint((Path('+repr(run)+')/"preflight.json").read_text())',30))
 print(json.dumps(summary));return summary
if __name__=='__main__':stage()
