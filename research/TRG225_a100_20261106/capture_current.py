"""One UTF8 full private inspect; stdout only counts/process/error summary."""
import datetime,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;PRIVATE=HERE.parents[1].parent/'advanced_MOC/experiments/latentmas_TRG225_a100_20261106'
def call(s,timeout=180):
 p=subprocess.run(['ssh','-o','BatchMode=yes','DevContainer','/home/ubuntu/.conda/envs/kv-graph/bin/python -'],input=s.encode(),capture_output=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')[-4000:])
 return p.stdout
def active():
 PRIVATE.mkdir(exist_ok=True);raw=call("from pathlib import Path\nprint(Path('/home/ubuntu/cloud_migration_20261106/TRG225/cloud_active.json').read_text())")
 (PRIVATE/'cloud_active.json').write_bytes(raw);return json.loads(raw)
def main():
 a=active();run=a['run']
 script="import sys,runpy\nfrom pathlib import Path\nr=Path("+repr(run)+");sys.path.insert(0,str(r));runpy.run_path(str(r/'inspect_run.py'),run_name='__main__')"
 raw=call(script);x=json.loads(raw);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');(PRIVATE/('inspect_'+stamp+'.json')).write_bytes(raw)
 assert all(x['snapshots'].values()) and not x['duplicates'] and not x['unexpected']
 assert sum(x['budget'].values())<=115000000
 summary={k:x[k] for k in ['combined_unique','target','remaining','new_unique','reused','duplicates','unexpected','queue','budget']}
 summary['manager_pid']=x['manager_identity']['pid'] if x['manager_identity'] else None
 summary['workers']=[dict(pid=w['pid'],slot=w.get('slot'),gpu=w['gpu'],live=bool(w['identity']),ready=bool(w['model_ready']),committed=w['committed'],failure=w['failure'].get('error') if w['failure'] else None) for w in x['workers']]
 (PRIVATE/'latest_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
