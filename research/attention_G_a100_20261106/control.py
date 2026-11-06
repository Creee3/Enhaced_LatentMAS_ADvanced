"""Private full snapshot and independent G audit; no raw answers on stdout."""
import argparse,contextlib,datetime,io,json,runpy,subprocess,sys,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
PRIVATE=REPO.parent/'advanced_MOC/experiments/latentmas_attention_G_a100_20261106'
PYTHON='/home/ubuntu/.conda/envs/kv-graph/bin/python'
def call(source,timeout=120):
 p=subprocess.run(['ssh','-o','BatchMode=yes','DevContainer',PYTHON+' -'],input=source.encode(),capture_output=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stderr.decode('utf-8',errors='replace')[-4000:])
 return p.stdout
def main():
 PRIVATE.mkdir(parents=True,exist_ok=True)
 raw=call("from pathlib import Path\nprint(Path('/home/ubuntu/cloud_migration_20261106/G_a100/cloud_active.json').read_text())")
 (PRIVATE/'cloud_active.json').write_bytes(raw);active=json.loads(raw)
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['capture','verify','archive']);ap.add_argument('--pid',type=int);args=ap.parse_args()
 if args.action in ['verify','archive']:
  assert args.pid
  module=types.ModuleType('deploy');module.HERE=HERE;module.PRIVATE=PRIVATE;module.call=call;sys.modules['deploy']=module
  filename='verify_formal_worker.py' if args.action=='verify' else 'archive_failed_worker.py'
  source=(HERE.parent/'attention_control225_20261103'/filename).read_text(encoding='utf-8').replace('.read_bytes().splitlines()', '.read_bytes().split(bytes([10]))')
  if args.action=='archive':
   insert="for extra in (run/'management').rglob('*'):\n if extra.is_file():add('run_snapshot/management/'+extra.relative_to(run/'management').as_posix(),extra)\nadd('run_snapshot/STOP',run/'STOP')\nadd('ledger_snapshot/cumulative_budget.json',run.parents[1]/'cumulative_budget.json')\ntracefile=out/'trace.jsonl'"
   source=source.replace("tracefile=out/'trace.jsonl'",insert)
  sys.argv=['verify_formal_worker.py','--pid',str(args.pid)]
  with contextlib.redirect_stdout(io.StringIO()) as output:exec(compile(source,'reused_G_management_adapter','exec'),{'__name__':'__main__'})
  value=json.loads(output.getvalue())
  if args.action=='archive':print(json.dumps({k:value[k] for k in ['archive_sha256','bytes','members','stream_sha','gzip_crc','tar_exact_members','all_member_sha','safe_full_extract_sha']}));print(json.dumps({k:value['cost'][k] for k in ['pid','gpu','committed','request_starts','raw_records','own_reserved','unknown','actual','manual_settlements']}))
  else:print(json.dumps(value))
  return
 script="import sys,runpy\nfrom pathlib import Path\np=Path("+repr(active['run'])+")\nsys.path.insert(0,str(p))\nrunpy.run_path(str(p/'inspect_run.py'),run_name='__main__')"
 raw=call(script);value=json.loads(raw)
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 (PRIVATE/('inspect_'+stamp+'.json')).write_bytes(raw)
 print(json.dumps({k:value[k] for k in ['combined_unique','target','remaining','duplicates','unexpected','queue','budget']}))
 print(json.dumps({'manager_pid':value['manager_identity']['pid'] if value['manager_identity'] else None,
 'workers':[{'pid':w['pid'],'gpu':w['gpu'],'live':bool(w['identity']),'ready':bool(w['model_ready']),'committed':w['committed'],'failure':w['failure'].get('error') if w['failure'] else None} for w in value['workers']]}))
if __name__=='__main__':main()
