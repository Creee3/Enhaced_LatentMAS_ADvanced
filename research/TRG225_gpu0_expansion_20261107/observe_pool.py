"""Compact startup/resource observation; no raw answers or scoring."""
import datetime,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'TRG225_a100_20261106'))
from capture_current import call,PRIVATE
def main():
 a=json.loads((PRIVATE/'cloud_active.json').read_text())
 script='''
import json,sys,subprocess,time
from pathlib import Path
r=Path(RUN_PATH);sys.path.insert(0,str(r))
from cloud_common import load,identity
from manager import owned
workers=load(r/'workers.json')
def rows(p):return [json.loads(x) for x in p.read_bytes().split(b'\\n') if x] if p.exists() else []
summary=dict(unix=time.time(),manager=identity(load(r/'manager.json')['identity']['pid']),workers=[dict(pid=w['pid'],gpu=w['gpu'],slot=w.get('slot'),live=bool(owned(w)),ready=bool(load(Path(w['out'])/'model_ready.json')),committed=len(rows(Path(w['out'])/'results.jsonl')),methods=sorted({x['method'] for x in rows(Path(w['out'])/'results.jsonl')}),failure=load(Path(w['out'])/'failure.json',{}).get('error')) for w in workers],manager_failure=load(r/'manager_failure.json'),gpu=subprocess.run(['nvidia-smi','-i','0,1','--query-gpu=index,utilization.gpu,memory.used,memory.free,power.draw,power.limit,clocks.sm,clocks.max.sm,pstate,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout)
print(json.dumps(summary))
'''.replace('RUN_PATH',repr(a['run']))
 raw=call(script);x=json.loads(raw)
 (PRIVATE/'management'/('gpu01_status_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json')).write_bytes(raw)
 print(json.dumps(x))
if __name__=='__main__':main()
