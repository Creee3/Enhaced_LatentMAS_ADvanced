"""Bounded no-model diagnostic of the two failed admitted GPUs using UUID mapping."""
import json
from deploy import PRIVATE,call


def main():
    active=json.loads((PRIVATE/'cloud_active.json').read_text(encoding='utf-8'))
    raw=call('''import json,sys,os,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
r=Path(RUN);sys.path.insert(0,str(r))
from cloud_common import load,save,identity
from gpu_probe import collect
assert (r/'STOP').exists()
workers=load(r/'workers.json',[])
assert all(identity(w['pid']) is None for w in workers)
def test(w):
 rows,errors=collect([w['gpu']])
 if not rows:return dict(gpu=w['gpu'],telemetry_errors=errors,model_forward_calls=0)
 fresh=rows[0]
 if fresh['utilization']>30 or fresh['free_mib']/fresh['total_mib']<.70:return dict(gpu=w['gpu'],busy=True,fresh=fresh,model_forward_calls=0)
 code="import json,torch; ok=torch.cuda.is_available(); print(json.dumps(dict(available=ok,count=torch.cuda.device_count(),torch=torch.__version__))); torch.cuda.init() if ok else None"
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=fresh['uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 try:
  v=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=45)
  return dict(gpu=w['gpu'],uuid=fresh['uuid'],fresh=fresh,returncode=v.returncode,stdout=v.stdout[-1000:],stderr=v.stderr[-1800:],model_forward_calls=0)
 except subprocess.TimeoutExpired:return dict(gpu=w['gpu'],timeout=True,model_forward_calls=0)
with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(test,workers))
proof=dict(unix=time.time(),results=results,no_model_loading=True,model_forward_calls=0,driver_or_container_modified=False)
save(r/'management/cuda_uuid_diagnostic.json',proof)
print(json.dumps(proof))
'''.replace('RUN',repr(active['run'])),90)
    (PRIVATE/'cuda_uuid_diagnostic.json').write_bytes(raw)
    print(raw.decode('utf-8'))


if __name__=='__main__':main()
