"""Download exact pip-report artifacts in resumable TLS ranges, then SSH upload."""
import concurrent.futures,hashlib,json,subprocess,time,urllib.parse,urllib.request
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
PRIVATE=REPO.parent/'advanced_MOC/experiments/cloud_migration_20261106'
DEST=PRIVATE/'wheelhouse'
REMOTE='/home/ubuntu/cloud_migration_20261106/wheelhouse'
CHUNK=8*1024*1024
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(CHUNK),b''):h.update(b)
 return h.hexdigest()
def main():
 DEST.mkdir(exist_ok=True)
 subprocess.run(['ssh','DevContainer','mkdir -p '+REMOTE],check=True)
 plan=json.loads((PRIVATE/'package_plan.json').read_text())
 receipts=[]
 # Download large artifacts first. Each final wheel is hash-checked before upload.
 for item in plan['install']:
  info=item['download_info'];url=info['url'];name=urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit('/',1)[1])
  assert name.endswith('.whl') and '/' not in name
  want=info['archive_info']['hashes']['sha256'];out=DEST/name
  if not out.exists():
   with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as r:size=int(r.headers['Content-Length'])
   folder=DEST/(name+'.chunks');folder.mkdir(exist_ok=True)
   def chunk(index):
    start=index*CHUNK;end=min(size,start+CHUNK)-1;p=folder/str(index)
    if p.exists() and p.stat().st_size==end-start+1:return p
    for attempt in range(4):
     try:
      request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
      with urllib.request.urlopen(request,timeout=45) as r:
       assert r.status==206 and r.headers['Content-Range'].startswith(f'bytes {start}-')
       data=r.read();assert len(data)==end-start+1
      p.write_bytes(data);return p
     except Exception:
      if attempt==3:raise
      time.sleep(1)
   with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    chunks=list(executor.map(chunk,range((size+CHUNK-1)//CHUNK)))
   with out.with_suffix('.part').open('wb') as f:
    for p in chunks:
     with p.open('rb') as source:
      for b in iter(lambda:source.read(CHUNK),b''):f.write(b)
   out.with_suffix('.part').rename(out)
  assert sha(out)==want,name
  check=subprocess.run(['ssh','DevContainer','test -f '+REMOTE+'/'+name+' && sha256sum '+REMOTE+'/'+name],capture_output=True,text=True)
  if check.returncode or check.stdout.split()[0]!=want:
   subprocess.run(['scp','-q',str(out),'DevContainer:'+REMOTE+'/'+name],check=True)
   check=subprocess.run(['ssh','DevContainer','sha256sum '+REMOTE+'/'+name],check=True,capture_output=True,text=True)
   assert check.stdout.split()[0]==want
  receipts.append({'file':name,'sha256':want,'bytes':out.stat().st_size})
  (PRIVATE/'wheelhouse_verified.json').write_text(json.dumps(receipts,indent=2)+'\n')
  print(json.dumps({'wheel_verified_uploaded':name,'count':len(receipts),'total':len(plan['install'])}),flush=True)
 print(json.dumps({'wheelhouse_complete':True,'files':len(receipts)}),flush=True)
if __name__=='__main__':main()
