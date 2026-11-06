"""Download only the 15 SHA-locked Qwen files, without loading a model."""
import concurrent.futures,fcntl,hashlib,json,os,shutil,threading,time,urllib.error,urllib.request
from pathlib import Path

ROOT=Path('/home/ubuntu/cloud_migration_20261106')
DEST=Path('/home/ubuntu/models/Qwen3-8B')
LOCK=json.loads((ROOT/'metadata/frozen_model_lock.json').read_text())
REVISION=LOCK['revision']
STATUS={};MUTEX=threading.Lock()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def status(name,**values):
    with MUTEX:
        STATUS.setdefault(name,{}).update(values)
        (ROOT/'verification/model_download_progress.json').write_text(json.dumps(STATUS,indent=2)+'\n')
def download(item):
    locks=ROOT/'verification/model_locks';locks.mkdir(exist_ok=True)
    with (locks/(item[0]+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return download_locked(item)
def download_locked(item):
    name,entry=item;dest=DEST/name
    metadata=ROOT/'metadata/public_model_metadata/Qwen3-8B'/name
    if metadata.exists() and not dest.exists():
        assert sha(metadata)==entry['sha256'];shutil.copy2(metadata,dest)
    if dest.exists():
        assert dest.stat().st_size==entry['bytes'] and sha(dest)==entry['sha256']
        status(name,verified=True,bytes=entry['bytes']);return name
    part=DEST/(name+'.part')
    for attempt in range(3):
        offset=part.stat().st_size if part.exists() else 0
        try:
            url=f'https://huggingface.co/Qwen/Qwen3-8B/resolve/{REVISION}/{name}?download=true&migration={int(time.time())}'
            headers={'User-Agent':'latentmas-cloud-migration/1.0'}
            if offset:headers['Range']=f'bytes={offset}-'
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as source:
                append=source.status==206
                if append:assert source.headers['Content-Range'].startswith(f'bytes {offset}-')
                else:assert not offset,'Server refused range resume; partial preserved'
                with part.open('ab' if append else 'wb') as out:
                    last=0
                    for block in iter(lambda:source.read(1024*1024),b''):
                        out.write(block);offset+=len(block)
                        if time.monotonic()-last>10:
                            status(name,bytes=offset,total=entry['bytes'],attempt=attempt+1);last=time.monotonic()
            assert part.stat().st_size==entry['bytes'] and sha(part)==entry['sha256']
            part.rename(dest);status(name,verified=True,bytes=entry['bytes']);return name
        except Exception as error:
            status(name,error=type(error).__name__+': '+str(error)[:200],attempt=attempt+1)
            if attempt==2:raise
            time.sleep(2)
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        for name in executor.map(download,LOCK['files'].items()):
            print(json.dumps({'model_file_verified':name}),flush=True)
    for name,e in LOCK['files'].items():assert sha(DEST/name)==e['sha256']
    result={'passed':True,'model':'Qwen3-8B','revision':REVISION,'files':15,
            'bytes':sum(x['bytes'] for x in LOCK['files'].values()),'all_frozen_shas_match':True,
            'directory':str(DEST),'model_loaded':False,'model_forward_calls':0}
    (ROOT/'verification/MODEL_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
