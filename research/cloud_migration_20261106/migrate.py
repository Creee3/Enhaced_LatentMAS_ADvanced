"""Transfer already verified private backups; never launch an experiment."""
import hashlib,json,subprocess,tarfile,time
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BACKUP=REPO.parent/'advanced_MOC/experiments/cloud_rescue_20261104'
PRIVATE=REPO.parent/'advanced_MOC/experiments/cloud_migration_20261106'
REMOTE='/home/ubuntu/cloud_migration_20261106'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','DevContainer']

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def run(argv,**kw):return subprocess.run(argv,check=True,**kw)

def main():
    PRIVATE.mkdir(parents=True,exist_ok=True)
    assert json.loads((BACKUP/'BACKUP_VERIFIED.json').read_text(encoding='utf-8'))['passed']
    extras={}
    for p in BACKUP.iterdir():
        if p.is_file() and (p.suffix in {'.json','.txt','.md','.bundle'} or p.name=='SHA256SUMS'):
            extras[p.name]=p
    for folder in ['public_model_metadata','local_management_state']:
        for p in (BACKUP/folder).rglob('*'):
            if p.is_file():extras[p.relative_to(BACKUP).as_posix()]=p
    for group in ['current','source']:
        extras[group+'_recovery_manifest.json']=BACKUP/'verified'/group/'recovery_manifest.json'
    extras['frozen_model_lock.json']=REPO/'snapshot/bundle/runtime/model_lock.json'
    with tarfile.open(PRIVATE/'metadata.tar.gz','w:gz') as tar:
        for name,p in sorted(extras.items()):tar.add(p,arcname=name,recursive=False)
    files={g+'.tar.gz':BACKUP/(g+'.tar.gz') for g in ['current','source','history']}
    files['metadata.tar.gz']=PRIVATE/'metadata.tar.gz'
    run(SSH+['mkdir -p '+REMOTE+'/archives '+REMOTE+'/verification'])
    receipts={}
    for name,p in files.items():
        digest=sha(p)
        print(json.dumps({'phase':'upload','file':name,'bytes':p.stat().st_size}),flush=True)
        final=REMOTE+'/archives/'+name
        check=subprocess.run(SSH+['test -f '+final+' && sha256sum '+final],capture_output=True,text=True)
        if check.returncode or check.stdout.split()[0]!=digest:
            run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',str(p),'DevContainer:'+final+'.part'])
            result=run(SSH+['sha256sum '+final+'.part'],capture_output=True,text=True)
            assert result.stdout.split()[0]==digest,'Transfer digest mismatch: '+name
            run(SSH+['mv '+final+'.part '+final])
        receipts[name]={'sha256':digest,'bytes':p.stat().st_size,'remote':final}
        (PRIVATE/'transfer_verified.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({'phase':'verified','file':name}),flush=True)
    run(['scp','-q',str(PRIVATE/'transfer_verified.json'),'DevContainer:'+REMOTE+'/verification/transfer_verified.json'])
    run(['scp','-q',str(HERE/'restore.py'),'DevContainer:'+REMOTE+'/restore.py'])
    print(json.dumps({'transfer_complete':True,'archives':len(receipts)}),flush=True)

if __name__=='__main__':main()
