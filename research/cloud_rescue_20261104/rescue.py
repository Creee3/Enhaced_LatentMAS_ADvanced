"""Stream backup privately, validate exact membership and extract regular files safely."""
import argparse,datetime,hashlib,io,json,os,subprocess,sys,tarfile,time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED,ALL_COMPLETED
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
PRIVATE=REPO.parent/'advanced_MOC/experiments/cloud_rescue_20261104'
PYTHON='/home/liuxin/.conda/envs/kv-graph/bin/python'


def save(p,v):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for i in range(100):
        try:tmp.replace(p);return
        except PermissionError:
            time.sleep(.05)
    # Progress telemetry must never abort an otherwise healthy backup stream.
    if p.name.endswith(('_transfer.json','_verification_progress.json')):return
    tmp.replace(p)


def archive(group):
    PRIVATE.mkdir(exist_ok=True)
    path=PRIVATE/(group+'.tar.gz');assert not path.exists(),'Archive exists: verify it rather than overwrite'
    partial=path.with_suffix(path.suffix+'.part');log=PRIVATE/(group+'_stream.log')
    if partial.exists():partial.rename(PRIVATE/(group+'_incomplete_'+str(time.time_ns())+'.tar.gz.part'))
    script=(HERE/'stream_archive.py').read_text(encoding='utf-8').replace('GROUP_VALUE',repr(group))
    h=hashlib.sha256();size=0;last=0;start=time.time()
    with partial.open('wb') as out,log.open('wb') as err:
        process=subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','DevContainer',PYTHON+' -'],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
        process.stdin.write(script.encode('utf-8'));process.stdin.close()
        while True:
            b=process.stdout.read(8*1024*1024)
            if not b:break
            out.write(b);h.update(b);size+=len(b)
            if time.time()-last>5:
                save(PRIVATE/(group+'_transfer.json'),dict(group=group,compressed_bytes=size,elapsed_seconds=time.time()-start,finished=False));last=time.time()
        out.flush()
        code=process.wait()
    assert code==0,'SSH stream failed; see private log'
    lines=log.read_text(encoding='utf-8').split('\n')
    receipt=json.loads(next(x[len('RECEIPT_JSON '):] for x in reversed(lines) if x.startswith('RECEIPT_JSON ')))
    assert receipt['stream_sha256']==h.hexdigest() and receipt['compressed_bytes']==size
    partial.replace(path);save(PRIVATE/(group+'_stream_receipt.json'),receipt)
    save(PRIVATE/(group+'_transfer.json'),dict(group=group,compressed_bytes=size,elapsed_seconds=time.time()-start,finished=True,stream_sha_match=True))
    print(json.dumps(dict(group=group,downloaded=True,compressed_bytes=size,source_files=receipt['source_files'],source_bytes=receipt['source_bytes'],stream_sha256=h.hexdigest(),changed_files=len(receipt['changed_source_files']))),flush=True)


def verify(group):
    archive=PRIVATE/(group+'.tar.gz');receipt=json.loads((PRIVATE/(group+'_stream_receipt.json')).read_text(encoding='utf-8'))
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==receipt['stream_sha256']
    dest=(PRIVATE/'verified'/group).resolve()
    assert dest.is_relative_to(PRIVATE.resolve())
    if os.name=='nt':dest=Path('\\\\?\\'+str(dest))
    dest.mkdir(parents=True,exist_ok=True)
    found={};mapped={};checked_parents=set();resolved_dest=dest.resolve();start=time.time()
    scheduled=set();component_cases={};pending=set();last_progress=[0]
    def persist(name,size_expected,parts,src):
        target=dest.joinpath(*parts);parent_key=tuple(parts[:-1])
        if parent_key not in checked_parents:
            target.parent.mkdir(parents=True,exist_ok=True)
            assert target.parent.resolve().is_relative_to(resolved_dest)
            checked_parents.add(parent_key)
        assert not target.is_symlink()
        digest=hashlib.sha256();size=0
        with target.open('wb') as out:
            for b in iter(lambda:src.read(8*1024*1024),b''):out.write(b);digest.update(b);size+=len(b)
        assert size==size_expected
        disk=hashlib.sha256()
        with target.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):disk.update(b)
        assert disk.hexdigest()==digest.hexdigest()
        return name,dict(bytes=size,sha256=digest.hexdigest())
    def remember(value):
        name,entry=value;found[name]=entry
        if len(found)-last_progress[0]>=500:
            save(PRIVATE/(group+'_verification_progress.json'),dict(extracted=len(found),elapsed_seconds=time.time()-start))
            last_progress[0]=len(found)
    with ThreadPoolExecutor(max_workers=8) as pool,tarfile.open(archive,mode='r|gz') as tar:
        for member in tar:
            name=PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and '..' not in name.parts and member.name not in scheduled
            scheduled.add(member.name)
            parts=[]
            reserved={'CON','PRN','AUX','NUL'}|{'COM'+str(i) for i in range(1,10)}|{'LPT'+str(i) for i in range(1,10)}
            for part in name.parts:
                unsafe=os.name=='nt' and (any(ord(c)<32 or c in '<>:"\\|?*' for c in part) or part.endswith((' ','.')) or part.split('.')[0].upper() in reserved or part.startswith('_WIN_'))
                candidate='_WIN_'+hashlib.sha256(part.encode('utf-8')).hexdigest() if unsafe else part
                case_key=(tuple(parts),candidate.casefold())
                if os.name=='nt' and case_key in component_cases and component_cases[case_key]!=candidate:
                    candidate='_WIN_'+hashlib.sha256(part.encode('utf-8')).hexdigest()
                component_cases.setdefault(case_key,candidate)
                parts.append(candidate)
            if tuple(parts)!=name.parts:mapped[member.name]='/'.join(parts)
            with tar.extractfile(member) as src:
                if member.size>2*1024*1024:
                    remember(persist(member.name,member.size,parts,src))
                else:
                    data=src.read();assert len(data)==member.size
                    pending.add(pool.submit(persist,member.name,member.size,parts,io.BytesIO(data)))
            if len(pending)>=64:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                for f in done:remember(f.result())
        for f in wait(pending,return_when=ALL_COMPLETED)[0]:remember(f.result())
    # Read the gzip stream through EOF, including its CRC/trailer (tar may stop at end blocks).
    import gzip
    with gzip.open(archive,'rb') as f:
        for _ in iter(lambda:f.read(8*1024*1024),b''):pass
    manifest=json.loads((dest/'recovery_manifest.json').read_text(encoding='utf-8'))
    assert set(found)==set(manifest['files'])|{'recovery_manifest.json'}
    assert len(found)==receipt['tar_members']
    assert all(found[n]['bytes']==e['bytes'] and found[n]['sha256']==e['sha256'] for n,e in manifest['files'].items())
    result=dict(group=group,passed=True,archive_sha256=h.hexdigest(),compressed_bytes=archive.stat().st_size,
        source_files=len(manifest['files']),source_bytes=sum(e['bytes'] for e in manifest['files'].values()),
        gzip_crc=True,tar_exact_members=True,per_member_sha=True,safe_full_extract_sha=True,
        source_changed_files=manifest['changed_source_files'],excluded=manifest['excluded'],recorded_links=manifest['symbolic_links_recorded_not_followed'],
        windows_encoded_paths=len(mapped),verified_unix=time.time(),verification_seconds=time.time()-start)
    if mapped:save(PRIVATE/(group+'_windows_path_map.json'),mapped)
    save(PRIVATE/(group+'_verified.json'),result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['excluded','recorded_links']}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['archive','verify']);ap.add_argument('group',choices=['current','history','source']);a=ap.parse_args()
    (archive if a.action=='archive' else verify)(a.group)
