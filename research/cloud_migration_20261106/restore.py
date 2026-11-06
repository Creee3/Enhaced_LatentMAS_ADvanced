"""Linux-only exact-membership, safe extraction and independent readback."""
import hashlib,json,os,sys,tarfile,time
from pathlib import Path,PurePosixPath

ROOT=Path('/home/ubuntu/cloud_migration_20261106')
STAGE=ROOT/'restored'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def safe(name):
    p=PurePosixPath(name)
    assert not p.is_absolute() and '..' not in p.parts and str(p)==name
    return p
def unpack(archive,expected,dest,marker):
    if marker.exists():
        r=load(marker);assert r['passed'];return r
    seen=set()
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            name=member.name;safe(name)
            assert member.isfile() and name not in seen
            seen.add(name)
            if name=='recovery_manifest.json':
                assert json.loads(tar.extractfile(member).read())['files']==expected
                continue
            assert name in expected,'Unexpected member: '+repr(name)
            entry=expected[name];assert member.size==entry['bytes']
            target=dest/name
            assert target.resolve().is_relative_to(dest.resolve())
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                assert target.is_file() and not target.is_symlink() and sha(target)==entry['sha256']
                h=hashlib.sha256()
                # Consume this already-restored member from the same gzip stream.
                source=tar.extractfile(member)
                for b in iter(lambda:source.read(8*1024*1024),b''):h.update(b)
            else:
                h=hashlib.sha256();source=tar.extractfile(member)
                with target.open('xb') as out:
                    for b in iter(lambda:source.read(8*1024*1024),b''):
                        out.write(b);h.update(b)
                target.chmod(entry['mode']&0o777)
                os.utime(target,ns=(entry['mtime_ns'],entry['mtime_ns']))
            assert h.hexdigest()==entry['sha256'],'Member SHA mismatch: '+repr(name)
    assert seen==set(expected)|{'recovery_manifest.json'}
    # tarfile may stop at the tar terminator; separately consume gzip to EOF for CRC.
    import gzip
    with gzip.open(archive,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):pass
    for name,entry in expected.items():
        p=dest/name
        assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
    result={'passed':True,'files':len(expected),'bytes':sum(e['bytes'] for e in expected.values()),
            'exact_members':True,'gzip_crc':True,'member_sha':True,'disk_readback_sha':True,'safe_extraction':True}
    marker.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result
def main():
    os.umask(0o027);STAGE.mkdir(exist_ok=True)
    receipts=load(ROOT/'verification/transfer_verified.json')
    for name,e in receipts.items():assert sha(ROOT/'archives'/name)==e['sha256']
    metadata=ROOT/'metadata';metadata.mkdir(exist_ok=True)
    with tarfile.open(ROOT/'archives/metadata.tar.gz','r:gz') as tar:
        for m in tar:
            safe(m.name);assert m.isfile()
            dest=metadata/m.name;assert dest.resolve().is_relative_to(metadata.resolve())
            dest.parent.mkdir(parents=True,exist_ok=True)
            data=tar.extractfile(m).read()
            if dest.exists():assert dest.read_bytes()==data
            else:dest.write_bytes(data)
    sets=set();results={}
    groups=sys.argv[1:] or ['current','source','history']
    assert set(groups)<= {'current','source','history'}
    for group in groups:
        expected=load(metadata/(group+'_recovery_manifest.json'))['files']
        assert not sets&set(expected);sets.update(expected)
        results[group]=unpack(ROOT/'archives'/(group+'.tar.gz'),expected,STAGE,ROOT/'verification'/(group+'_restored.json'))
        print(json.dumps({'group':group,**results[group]}),flush=True)
    if set(groups)!= {'current','source','history'}:
        print(json.dumps({'partial_restoration_passed':True,'files':len(sets)}),flush=True)
        return
    assert len(sets)==92648
    final=Path('/home/ubuntu/ICML2027')
    if not final.exists():(STAGE/'ICML2027').rename(final)
    assert final.is_dir()
    result={'passed':True,'groups':results,'files':len(sets),'root':str(final),'experiments_resumed':False}
    (ROOT/'verification/RESTORE_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'restoration_passed':True,'files':len(sets),'root':str(final)}),flush=True)
if __name__=='__main__':main()
