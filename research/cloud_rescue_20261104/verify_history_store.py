"""Full logical extraction + independent chunk readback, preserving Linux names on NTFS."""
import gzip,hashlib,json,sqlite3,tarfile,time
from pathlib import PurePosixPath
from rescue import PRIVATE,save


def main():
    archive=PRIVATE/'history.tar.gz';receipt=json.loads((PRIVATE/'history_stream_receipt.json').read_text(encoding='utf-8'))
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==receipt['stream_sha256']
    database=PRIVATE/'verified/history.sqlite'
    assert not database.exists(),'Do not overwrite a content store; inspect existing proof'
    con=sqlite3.connect(database)
    con.executescript('PRAGMA journal_mode=DELETE; PRAGMA synchronous=NORMAL; PRAGMA cache_size=-65536; CREATE TABLE files(path TEXT PRIMARY KEY, size INTEGER, sha256 TEXT); CREATE TABLE chunks(path TEXT, seq INTEGER, data BLOB, PRIMARY KEY(path,seq)); CREATE TABLE metadata(name TEXT PRIMARY KEY,value TEXT);')
    found={};start=time.time();chunk_size=8*1024*1024;pending_bytes=0
    with gzip.open(archive,'rb') as gz:
        with tarfile.open(fileobj=gz,mode='r|') as tar:
            for member in tar:
                path=PurePosixPath(member.name)
                assert member.isfile() and not path.is_absolute() and '..' not in path.parts and member.name not in found
                digest=hashlib.sha256();size=0;seq=0
                with tar.extractfile(member) as src:
                    for b in iter(lambda:src.read(chunk_size),b''):
                        digest.update(b);size+=len(b)
                        con.execute('INSERT INTO chunks VALUES (?,?,?)',(member.name,seq,b));seq+=1
                        pending_bytes+=len(b)
                        if pending_bytes>=128*1024*1024:con.commit();pending_bytes=0
                assert size==member.size
                entry=dict(bytes=size,sha256=digest.hexdigest());found[member.name]=entry
                con.execute('INSERT INTO files VALUES (?,?,?)',(member.name,size,entry['sha256']))
                if len(found)%1000==0:
                    con.commit();save(PRIVATE/'history_store_progress.json',dict(phase='extract_to_sqlite',files=len(found),seconds=time.time()-start))
        # Continue the same gzip reader to EOF: verifies CRC without a second inflation pass.
        for _ in iter(lambda:gz.read(chunk_size),b''):pass
    con.commit()
    manifest_bytes=b''.join(row[0] for row in con.execute('SELECT data FROM chunks WHERE path=? ORDER BY seq',('recovery_manifest.json',)))
    manifest=json.loads(manifest_bytes)
    assert set(found)==set(manifest['files'])|{'recovery_manifest.json'} and len(found)==receipt['tar_members']
    assert all(found[n]['bytes']==e['bytes'] and found[n]['sha256']==e['sha256'] for n,e in manifest['files'].items())
    con.execute('INSERT INTO metadata VALUES (?,?)',('recovery_manifest',manifest_bytes.decode('utf-8')));con.commit()
    verified=0
    for name,entry in found.items():
        digest=hashlib.sha256();size=0
        for (b,) in con.execute('SELECT data FROM chunks WHERE path=? ORDER BY seq',(name,)):
            digest.update(b);size+=len(b)
        assert size==entry['bytes'] and digest.hexdigest()==entry['sha256']
        verified+=1
        if verified%2000==0:save(PRIVATE/'history_store_progress.json',dict(phase='independent_sqlite_readback',files=verified,seconds=time.time()-start))
    assert con.execute('PRAGMA integrity_check').fetchall()==[('ok',)]
    con.close()
    (PRIVATE/'history_recovery_manifest.json').write_bytes(manifest_bytes)
    result=dict(group='history',passed=True,archive_sha256=h.hexdigest(),compressed_bytes=archive.stat().st_size,
        source_files=len(manifest['files']),source_bytes=sum(e['bytes'] for e in manifest['files'].values()),
        gzip_crc=True,tar_exact_members=True,per_member_sha=True,safe_full_extract_sha=True,
        extraction_format='sqlite_chunk_store',extracted_content_store='verified/history.sqlite',sqlite_integrity_check=True,
        source_changed_files=manifest['changed_source_files'],excluded=manifest['excluded'],recorded_links=manifest['symbolic_links_recorded_not_followed'],
        windows_encoded_paths=0,linux_names_preserved_in_database=True,verified_unix=time.time(),verification_seconds=time.time()-start)
    save(PRIVATE/'history_verified.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['excluded','recorded_links']}),flush=True)


if __name__=='__main__':main()
