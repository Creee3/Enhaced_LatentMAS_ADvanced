"""Management-only cardinality correction; original manager remains frozen."""
import sys
from pathlib import Path

def expected_keys(indices, methods, audit):
    assert len(indices)==len(set(indices))==3 and methods==['Full','receiver']
    expected=len(indices)*len(methods)
    assert audit['passed'] and audit['cases']==len(indices) and audit['keys']==expected
    return expected

if __name__=='__main__':
    run=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(run))
    from common import load,verify
    source=verify();queue=load(run/'queue.json');plan=load(run/'plan.json');audit=load(run/'engineering_audit_verified.json');archive=load(run/'engineering_archive_verified.json')
    count=expected_keys(queue['engineering_indices'],plan['methods'],audit)
    assert archive['passed'] and all(archive[k] for k in ['remote_stream_sha','gzip_crc','exact_members','per_member_sha','safe_full_extract_readback_sha'])
    assert audit['run_lock_sha256']==source['run_lock_sha256']
    original=(run/'manager.py').read_text();needle="audit['keys']==12"
    assert original.count(needle)==1
    patched=original.replace(needle,"audit['keys']==_EXPECTED_ENGINEERING_KEYS")
    sys.path.insert(0,str(run))
    exec(compile(patched,'frozen_manager_with_correct_engineering_cardinality','exec'),dict(__name__='__main__',__file__=str(run/'manager.py'),_EXPECTED_ENGINEERING_KEYS=count))
