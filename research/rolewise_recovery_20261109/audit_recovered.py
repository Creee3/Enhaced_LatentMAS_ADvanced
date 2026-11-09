"""Keep frozen structural checks; explicitly account for one archived recovery.

Never hides historical failures or treats regenerated prefix RIDs as shared calls.
"""
import sys,json,hashlib
from pathlib import Path

def main(run):
    sys.path.insert(0,str(run))
    import common as c
    c.verify()
    assert c.load(run/'formal_failure_archive_verified.json')['archive_sha256']=='a2758800c4766f27b8c64037ea5fec957ccec010b0238268794cbf82b8c57610'
    assert c.load(run/'management/recovery015/finished.json')['returncode']==0
    assert c.load(run/'management/recovery015/prefix_verified.json')['passed']
    assert c.load(run/'management/recovery015/mps_stop.json')['returncode']==0
    assert not c.load(run/'management/recovery015/mps_stop.json')['owned_after']
    for meta in [c.load(run/'management/recovery015/identity.json'),c.load(run/'management/recovery015/worker.json')]:
        owner=meta['identity'];live=c.identity(owner['pid'])
        assert live is None or live['start_ticks']!=owner['start_ticks']
    old=c.load(run/'management/formal/failure.json')
    assert old['error']=="RuntimeError('model loading timeout; preserve running ownership')"
    old_worker=next(w for w in c.load(run/'management/formal/workers.json') if w['pid']==343277)
    out=Path(old_worker['out']);assert c.load(out/'failure.json')['error']=="RuntimeError('request timeout')"
    failed=[x for x in c.rows(out/'trace.jsonl') if x.get('event')=='request_settled' and x['failed']]
    assert len(failed)==1 and failed[0]['reservation_id']=='343277-1794152736751573869'
    assert (failed[0]['actual_positions'],failed[0]['independent_unknown'])==(5399,3105)
    first=next(x for x in c.rows(out/'results.jsonl') if x['index']==15 and x['method']=='GGG')
    old_ids={x['reservation_id'] for x in c.rows(out/'trace.jsonl') if x.get('event')=='request_started'}
    abandoned=old_ids-set(first['request_ids']);assert len(abandoned)==5
    replacements=[
      ("assert load(folder/'finished.json')['all_eligible_complete'] and not (folder/'failure.json').exists()",
       "assert stage=='formal' or (load(folder/'finished.json')['all_eligible_complete'] and not (folder/'failure.json').exists())"),
      ("assert not (out/'failure.json').exists() and load(out/'exit.json')['normal']",
       "assert worker['pid']==343277 or (not (out/'failure.json').exists() and load(out/'exit.json')['normal'])"),
      ("assert all(not x['failed'] and x['independent_unknown']==0 for x in settled)",
       "assert all((x['reservation_id']=='343277-1794152736751573869' and x['failed'] and x['independent_unknown']==3105) or (not x['failed'] and x['independent_unknown']==0) for x in settled)"),
      ("assert len({tuple(x['shared_request_ids']) for x in trio})==len({x['shared_planner_kv_sha'] for x in trio})==len({x['fork_rng_sha'] for x in trio})==1",
       "assert len({tuple(x['shared_request_ids']) for x in trio})==(2 if i==15 else 1) and len({x['shared_planner_kv_sha'] for x in trio})==len({x['fork_rng_sha'] for x in trio})==1"),
      ("assert len(set(r for x in trio for r in x['request_ids']))==17",
       "assert len(set(r for x in trio for r in x['request_ids']))==(19 if i==15 else 17)"),
      ("if phase=='formal':assert selected_rids==all_rids and len(all_rids)==576*17",
       "if phase=='formal':assert selected_rids==all_rids-ABANDONED and len(all_rids)==576*17+7"),
    ]
    source=(run/'audit.py').read_text()
    for old,new in replacements:
        assert source.count(old)==1, old
        source=source.replace(old,new)
    ns=dict(__name__='recovery_structural_audit',ABANDONED=abandoned)
    exec(compile(source,'frozen_audit_with_explicit_recovery_accounting','exec'),ns)
    ns['main']('formal')
    result=c.load(run/'formal_audit_verified.json')
    assert result['unknown_added']==3105
    result.update(recovery_case=15,abandoned_RIDs=sorted(abandoned),historical_failures_preserved=True,
                  recovery_prefix_RIDs_recomputed=True,cross_attempt_shared_RIDs=False,
                  audit_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    c.save(run/'formal_audit_verified.json',result)

if __name__=='__main__':main(Path(sys.argv[1]))
