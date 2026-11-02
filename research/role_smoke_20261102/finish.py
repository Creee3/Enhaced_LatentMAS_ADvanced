"""Archive a stopped smoke and verify all members, then summarize engineering checks."""
import datetime,gzip,hashlib,io,json,math,tarfile
from pathlib import Path,PurePosixPath
from remote import ROOT,call
from capture import PRIVATE,HERE

def main():
    active=json.loads((PRIVATE/'active.json').read_text(encoding='utf-8'))
    script='''import io,json,sys,tarfile,hashlib
from pathlib import Path
run=Path(RUN_VALUE);out=Path(OUT_VALUE);sys.path.insert(0,str(run))
from cloud_common import identity
assert identity(PID_VALUE) is None, 'worker still live; no final archive'
assert (out/'exit.json').exists()
base=run.parents[1]/'evidence/gpqa_repair_8096_20261028T161254Z/bundle'
files={}
for p in sorted(run.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts:
  assert not p.is_symlink();files['run/'+p.relative_to(run).as_posix()]=p
lock=json.loads((base/'source_lock.json').read_text())
for n in list(lock['files'])+['source_lock.json']:
 p=base/n;assert p.is_file() and not p.is_symlink();files['referenced_bundle37/'+n]=p
manifest={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in files.items()}
raw=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\\n').encode()
class HashWriter:
 def __init__(self):self.digest=hashlib.sha256();self.size=0
 def write(self,b):self.digest.update(b);self.size+=len(b);return sys.stdout.buffer.write(b)
 def flush(self):sys.stdout.buffer.flush()
stream=HashWriter()
with tarfile.open(fileobj=stream,mode='w|gz') as tar:
 for n,p in files.items():tar.add(p,arcname=n,recursive=False)
 info=tarfile.TarInfo('archive_manifest.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
stream.flush()
(run.parent/(run.name+'_archive_stream.json')).write_text(json.dumps(dict(sha256=stream.digest.hexdigest(),bytes=stream.size))+'\\n')
'''.replace('RUN_VALUE',repr(active['run'])).replace('OUT_VALUE',repr(active['attempt'])).replace('PID_VALUE',str(active['identity']['pid']))
    raw=call(script,180)
    digest=hashlib.sha256(raw).hexdigest()
    remote_proof=json.loads(call('from pathlib import Path\nprint(Path('+repr(str(Path(active['run']).parent/(Path(active['run']).name+'_archive_stream.json')).replace('\\','/'))+').read_text())'))
    assert digest==remote_proof['sha256'] and len(raw)==remote_proof['bytes']
    archive=PRIVATE/(Path(active['run']).name+'_full_'+digest[:12]+'.tar.gz')
    archive.write_bytes(raw)
    # Reading to EOF explicitly checks gzip CRC, independently of tar's EOF.
    plain=gzip.decompress(raw)
    extract=PRIVATE/('verified_extracted_'+digest[:12]);extract.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(plain),mode='r:') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        assert len(names)==len(set(names))
        manifest=json.load(tar.extractfile('archive_manifest.json'))
        assert set(names)==set(manifest)|{'archive_manifest.json'}
        for m in members:
            name=PurePosixPath(m.name)
            assert m.isfile() and not name.is_absolute() and '..' not in name.parts
            b=tar.extractfile(m).read();target=extract.joinpath(*name.parts)
            assert target.resolve().is_relative_to(extract.resolve())
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
            if m.name in manifest:
                entry=manifest[m.name]
                assert len(b)==entry['bytes'] and hashlib.sha256(b).hexdigest()==entry['sha256']
                assert hashlib.sha256(target.read_bytes()).hexdigest()==entry['sha256']
    proof=dict(archive_sha256=hashlib.sha256(raw).hexdigest(),archive_bytes=len(raw),members=len(names),
        stream_sha=True,gzip_crc=True,tar_exact_set=True,all_member_sha=True,safe_full_extract_sha=True)
    (PRIVATE/'archive_verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    out=extract/'run'/'attempts'/Path(active['attempt']).name
    exit=json.loads((out/'exit.json').read_text());plan=json.loads((extract/'run/plan.json').read_text())
    rows=[json.loads(s) for s in (out/'results.jsonl').read_text(encoding='utf-8').split('\n') if s] if (out/'results.jsonl').exists() else []
    expected={(x['question_id'],r,m) for x in plan['questions'] for r in [1,2,3] for m in plan['arms']}
    keys={(x['question_id'],x['repeat'],x['method']) for x in rows}
    trace=[json.loads(s) for s in (out/'trace.jsonl').read_text(encoding='utf-8').split('\n') if s] if (out/'trace.jsonl').exists() else []
    starts={x['reservation_id']:x for x in trace if x['event']=='request_started'}
    settles=[x for x in trace if x['event']=='request_settled']
    assert len(settles)==len({x['reservation_id'] for x in settles})
    assert set(starts)=={x['reservation_id'] for x in settles}, 'unsettled own RID'
    repeats=[];timings=[]
    for q in plan['questions']:
        for method in plan['arms']:
            group=[r for r in rows if r['question_id']==q['question_id'] and r['method']==method]
            repeats.append(dict(question_id=q['question_id'],method=method,completed=len(group),
                identical_position_selection=len({r['selected_position_sha'] for r in group})==1 if group else None,
                identical_generated_tokens=len({r['token_sha'] for r in group})==1 if group else None))
            if group:
                ts=[r['timing'] for r in group]
                fields=['required_shared_judger_probe_seconds','selector_and_gather_seconds','judger_seconds','reconstructed_standalone_seconds']
                assert all(math.isfinite(t[f]) and t[f]>=0 for t in ts for f in fields)
                timings.append(dict(question_id=q['question_id'],method=method,**{f:dict(mean=sum(t[f] for t in ts)/len(ts),min=min(t[f] for t in ts),max=max(t[f] for t in ts)) for f in fields},
                    generated_tokens=[r['generated_tokens'] for r in group]))
    complete=bool(exit['completed'] and len(rows)==len(keys)==18 and keys==expected)
    if complete:
        assert len(starts)==54
        for row in rows:
            assert row['positions_shape']==[36,8,130] and row['retained_kv_bytes']==19169280
            assert row['source_k_equal_measured'] and row['source_v_equal_measured'] and row['protected_latent_sink']
            assert row['timing']['cuda_sync_both_boundaries'] and row['parser_sha']=='92163f3a5389c4a7e25868ff931fc919703eb1e500ef74dc03b4810ab6f0225d'
    report=dict(complete=complete,answers=len(rows),unique=len(keys),missing=len(expected-keys),archive=proof,
        requests=len(starts),actual_positions=sum(x['actual_positions'] for x in settles),
        new_unknown=sum(x['independent_unknown'] for x in settles),repeats=repeats,timings=timings,
        no_effectiveness_claim=True,reconstructed_times_not_independent_e2e=True,exit=exit)
    (PRIVATE/'engineering_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'engineering_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=complete,answers=len(rows),missing=report['missing'],requests=len(starts),actual_positions=report['actual_positions'],new_unknown=report['new_unknown'],archive=proof,repeats=repeats),ensure_ascii=False))

if __name__=='__main__':main()
