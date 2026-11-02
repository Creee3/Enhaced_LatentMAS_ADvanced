"""Synchronize derived management records outside the archived immutable run."""
import base64,hashlib,json,subprocess
from capture import PRIVATE,HERE
from remote import ROOT,call

def main():
    active=json.loads((PRIVATE/'active.json').read_text(encoding='utf-8'))
    report=json.loads((HERE/'engineering_report.json').read_text(encoding='utf-8'))
    assert report['complete'] and report['answers']==18
    files={n:(HERE/n).read_bytes() for n in ['task_plan.md','findings.md','progress.md','README.md','engineering_report.md','engineering_report.json','smoke_config.json']}
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip()
    status=dict(run=active['run'],status='complete_archived_engineering_checked',worker=None,
        completed=18,target=18,requests=54,actual_positions=report['actual_positions'],new_unknown=0,
        archive=report['archive'],local_commit=revision,pushed=False,monitor_created=False)
    files['completion.json']=(json.dumps(status,ensure_ascii=False,indent=2)+'\n').encode()
    local=PRIVATE/'management';local.mkdir(exist_ok=True)
    for n,b in files.items():(local/n).write_bytes(b)
    payload={n:base64.b64encode(b).decode() for n,b in files.items()}
    digests={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    script='''import base64,json,hashlib,sys
from pathlib import Path
run=Path(RUN_VALUE);sys.path.insert(0,str(run))
from cloud_common import identity
assert identity(PID_VALUE) is None
assert json.loads(Path(OUT_VALUE).joinpath('exit.json').read_text())['completed']
dest=run.parents[1]/'management'/run.name;dest.mkdir(parents=True,exist_ok=True)
for n,b in PAYLOAD_VALUE.items():
 p=dest/n;p.write_bytes(base64.b64decode(b));assert hashlib.sha256(p.read_bytes()).hexdigest()==DIGESTS_VALUE[n]
print(json.dumps(dict(management=str(dest),files=len(PAYLOAD_VALUE),all_sha_match=True)))
'''.replace('RUN_VALUE',repr(active['run'])).replace('PID_VALUE',str(active['identity']['pid'])).replace('OUT_VALUE',repr(active['attempt'])).replace('PAYLOAD_VALUE',repr(payload)).replace('DIGESTS_VALUE',repr(digests))
    result=call(script,90);(PRIVATE/'management_sync_verification.json').write_bytes(result)
    (PRIVATE/'cloud_active.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(result.decode())

if __name__=='__main__':main()
