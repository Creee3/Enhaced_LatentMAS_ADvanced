"""UTF8 current management copies; never rewrite any locked remote file."""
import base64
import hashlib
import json
import subprocess
from deploy import PRIVATE,HERE,call,encoded


def main():
    active=json.loads((PRIVATE/'cloud_active.json').read_text(encoding='utf-8'))
    files={n:(HERE/n).read_bytes() for n in ['README.md','protocol.md','task_plan.md','progress.md','findings.md','worker_compatibility.json']}
    files['deployment_receipt.json']=encoded(dict(run=active['run'],manager_pid=active.get('manager_pid'),
        status=active['status'],old_run_stays_paused=True,automation_id=active.get('automation_id'),
        local_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        source_lock_sha256=active['source_lock_sha256']))
    for name in ['launch_receipt.json','launch_preflight.json','monitor_prompt.txt']:
        if (PRIVATE/name).exists():files[name]=(PRIVATE/name).read_bytes()
    destination=PRIVATE/'management';destination.mkdir(exist_ok=True)
    for n,b in files.items():(destination/n).write_bytes(b)
    payload={n:base64.b64encode(b).decode() for n,b in files.items()}
    expected={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    raw=call('''import json,hashlib,base64
from pathlib import Path
run=Path(RUN);assert hashlib.sha256((run/'run_lock.json').read_bytes()).hexdigest()==LOCK
dest=run/'management';dest.mkdir(exist_ok=True)
for n,b in PAYLOAD.items():
 p=dest/n;p.write_bytes(base64.b64decode(b));assert hashlib.sha256(p.read_bytes()).hexdigest()==DIGESTS[n]
print(json.dumps(dict(files=len(PAYLOAD),all_sha_match=True,scientific_lock_unchanged=True)))
'''.replace('RUN',repr(active['run'])).replace('LOCK',repr(active['source_lock_sha256'])).replace('PAYLOAD',repr(payload)).replace('DIGESTS',repr(expected)),90)
    (PRIVATE/'management_sync_verified.json').write_bytes(raw)
    print(raw.decode('utf-8'))


if __name__=='__main__':main()
