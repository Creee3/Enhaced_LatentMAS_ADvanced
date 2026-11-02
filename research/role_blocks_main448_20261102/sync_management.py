"""Sync non-scientific management records without changing the run lock."""
import base64,hashlib,json,subprocess
from deploy import PRIVATE,HERE,OLD,ROOT,call

def main():
    active=json.loads((PRIVATE/'cloud_active.json').read_text(encoding='utf-8'))
    reference=dict(frozen_parent_main_run_label='run',label_is_extraction_directory_only=True,
        resolved_parent_main_run=OLD.parents[2].name,remote_parent=ROOT+'/evidence/'+OLD.parents[2].name,
        local_extracted_parent=str(OLD),source_lock_sha256=hashlib.sha256((OLD/'run_lock.json').read_bytes()).hexdigest(),
        parent_archive_sha256='11393167cf8d3789d7f6dd492027582800b223da2ed8f91fd9fa1ac4ae067eea',
        original_answer_snapshot_unchanged=True,scientific_lock_changed=False)
    files={n:(HERE/n).read_bytes() for n in ['README.md','task_plan.md','progress.md','findings.md','worker_compatibility.json']}
    files['source_reference_clarification.json']=(json.dumps(reference,ensure_ascii=False,indent=2)+'\n').encode()
    files['deployment_receipt.json']=(json.dumps(dict(run=active['run'],manager_pid=active['manager_pid'],automation_id='gpqa-448',max_workers=6,
        local_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),source_lock_sha256=active['source_lock_sha256']),indent=2)+'\n').encode()
    destination=PRIVATE/'management';destination.mkdir(exist_ok=True)
    for n,b in files.items():(destination/n).write_bytes(b)
    payload={n:base64.b64encode(b).decode() for n,b in files.items()}
    expected={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    script='''import json,hashlib,base64
from pathlib import Path
run=Path(RUN_VALUE);dest=run/'management';dest.mkdir(exist_ok=True)
assert hashlib.sha256((run/'run_lock.json').read_bytes()).hexdigest()==LOCK_VALUE
for n,b in PAYLOAD_VALUE.items():
 p=dest/n;p.write_bytes(base64.b64decode(b));assert hashlib.sha256(p.read_bytes()).hexdigest()==DIGESTS_VALUE[n]
print(json.dumps(dict(files=len(PAYLOAD_VALUE),all_sha_match=True,scientific_lock_unchanged=True)))
'''.replace('RUN_VALUE',repr(active['run'])).replace('LOCK_VALUE',repr(active['source_lock_sha256'])).replace('PAYLOAD_VALUE',repr(payload)).replace('DIGESTS_VALUE',repr(expected))
    raw=call(script,90);(PRIVATE/'management_sync_verified.json').write_bytes(raw)
    active['automation_id']='gpqa-448';active['management']=str(destination)
    (PRIVATE/'cloud_active.json').write_text(json.dumps(active,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(raw.decode())

if __name__=='__main__':main()
