"""Explicit post-recovery audit/archive/frozen scoring; zero inference."""
import base64,hashlib,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'research/rolewise_TRG_20261108'),str(REPO/'research/TRG225_a100_20261106')]
from collect import collect,PRIVATE
import collect as archive_module
from capture_current import call

def main():
    run=json.loads((PRIVATE/'cloud_active.json').read_text())['run']
    b=Path(__file__).with_name('audit_recovered.py').read_bytes()
    s='from pathlib import Path\nimport base64,hashlib,json\nr=Path('+repr(run)+')\np=r/"management/audit_recovered.py"\nb=base64.b64decode('+repr(base64.b64encode(b).decode())+')\nassert not p.exists() or p.read_bytes()==b\np.write_bytes(b)\nprint(json.dumps(dict(sha256=hashlib.sha256(b).hexdigest())))'
    call(s)
    original=archive_module.call
    def adapted(script,*args,**kwargs):
        if "str(r/'audit.py'),phase" in script:
            assert script.count("str(r/'audit.py'),phase")==1
            script=script.replace("str(r/'audit.py'),phase","str(r/'management/audit_recovered.py'),str(r)")
        return original(script,*args,**kwargs)
    archive_module.call=adapted
    if not (PRIVATE/'formal_archive_verified.json').exists():collect('formal')
    s='from pathlib import Path\nimport subprocess,sys\nr=Path('+repr(run)+')\np=subprocess.run([sys.executable,str(r/"score.py")],cwd=r,capture_output=True,text=True)\nassert p.returncode==0,p.stderr\nprint(p.stdout)'
    if not (PRIVATE/'final_summary.json').exists():print(call(s,600).decode())
    if not (PRIVATE/'formal_final_archive_verified.json').exists():collect('formal',final=True)
    a=json.loads((PRIVATE/'cloud_active.json').read_text());a.update(status='COMPLETED_ARCHIVED_SCORED',complete=True,target_keys=1728,historical_failures_preserved=True)
    (PRIVATE/'cloud_active.json').write_text(json.dumps(a,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=True,keys=1728)))

if __name__=='__main__':main()
