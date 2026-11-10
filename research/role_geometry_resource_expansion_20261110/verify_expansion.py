"""Verify added model readiness and frozen live mechanics; no generation."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'role_geometry_run_20261110'))
from remote import call,active,PRIVATE

def main():
 script=r"""from pathlib import Path
import sys,json,time,subprocess
r=Path(RUN);sys.path.insert(0,str(r));from common import load,save,identity,verify,records,key
f=r/'management';ws=load(f/'workers.json');assert len(ws)==8 and {g:sum(w['gpu_index']==g for w in ws) for g in (0,1,3)}=={0:3,1:4,3:1}
old=load(f/'expansion_predecessor.json')['workers'];assert all(any(w['identity']==a['identity'] and identity(a['pid'])==a['identity'] for w in ws) for a in old)
assert not (f/'failure.json').exists()
for w in ws:
 assert identity(w['pid'])==w['identity'];out=Path(w['out']);a=load(out/'model_ready.json')
 assert a['weight_files']==15 and a['bf16_cuda'] and a['parameter_bytes']==16381470720 and a['identity_norm'] and abs(a['target_norm']-1.3758320808410645)<1e-6 and not (out/'failure.json').exists()
rs=records();assert len(rs)==len({key(row) for row in rs});q=load(r/'queue.json');assert len(rs)==sum(len(t['committed_methods']) for t in q['tasks'])
phase='formal' if (r/'engineering_release.json').exists() else 'engineering'
p=subprocess.run([sys.executable,str(r/'verify_live.py'),phase],cwd=r,capture_output=True,text=True,timeout=150);assert p.returncode==0,p.stderr[-1500:]
live=load(f/'workers_verified.json');assert live['passed'] and not live['verified_failures']
proof=dict(passed=True,unix=time.time(),science=verify(),max_workers=8,per_gpu={'0':3,'1':4,'3':1},new_workers=[dict(pid=w['pid'],identity=w['identity'],gpu=w['gpu_index'],slot=w['slot']) for w in ws if w['slot'] in (6,7)],original_six_identities_preserved=True,all_eight_ready=True,independent_frozen_live_verifier_passed=True,verified_committed_rows=sum(w.get('verified_new_rows',0) for w in live['workers']),unique_keys=len(rs),no_generation_or_settlement_by_verifier=True,phase=phase)
save(f/'resource_expansion_verified.json',proof);print(json.dumps(proof))
""".replace('RUN',repr(active()['run']))
 raw=call(script,180);(PRIVATE/'resource_expansion_verified.json').write_bytes(raw);x=json.loads(raw)
 print(json.dumps(dict(passed=x['passed'],workers=x['per_gpu'],new_pids=[w['pid'] for w in x['new_workers']],all_ready=x['all_eight_ready'],old_six_preserved=x['original_six_identities_preserved'],verified_rows=x['verified_committed_rows'],keys=x['unique_keys'],phase=x['phase'])))
if __name__=='__main__':main()
