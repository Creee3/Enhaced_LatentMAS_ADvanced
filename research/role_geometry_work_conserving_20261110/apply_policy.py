"""Explicitly authorized queue-only removal of scheduling batch barriers."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'role_geometry_run_20261110'))
from remote import call, active, PRIVATE

SCRIPT = r'''
import sys,json,time,copy,hashlib
from pathlib import Path
r=Path(RUN);sys.path.insert(0,str(r))
from common import load,save,verify,locked,identity,choose
source=verify();folder=r/'management';proof=folder/'work_conserving_applied.json'
assert not proof.exists(), 'policy already applied; do not repeat'
assert (r/'engineering_release.json').exists()
assert not (r/'STOP').exists() and not (r/'MANAGEMENT_HOLD').exists()
assert not (folder/'failure.json').exists()
workers=load(folder/'workers.json');assert len(workers)==8
before_id=[identity(w['pid']) for w in workers]
assert all(a==w['identity'] for a,w in zip(before_id,workers))
with locked():
 q=load(r/'queue.json');before=copy.deepcopy(q)
 assert len(q['tasks'])==120 and not any(t['state']=='held' for t in q['tasks'])
 assert all(t['batch']==t['index']//30 for t in q['tasks'])
 save(folder/'queue_before_work_conserving.json',before)
 for t in q['tasks']:
  t['reporting_batch']=t['batch'];t['batch']=0
 q['scheduling_policy']='work_conserving_index_order_v1'
 # Exact field-level invariant: no key, ownership, method or result is changed.
 restored=copy.deepcopy(q);restored.pop('scheduling_policy')
 for t in restored['tasks']:t['batch']=t.pop('reporting_batch')
 assert restored==before
 hypothetical=copy.deepcopy(q)
 for t in hypothetical['tasks']:
  if t['index']<30 and t['state']=='pending':t['state']='running'
 remaining=[t for t in hypothetical['tasks'] if t['state']=='pending' and t['methods']]
 if remaining:assert choose(hypothetical,'formal')['index']==remaining[0]['index']
 save(r/'queue.json',q)
 result=dict(passed=True,unix=time.time(),policy=q['scheduling_policy'],original_reporting_groups=[list(range(i,i+30)) for i in range(0,120,30)],source=source,workers=before_id,keys_before=sum(len(t['committed_methods']) for t in before['tasks']),only_task_changes=['batch','reporting_batch'],model_reloads=0,forwards_by_policy_change=0)
 save(proof,result)
print(json.dumps(dict(passed=True,unix=result['unix'],keys_before=result['keys_before'],workers=len(workers),source_unchanged=True,model_reloads=0)))
'''

def test():
    # Exercise the actual frozen selector with first-batch stragglers present.
    path=Path(__file__).resolve().parent.parent/'role_geometry_run_20261110/common.py'
    import ast
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='choose')
    ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    q=dict(engineering_indices=[0],tasks=[dict(index=i,batch=i//30,state='running' if i<30 else 'pending',methods=['R']) for i in range(120)])
    assert ns['choose'](q,'formal') is None
    for t in q['tasks']:t['reporting_batch']=t['batch'];t['batch']=0
    assert ns['choose'](q,'formal')['index']==30
    q['tasks'][30]['state']='complete';q['tasks'][30]['methods']=[]
    assert ns['choose'](q,'formal')['index']==31
    assert ns['choose'](q,'engineering') is None
    assert [t['reporting_batch'] for t in q['tasks']]==[i//30 for i in range(120)]
    print('5 scheduling checks passed; no cloud calls')

if __name__=='__main__':
    if '--apply' not in sys.argv:test()
    else:
        raw=call(SCRIPT.replace('RUN',repr(active()['run'])),90)
        (PRIVATE/'work_conserving_applied.json').write_bytes(raw)
        print(raw.decode())
