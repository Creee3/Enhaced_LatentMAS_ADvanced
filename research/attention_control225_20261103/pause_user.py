"""Explicit human pause: stop only exact recorded own processes, no CUDA probes."""
import json
from deploy import PRIVATE,HERE,call,encoded


def main():
    active=json.loads((PRIVATE/'cloud_active.json').read_text(encoding='utf-8'))
    raw=call('''import sys,json,os,time,signal
from pathlib import Path
r=Path(RUN);sys.path.insert(0,str(r))
from cloud_common import load,save,identity
assert os.getuid()==1033
roots=[r,Path(SUPERSEDED),Path(PARENT)]
stopped=[];observations=[]
def same(actual,expected):
 return actual and all(actual[k]==expected[k] for k in ['uid','start_ticks','argv','CUDA_VISIBLE_DEVICES'])
def stop(expected,role):
 pid=expected['pid'];actual=identity(pid)
 if actual is None:observations.append(dict(pid=pid,role=role,already_exited=True));return
 if not same(actual,expected):
  observations.append(dict(pid=pid,role=role,identity_mismatch_not_touched=True));return
 os.kill(pid,signal.SIGTERM);stopped.append(dict(pid=pid,role=role,initial=expected,signal='SIGTERM'))
for root in roots:
 assert root.is_dir()
 (root/'STOP').write_text('真人明确暂停2026-11-03；未经真人明确恢复不得推进。',encoding='utf-8')
 for worker in load(root/'workers.json',[]):stop(worker['initial_identity'],'worker:'+root.name)
 for name in ['manager.json','bootstrap.json']:
  v=load(root/name)
  if v:stop(v['identity'],name+':'+root.name)
# The original SSH preflight was observed immediately after our timed-out launch.
# Exact UID/start/argv/CVD AND the original SSH stdin pipe must still match.
expected=dict(pid=1454644,uid=1033,start_ticks='191371140',argv=[PYTHON,'-'],CUDA_VISIBLE_DEVICES='')
actual=identity(expected['pid'])
if same(actual,expected):
 p=Path('/proc')/str(expected['pid'])/'fd/0'
 if os.readlink(p)=='pipe:[441297136]':stop(expected,'correlated_original_zero_forward_preflight')
 else:observations.append(dict(pid=expected['pid'],stdin_mismatch_not_touched=True))
else:observations.append(dict(pid=expected['pid'],preflight_exited_or_identity_changed=True))
time.sleep(1)
for entry in stopped:
 actual=identity(entry['pid'])
 if same(actual,entry['initial']):
  os.kill(entry['pid'],signal.SIGKILL);entry['signal']='SIGKILL_after_TERM'
time.sleep(1)
for entry in stopped:entry['exited']=not same(identity(entry['pid']),entry['initial'])
budget=load(r.parents[1]/'cumulative_budget.json')
names={x.name for x in roots}
own={k:v for k,v in budget['active_reservations'].items() if v.get('metadata',{}).get('experiment') in names}
proof=dict(unix=time.time(),user_requested_pause=True,no_resume_without_explicit_user=True,
 automation_status='PAUSED',stopped=stopped,observations=observations,own_reservations=len(own),
 all_signaled_processes_exited=all(x['exited'] for x in stopped),
 budget={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']},
 no_cuda_probe=True,no_model_forward=True,original_answers_unchanged=True)
save(r/'management/user_pause_20261103.json',proof)
print(json.dumps(proof,ensure_ascii=False))
'''.replace('RUN',repr(active['run'])).replace('SUPERSEDED',repr(active['superseded_staging'])).replace('PARENT',repr(active['parent_run'])).replace('PYTHON',repr('/home/liuxin/.conda/envs/kv-graph/bin/python')),60)
    proof=json.loads(raw);(PRIVATE/'user_pause_20261103.json').write_bytes(raw)
    active.update(status='PAUSED_BY_USER',no_resume_without_explicit_user=True,automation_status='PAUSED',
        own_reserved_count=proof['own_reservations'],all_signaled_processes_exited=proof['all_signaled_processes_exited'],
        last_user_pause='2026-11-03: 应该是云端gpu又出问题了，先不弄了')
    (PRIVATE/'cloud_active.json').write_bytes(encoded(active))
    ref=json.loads((HERE/'run_reference.json').read_text(encoding='utf-8'))
    ref.update(status='PAUSED_BY_USER',no_resume_without_explicit_user=True,automation_status='PAUSED')
    (HERE/'run_reference.json').write_bytes(encoded(ref))
    print(json.dumps(proof,ensure_ascii=False))


if __name__=='__main__':main()
