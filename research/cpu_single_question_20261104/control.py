"""Explicit one-shot CPU launch, small status and private artifact collection."""
import argparse,base64,datetime,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'role_smoke_20261102'))
from remote import ROOT,PYTHON,call
PRIVATE=REPO.parent/'advanced_MOC/experiments/latentmas_cpu_single_question_20261104'
G=REPO.parent/'advanced_MOC/experiments/latentmas_attention_control225_20261103'


def enc(v):return (json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8')


def launch():
    assert not (PRIVATE/'active.json').exists(),'Inspect existing CPU pilot; do not launch twice'
    PRIVATE.mkdir(parents=True,exist_ok=True)
    assert json.loads((G/'cloud_active.json').read_text(encoding='utf-8'))['status']=='PAUSED_BY_USER'
    item=json.loads((G/'questions.json').read_text(encoding='utf-8'))[0]
    files={'worker.py':(HERE/'worker.py').read_bytes(),'question.json':enc(item),
           'author_parser.py':(REPO/'snapshot/run/author_parser.py').read_bytes()}
    lock={n.replace('.','_')+'_sha256':hashlib.sha256(b).hexdigest() for n,b in files.items()}
    lock.update(worker_sha256=hashlib.sha256(files['worker.py']).hexdigest(),question_sha256=hashlib.sha256(files['question.json']).hexdigest(),
                protocol='CPU-only BF16,16threads,batch1,single model,nonthinking,direct_answer,cap64,seed42',
                user_authorization='2026-11-04 先测一道题吧',old_experiments_stay_paused=True)
    files['run_lock.json']=enc(lock)
    name='cpu_single_question_bf16_direct64_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run=ROOT+'/evidence/'+name
    payload={n:base64.b64encode(b).decode() for n,b in files.items()}
    raw=call('''import os,json,base64,subprocess,time
from pathlib import Path
r=Path(RUN);assert not r.exists();r.mkdir();(r/'raw').mkdir()
for n,b in PAYLOAD.items():(r/n).write_bytes(base64.b64decode(b))
root=r.parents[1];budget=json.loads((root/'cumulative_budget.json').read_text())
assert sum(budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens'])+32832<=115000000
with (r/'worker.log').open('ab') as log:
 p=subprocess.Popen([PYTHON,str(r/'worker.py')],cwd=r,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='16',MKL_NUM_THREADS='16',OPENBLAS_NUM_THREADS='16',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(100):
 proc=Path('/proc')/str(p.pid)
 argv=[x.decode() for x in (proc/'cmdline').read_bytes().split(bytes([0])) if x]
 if argv==[PYTHON,str(r/'worker.py')]:break
 time.sleep(.02)
assert argv==[PYTHON,str(r/'worker.py')]
fields=(proc/'stat').read_text().rsplit(')',1)[1].split()
identity=dict(pid=p.pid,uid=proc.stat().st_uid,start_ticks=fields[19],argv=argv,CUDA_VISIBLE_DEVICES='')
receipt=dict(run=str(r),identity=identity,budget_before={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']},old_experiments_stay_paused=True)
(r/'launch_receipt.json').write_text(json.dumps(receipt))
print(json.dumps(receipt))
'''.replace('RUN',repr(run)).replace('PAYLOAD',repr(payload)).replace('[PYTHON,','['+repr(PYTHON)+','),60)
    value=json.loads(raw)
    for n,b in files.items():(PRIVATE/n).write_bytes(b)
    (PRIVATE/'active.json').write_bytes(raw)
    print(json.dumps(value))


def status(collect=False):
    active=json.loads((PRIVATE/'active.json').read_text(encoding='utf-8'))
    raw=call('''import os,json,base64,time,hashlib
from pathlib import Path
r=Path(RUN);expected=IDENTITY;p=Path('/proc')/str(expected['pid'])
alive=False;rss=0;state=None
try:
 fields=(p/'stat').read_text().rsplit(')',1)[1].split()
 argv=[x.decode() for x in (p/'cmdline').read_bytes().split(bytes([0])) if x]
 alive=fields[0]!='Z' and p.stat().st_uid==expected['uid'] and fields[19]==expected['start_ticks'] and argv==expected['argv']
 state=fields[0];rss=int(fields[21])*os.sysconf('SC_PAGE_SIZE')/2**30
except (FileNotFoundError,ProcessLookupError):pass
def load(n):return json.loads((r/n).read_text()) if (r/n).exists() else None
budget=json.loads((r.parents[1]/'cumulative_budget.json').read_text())
own={k:v for k,v in budget['active_reservations'].items() if v.get('metadata',{}).get('experiment')==r.name}
trace=[json.loads(x) for x in (r/'trace.jsonl').read_bytes().split(b'\\n') if x] if (r/'trace.jsonl').exists() else []
result=load('result.json');failure=load('failure.json')
summary=dict(alive=alive,process_state=state,rss_gib=round(rss,2),state=load('state.json'),verified_weight_files=sum(x['event']=='model_file_verified' for x in trace),last_progress=next((x for x in reversed(trace) if x['event']=='native_forward_progress'),None),result={k:v for k,v in result.items() if k!='request_receipt'} if result else None,failure=failure,own_reserved_count=len(own),budget={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']})
if COLLECT:
 assert not alive and (result or failure) and not own
 files={p.relative_to(r).as_posix():base64.b64encode(p.read_bytes()).decode() for p in r.rglob('*') if p.is_file()}
 manifest={n:hashlib.sha256(base64.b64decode(b)).hexdigest() for n,b in files.items()}
 print(json.dumps(dict(summary=summary,files=files,sha256=manifest)))
else:print(json.dumps(summary))
'''.replace('RUN',repr(active['run'])).replace('IDENTITY',repr(active['identity'])).replace('COLLECT',repr(collect)),60)
    value=json.loads(raw)
    if collect:
        dest=PRIVATE/'collected';dest.mkdir(exist_ok=True)
        for n,b in value['files'].items():
            p=dest/n;assert p.resolve().is_relative_to(dest.resolve())
            p.parent.mkdir(parents=True,exist_ok=True);data=base64.b64decode(b)
            assert hashlib.sha256(data).hexdigest()==value['sha256'][n];p.write_bytes(data)
        (PRIVATE/'collection_manifest.json').write_bytes(enc(value['sha256']))
        value=value['summary']
    (PRIVATE/'latest_status.json').write_bytes(enc(value))
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['launch','status','collect']);a=ap.parse_args()
    launch() if a.action=='launch' else status(a.action=='collect')
