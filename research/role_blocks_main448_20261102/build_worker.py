"""Generate queue plumbing around byte-identical tested scientific functions."""
import ast,hashlib,json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent/'role_smoke_20261102'

def build():
    text=(PARENT/'smoke_worker.py').read_text(encoding='utf-8')
    text=text.replace('"""Explicitly launched two-question engineering run; immutable upstream imports."""',
        '"""Authorized main448 queue; science identical to the passed smoke."""')
    text=text.replace('from cloud_common import save, append, load, sha, identity',
        'from cloud_common import save, append, load, sha, identity, qlock, all_rows, key\nfrom queue_logic import choose')
    text=text.replace('def verify():','def verify(snapshots=False):')
    text=text.replace("    assert sha(BASE/'source_lock.json')", "    if snapshots:\n        for name,entry in lock['snapshots'].items():\n            assert sha(RUN/name)==entry['sha256'], name\n    assert sha(BASE/'source_lock.json')",1)
    text=text.replace('count=0;current=None','count=0;current=None;current_task=None')
    text=text.replace('source=verify();plan=', 'source=verify(snapshots=True);plan=')
    old='        for repeat in range(1,4):\n            for item in items:\n'
    new='''        while True:
            with qlock():
                queue=load(RUN/'queue.json');task=choose(queue['tasks'])
                if task is not None:
                    task.update(state='running',owner_pid=os.getpid(),attempt=str(out),started=time.time())
                    save(RUN/'queue.json',queue);current_task=dict(task)
            if task is None:break
            item=items[task['item_index']];repeat=1
            if True:
'''
    assert old in text;text=text.replace(old,new)
    text=text.replace("                for mode in order:\n", "                for mode in order:\n                    if mode not in current_task['methods']:continue\n")
    text=text.replace("                    selections[mode]=position.detach().cpu()", "                    actual_kv_bytes=sum(t.numel()*t.element_size() for pair in pairs(compact) for t in pair)\n                    assert actual_kv_bytes==19169280\n                    selections[mode]=position.detach().cpu()")
    text=text.replace("row=dict(**current,method=mode,status='ok',engineering_only=True,latent_steps=10,seed=42,",
        "row=dict(**current,dataset=item['dataset'],is_diamond=item['is_diamond'],gold=item['gold'],correct=bool(pred and pred==normalize_answer(str(item['gold']))),method=mode,status='ok',engineering_only=False,latent_steps=10,seed=42,")
    text=text.replace('retained_kv_bytes=130*147456,','retained_kv_bytes=actual_kv_bytes,')
    text=text.replace("                    append(out/'results.jsonl',row);count+=1", "                    with qlock():\n                        assert key(row) not in {key(x) for x in all_rows()}, 'duplicate new answer'\n                        append(out/'results.jsonl',row)\n                    count+=1")
    text=text.replace("                for x,y in [('T','R'),('R','R-shuffle')]:\n", "                for x,y in [('T','R'),('R','R-shuffle')]:\n                    if x not in selections or y not in selections:continue\n")
    text=text.replace("                del captured,cache,mapping,selections;gc.collect();torch.cuda.empty_cache()", """                del captured,cache,mapping,selections;gc.collect();torch.cuda.empty_cache()
                with qlock():
                    queue=load(RUN/'queue.json');task=next(t for t in queue['tasks'] if t['id']==current_task['id'])
                    assert task['owner_pid']==os.getpid()
                    assert {(item['dataset'],item['question_id'],10,m) for m in task['methods']} <= {key(x) for x in all_rows()}
                    task.update(state='complete',ended=time.time());save(RUN/'queue.json',queue)
                record('last_task_commit.json',dict(task=current_task['id']))
                current_task=None""")
    text=text.replace("record('exit.json',dict(completed=count==18,committed=count))", "record('exit.json',dict(completed=True,committed=count,released_idle_tail=True))")
    text=text.replace("    except BaseException as exc:\n", """    except BaseException as exc:
        if current_task is not None:
            with qlock():
                queue=load(RUN/'queue.json');task=next(t for t in queue['tasks'] if t['id']==current_task['id'])
                task.update(state='failed_held',failure=repr(exc),ended=time.time());save(RUN/'queue.json',queue)
""")
    ast.parse(text)
    def scientific_functions(source):
        names={'args_for','timed','stage','probe','judge','final_select','validate'}
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name in names}
    assert scientific_functions(text)==scientific_functions((PARENT/'smoke_worker.py').read_text(encoding='utf-8'))
    (HERE/'cloud_worker.py').write_text(text,encoding='utf-8',newline='\n')
    (HERE/'block_selector.py').write_bytes((PARENT/'block_selector.py').read_bytes())
    proof=dict(scientific_functions_ast_equal=True,selector_byte_equal=True,
        parent_worker_sha256=hashlib.sha256((PARENT/'smoke_worker.py').read_bytes()).hexdigest(),
        worker_sha256=hashlib.sha256((HERE/'cloud_worker.py').read_bytes()).hexdigest(),
        changes=['dynamic queue and per-key durable commit','one repeat only','metadata and actual tensor byte count'])
    (HERE/'worker_compatibility.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    return proof

if __name__=='__main__':print(json.dumps(build()))
