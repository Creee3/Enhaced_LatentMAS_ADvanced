"""Build a separately versioned G run; never mutates the historical run."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'role_blocks_main448_20261102'
REPO=HERE.parents[1]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build():
    original=(OLD/'cloud_worker.py').read_text(encoding='utf-8')
    worker=original.replace('Authorized main448 queue; science identical to the passed smoke.',
                            'Authorized G-only matched225 attention control; frozen prefix and gather.')
    assert worker.count('from block_selector import select')==1
    worker=worker.replace('from block_selector import select','from attention_selector import select')
    assert worker.count("order=['T','R','R-shuffle']")==1
    worker=worker.replace("order=['T','R','R-shuffle']", "order=['G']")
    worker=worker.replace("source=verify(snapshots=True);plan=load(RUN/'plan.json');items=load(RUN/'questions.json')",
        "source=verify(snapshots=True);plan=load(RUN/'plan.json');items=load(RUN/'questions.json')\n"
        "        assert plan['methods']==['G']\n"
        "        references=[json.loads(x) for x in (RUN/'baseline_results.jsonl').read_text(encoding='utf-8').splitlines() if x]\n"
        "        reference_T={r['question_id']:r for r in references if r['method']=='T'}\n"
        "        assert len(reference_T)==len(items)==225")
    marker="                (captured,pr),probe_seconds=timed(lambda:probe(item,'judger',cache,absolute,True))"
    assert worker.count(marker)==1
    worker=worker.replace(marker,"""                reference=reference_T[item['question_id']]
                assert [x['selected_position_sha256'] for x in prefix_transfers]==[x['selected_position_sha256'] for x in reference['prefix_transfers']], 'T prefix position mismatch; preserve scene'
                assert [[x[k] for k in ['original_before','original_end','prompt_len']] for x in stages]==[[x[k] for k in ['original_before','original_end','prompt_len']] for x in reference['prefix_stages']], 'T stage length mismatch'
                record('prefix_match.json',dict(**current,matched_T_prefix_positions=True))
"""+marker)
    worker=worker.replace('answer_text=answer,prediction=pred,parser_sha=',
                          'answer_text=answer,prediction=pred,matched_T_prefix_positions=True,parser_sha=')
    # Scientific functions (including gather and causal query capture) stay AST identical.
    names={'args_for','timed','stage','probe','judge','final_select','validate'}
    def funcs(source):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(source))
                if isinstance(n,ast.FunctionDef) and n.name in names}
    assert funcs(worker)==funcs(original)
    (HERE/'cloud_worker.py').write_text(worker,encoding='utf-8',newline='\n')
    for name in ['block_selector.py','management_identity.py','manager_recovery.py']:
        (HERE/name).write_bytes((OLD/name).read_bytes())
    manager=(OLD/'manager.py').read_text(encoding='utf-8')
    manager=manager.replace('target=1344,reused=6,new_target=1338',
                            "target=plan['total_target'],reused=0,new_target=plan['new_target']")
    (HERE/'manager.py').write_text(manager,encoding='utf-8',newline='\n')
    for name in ['cloud_common.py','stage_ops.py','current_jive.py','author_parser.py','queue_logic.py']:
        (HERE/name).write_bytes((REPO/'snapshot/run'/name).read_bytes())
    # Apply the proven management identity guard to worker metadata too.
    # No algorithm/generation edits; unreadable or foreign processes cannot be claimed.
    common=(HERE/'cloud_common.py').read_text(encoding='utf-8')
    common=common.replace("        fields=(p/'stat')", "        if p.stat().st_uid != os.getuid():return None\n        fields=(p/'stat')")
    common=common.replace('except (FileNotFoundError,ProcessLookupError):',
                          'except (PermissionError,FileNotFoundError,ProcessLookupError):')
    (HERE/'cloud_common.py').write_text(common,encoding='utf-8',newline='\n')
    proof=dict(parent_worker_sha256=sha(OLD/'cloud_worker.py'),worker_sha256=sha(HERE/'cloud_worker.py'),
               science_functions_ast_equal=sorted(names),selector='attention mass G; no V-dependent ranking',
               immutable_copied_files={n:sha(HERE/n) for n in ['block_selector.py','stage_ops.py','current_jive.py','author_parser.py','queue_logic.py']},
               control_checks=['G-only queue','frozen T prefix selected-position SHA before each final probe'],
               management_changes=['dynamic plan counts','UID-first PermissionError identity guard'])
    (HERE/'worker_compatibility.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    inspect=(OLD/'inspect_run.py').read_text(encoding='utf-8')
    inspect=inspect.replace("cohort=load(RUN/'cohort_questions.json');", "plan=load(RUN/'plan.json');cohort=load(RUN/'cohort_questions.json');")
    inspect=inspect.replace("['T','R','R-shuffle']", "plan['methods']")
    inspect=inspect.replace('target=1344,',"target=plan['total_target'],")
    inspect=inspect.replace('diamond_target=594,baseline_reused=1792,',
                            "diamond_target=plan['diamond_new_methods_target'],baseline_reused=plan['baseline_readonly'],")
    (HERE/'inspect_run.py').write_text(inspect,encoding='utf-8',newline='\n')
    (HERE/'capture_current.py').write_bytes((OLD/'capture_current.py').read_bytes())
    audit=(OLD/'verify_formal_worker.py').read_text(encoding='utf-8').replace("['T','R','R-shuffle']", "['G']")
    audit=audit.replace("assert row['source_k_equal_measured']", "assert row['matched_T_prefix_positions']\n assert row['source_k_equal_measured']")
    audit=audit.replace("formal={m:False for m in ['G']};checked=[]",
        "references=[json.loads(x) for x in (run/'baseline_results.jsonl').read_bytes().splitlines() if x]\n"
        "reference_T={r['question_id']:r for r in references if r['method']=='T'}\n"
        "formal={m:False for m in ['G']};checked=[]")
    audit=audit.replace(" rec=row['judger_receipt'];rid=rec['reservation_id']",
        " ref=reference_T[row['question_id']]\n"
        " assert [x['selected_position_sha256'] for x in row['prefix_transfers']]==[x['selected_position_sha256'] for x in ref['prefix_transfers']]\n"
        " assert row['original_end']==ref['original_end']\n"
        " assert len(row['selector_diagnostics'])==36\n"
        " assert all(d['mode']=='G' and d['bank_rows']<=192 and d['prompt_budget']==96 for d in row['selector_diagnostics'])\n"
        " rec=row['judger_receipt'];rid=rec['reservation_id']")
    audit=audit.replace("assert ready['verified_weight_files']==15", "assert abs(ready['target_norm']-1.3758320808410645)<1e-6\nassert ready['verified_weight_files']==15")
    (HERE/'verify_formal_worker.py').write_text(audit,encoding='utf-8',newline='\n')
    (HERE/'archive_failed_worker.py').write_bytes((OLD/'archive_failed_worker.py').read_bytes())
    return proof


if __name__=='__main__':print(json.dumps(build()))
