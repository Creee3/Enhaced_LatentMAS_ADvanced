"""Offline audit of the single CPU pilot; no model imports or raw-answer output."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
PRIVATE=REPO.parent/'advanced_MOC/experiments/latentmas_cpu_single_question_20261104'
dest=PRIVATE/'collected'
def load(n):return json.loads((dest/n).read_text(encoding='utf-8'))
def rows(n):return [json.loads(x) for x in (dest/n).read_bytes().split(b'\n') if x]


def main():
    manifest=json.loads((PRIVATE/'collection_manifest.json').read_text(encoding='utf-8'))
    for n,digest in manifest.items():assert hashlib.sha256((dest/n).read_bytes()).hexdigest()==digest
    lock=load('run_lock.json')
    for name,k in [('worker.py','worker_sha256'),('question.json','question_sha256'),('author_parser.py','author_parser_py_sha256')]:
        assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==lock[k]
    sys.path.insert(0,str(dest));from author_parser import extract_gsm8k_answer,normalize_answer
    result=load('result.json');answer=load('answer_private.json');question=load('question.json')
    assert result['completed'] and result['engineering_only'] and not result['formal_G_result']
    assert result['cpu_only'] and result['device']=='cpu' and result['dtype']=='bfloat16'
    assert result['parameter_bytes']==16381470720 and result['generated_tokens']==len(answer['generated_ids'])==6
    assert result['threads']==16 and not result['enable_thinking'] and not result['hit_cap']
    prediction=normalize_answer(extract_gsm8k_answer(answer['answer_text']))
    assert prediction==result['prediction'] and result['correct']==bool(prediction and prediction==normalize_answer(str(question['gold'])))
    raw=rows('raw/requests_raw.jsonl');trace=rows('trace.jsonl')
    starts=[x for x in trace if x['event']=='request_started'];settles=[x for x in trace if x['event']=='request_settled']
    inputs=[x for x in raw if x['event']=='request_input'];outputs=[x for x in raw if x['event']=='request_output']
    assert len(starts)==len(settles)==len(inputs)==len(outputs)==1
    assert starts[0]['reservation_id']==settles[0]['reservation_id']==inputs[0]['reservation_id']==outputs[0]['reservation_id']
    assert outputs[0]['generated_ids']==answer['generated_ids'] and not outputs[0]['failed']
    assert len(inputs[0]['input']['input_ids'])==result['input_tokens']==225
    assert settles[0]['actual_positions']==230 and settles[0]['forward_calls']==6
    assert not settles[0]['failed'] and settles[0]['independent_unknown']==0
    assert sum(x['event']=='model_file_verified' for x in trace)==15
    status=json.loads((PRIVATE/'latest_status.json').read_text(encoding='utf-8'))
    assert not status['alive'] and status['own_reserved_count']==0
    report=dict(passed=True,files_sha_verified=len(manifest),one_request_only=True,model_sha_files=15,
        cpu_only=True,prediction_recomputed=True,raw_token_ids_match=True,actual_positions=230,forwards=6,
        incremental_unknown=0,own_reserved_count=0,process_exited=True,no_model_calls_in_audit=True,
        not_a_formal_G_result=True,one_question_not_accuracy_evidence=True)
    (PRIVATE/'audit_verified.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
