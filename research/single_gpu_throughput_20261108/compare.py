"""Aggregate-only exact-key old/new comparison. No semantic answer review."""
import json,importlib.util,hashlib
from pathlib import Path
ROOT=Path('D:/liuxin_workspace/ICML2027/advanced_MOC/experiments')
OLD=ROOT/'latentmas_TRG225_a100_20261106/completed_verified_0e54558e885e/run'
def rows(p):return [json.loads(b) for b in p.read_bytes().split(b'\n') if b]
def key(r):return r['dataset'],r['question_id'],r['latent_steps'],r['method']
def main():
 old={key(r):r for r in rows(OLD/'baseline_results.jsonl')}
 newrows=rows(OLD/'reused_results.jsonl')
 for p in (OLD/'attempts').glob('*/results.jsonl'):newrows+=rows(p)
 new={key(r):r for r in newrows};assert len(new)==len(newrows)==675
 parser=OLD/'author_parser.py';assert hashlib.sha256(parser.read_bytes()).hexdigest()=='92163f3a5389c4a7e25868ff931fc919703eb1e500ef74dc03b4810ab6f0225d'
 spec=importlib.util.spec_from_file_location('frozen',parser);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
 for r in list(old.values())+list(new.values()):
  prediction=p.normalize_answer(p.extract_gsm8k_answer(r['answer_text']))
  assert prediction==r['prediction']
  r['correct']=bool(prediction and prediction==p.normalize_answer(str(r['gold'])))
 result={}
 for m in ['T','R']:
  pairs=[(old[k],v) for k,v in new.items() if k[-1]==m];assert len(pairs)==225
  assert all(x['gold']==y['gold'] and x['parser_sha']==y['parser_sha'] and x['seed']==y['seed']==42 and x['retained_kv_bytes']==y['retained_kv_bytes'] for x,y in pairs)
  prefix=lambda r:[x['selected_position_sha256'] for x in r['prefix_transfers']]
  result[m]=dict(n=225,old_correct=sum(x['correct'] for x,y in pairs),new_correct=sum(y['correct'] for x,y in pairs),old_only_correct=sum(x['correct'] and not y['correct'] for x,y in pairs),new_only_correct=sum(y['correct'] and not x['correct'] for x,y in pairs),same_prefix_position_sha=sum(prefix(x)==prefix(y) for x,y in pairs),same_final_position_sha=sum(x['selected_position_sha']==y['selected_position_sha'] for x,y in pairs),same_output_token_sha=sum(x['token_sha']==y['token_sha'] for x,y in pairs),same_final_original_length=sum(x['original_end']==y['original_end'] for x,y in pairs))
 target=ROOT/'single_gpu_throughput_20261108';target.mkdir(exist_ok=True);(target/'old_new_comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
