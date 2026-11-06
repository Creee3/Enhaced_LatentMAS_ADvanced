"""Frozen offline completion-only scorer. Never reads accuracy before full coverage."""
import argparse,hashlib,importlib.util,json,math,sys
from pathlib import Path
METHODS=('T','R','G')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
 b=p.read_bytes();assert not b or b.endswith(bytes([10])),'Partial durable JSONL'
 return [json.loads(x) for x in b.split(bytes([10])) if x]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def key(x):return x['dataset'],x['question_id'],x['latent_steps'],x['method']
def exact_mcnemar(first,second):
 assert len(first)==len(second)
 b=sum(bool(x) and not bool(y) for x,y in zip(first,second));c=sum(bool(y) and not bool(x) for x,y in zip(first,second))
 n=b+c;numerator=min(2**n,2*sum(math.comb(n,i) for i in range(min(b,c)+1))) if n else 1;denominator=2**n
 p=numerator/denominator
 return dict(first_only_correct=b,second_only_correct=c,discordant=n,net_correct_difference=b-c,
  two_sided_exact_p=p,p_numerator=str(numerator),p_denominator=str(denominator),alpha=.05,
  first_net_improvement_and_significant=b>c and p<.05)
def coverage(data,questions):
 expected={(q['dataset'],q['question_id'],10,m) for q in questions for m in METHODS}
 keys=list(map(key,data));assert len(keys)==len(set(keys)) and set(keys)==expected,'Full exact coverage required before scoring'
 return {key(r):r for r in data}
def scores(data,questions,extract,normalize):
 bykey=coverage(data,questions);result={}
 for q in questions:
  for method in METHODS:
   k=(q['dataset'],q['question_id'],10,method);r=bykey[k]
   predicted=normalize(extract(r['answer_text']));correct=bool(predicted and predicted==normalize(str(q['gold'])))
   assert predicted==r['prediction'] and correct==r['correct'] and str(q['gold'])==str(r['gold'])
   assert r['parser_sha']=='92163f3a5389c4a7e25868ff931fc919703eb1e500ef74dc03b4810ab6f0225d'
   result[k]=correct
 return result
def scope_summary(data,questions,correct,primary=False):
 bykey=coverage(data,questions);metrics={};vectors={}
 for method in METHODS:
  selected=[bykey[(q['dataset'],q['question_id'],10,method)] for q in questions]
  vector=[correct[key(x)] for x in selected];vectors[method]=vector
  actual=sum(x['retained_kv_bytes'] for x in selected);full=sum(x['full_bytes_from_original_positions'] for x in selected)
  assert all(x['retained_kv_bytes']==19169280 for x in selected) and full>actual
  metrics[method]=dict(correct=sum(vector),questions=len(vector),accuracy=sum(vector)/len(vector),
   measured_retained_KV_bytes_sum=actual,uncompressed_KV_reference_bytes_sum=full,
   reference_definition='original pre-Judger positions times measured BF16 KV bytes/position; no measured communication/GPU/FLOP claim',
   retained_ratio=actual/full,reduced_ratio=1-actual/full)
 return dict(methods=metrics,**{'R_vs_T_primary' if primary else 'R_vs_T_subgroup_exploratory':exact_mcnemar(vectors['R'],vectors['T'])},
  R_vs_G_exploratory=exact_mcnemar(vectors['R'],vectors['G']),T_vs_G_exploratory=exact_mcnemar(vectors['T'],vectors['G']))
def trace_cost(run):
 receipts={}
 for p in (run/'attempts').glob('*/trace.jsonl'):
  for x in rows(p):
   if x['event']!='request_settled':continue
   rid=x['reservation_id'];assert rid not in receipts,'Duplicate settlement RID';receipts[rid]=x
 return receipts
def cost_summary(receipts):
 return dict(unique_RIDs=len(receipts),actual_positions=sum(x['actual_positions'] for x in receipts.values()),
  unknown_upper_bound=sum(x['independent_unknown'] for x in receipts.values()),failed_requests=sum(bool(x['failed']) for x in receipts.values()),unit='token positions, not dollars')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path);args=ap.parse_args();run=args.run
 # All completion gates precede parsing answer text or computing correctness.
 lock=load(run/'run_lock.json')
 for group in ['files','snapshots']:
  for name,entry in lock[group].items():assert sha(run/name)==entry['sha256']
 questions=load(run/'questions.json');assert len(questions)==225 and sum(q['is_diamond'] for q in questions)==106
 data=rows(run/'reused_results.jsonl')
 for p in (run/'attempts').glob('*/results.jsonl'):data+=rows(p)
 coverage(data,questions);assert len(data)==675
 assert all(x['state']=='complete' for x in load(run/'queue.json')['tasks']) and load(run/'manager_exit.json')['completed']
 sys.path.insert(0,str(run));from cloud_common import identity
 for w in load(run/'workers.json'):
  actual=identity(w['pid']);assert not actual or any(actual[k]!=w['initial_identity'][k] for k in ['uid','start_ticks','argv','CUDA_VISIBLE_DEVICES'])
 manager=load(run/'manager.json')['identity'];actual=identity(manager['pid']);assert not actual or any(actual[k]!=manager[k] for k in ['uid','start_ticks','argv','CUDA_VISIBLE_DEVICES'])
 budget=load(run.parents[1]/'cumulative_budget.json');assert budget['reserved_tokens']==0 and not budget['active_reservations']
 assert sum(budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens'])<=115000000
 archive=load(run/'management/completion_archive_verified.json')
 assert archive['passed'] and all(archive[k] for k in ['remote_stream_sha','gzip_crc','exact_members','per_member_sha','safe_full_extract_readback_sha'])
 analysis_lock=load(run/'management/analysis_program_lock.json');assert analysis_lock['sha256']==sha(Path(__file__))
 parser=run/'author_parser.py';assert sha(parser)=='92163f3a5389c4a7e25868ff931fc919703eb1e500ef74dc03b4810ab6f0225d'
 spec=importlib.util.spec_from_file_location('frozen_author_parser',parser);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 correct=scores(data,questions,module.extract_gsm8k_answer,module.normalize_answer)
 summaries={}
 for label,subset in [('all225',questions),('Diamond106',[q for q in questions if q['is_diamond']]),('nonDiamond119',[q for q in questions if not q['is_diamond']])]:
  ids={(q['dataset'],q['question_id']) for q in subset};summaries[label]=scope_summary([r for r in data if (r['dataset'],r['question_id']) in ids],subset,correct,primary=label=='all225')
 pre=load(run/'launch_preflight.json');current=trace_cost(run);current_cost=cost_summary(current)
 assert current_cost['actual_positions']==budget['actual_tokens']-pre['budget_before']['actual_tokens']
 assert current_cost['unknown_upper_bound']==budget['unknown_cost_upper_bound_tokens']-pre['budget_before']['unknown_cost_upper_bound_tokens']
 pilot=Path(pre['pilot_run']);pilot_receipts=trace_cost(pilot);assert not current.keys()&pilot_receipts.keys()
 precursor=run.parents[1]/'evidence/attention_G_a100_matched225_s42_l10_20261106T151824Z';precursor_receipts=trace_cost(precursor)
 assert not (current.keys()|pilot_receipts.keys())&precursor_receipts.keys()
 result=dict(passed=True,exact_keys=675,scorer_sha256=sha(Path(__file__)),scopes=summaries,
  costs=dict(new_run=current_cost,reused_pilot=cost_summary(pilot_receipts),prior_A100_prefix_failure=cost_summary(precursor_receipts),
   all_three_deduplicated=cost_summary(dict(current,**pilot_receipts,**precursor_receipts))),
  caveats=['Previously viewed/development questions; exploratory, not new blind test','Diamond is a subset, not an independent replication','No R-shuffle; no role-boundary-specific causal claim','No cross-hardware timing aggregation','No original answer edited; author parser recomputed offline'])
 (run/'management/final_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(dict(passed=True,exact_keys=675,scopes=summaries,costs=result['costs']),ensure_ascii=False))
if __name__=='__main__':main()
