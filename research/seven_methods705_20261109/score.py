"""Frozen full-scope scoring only after BOTH raw archives and all original answers pass."""
import ast,json,math,sys,itertools
from pathlib import Path
from common import HERE,load,save,rows,verify,sha
from task_adapter import parse_answer,normalized_gold

def main():
 plan=load(HERE/'plan.json');other=Path(plan['other_run']);runs={load(p/'plan.json')['kind']:p for p in [HERE,other]};hl=runs['HL'];ext=runs['extension']
 for p in runs.values():
  assert load(p/'formal_audit_verified.json')['passed'] and load(p/'formal_archive_verified.json')['passed']
  lock=load(p/'run_lock.json')
  for n,m in lock['files'].items():assert sha(p/n)==m['sha256']
 refs=load(hl/'reference_lock.json')['files'];old=[]
 for ref in refs:
  p=Path(ref['path']);assert p.stat().st_size==ref['bytes'] and sha(p)==ref['sha256']
  if p.name=='results.jsonl':old.extend(rows(p))
 assert len(old)==2880
 new=[r for p in runs.values() for f in (p/'attempts').glob('*/results.jsonl') for r in rows(f)];assert len(new)==2055
 items=load(hl/'questions.json');assert len(items)==705
 methods=['Full','receiver','GGG','RRR','TTT','H','L'];by={(r['dataset'],r['question_id'],r['method']):r for r in old+new}
 expected={(q['dataset'],q['question_id'],m) for q in items for m in methods};assert len(by)==len(expected)==4935 and set(by)==expected
 refscore=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025/evidence/rolewise_TRG576_s42_l10_20261108T141249Z/score.py')
 assert sha(refscore)==load(refscore.parent/'run_lock.json')['files']['score.py']['sha256']
 tree=ast.parse(refscore.read_text());defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['exact_mcnemar','wilson']];assert len(defs)==2
 ns=dict(math=math);exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),'frozen_statistics','exec'),ns)
 sys.path.insert(0,str(HERE.parent/'TRG225_a100_s42_l10_20261106T155639Z'));from author_parser import extract_gsm8k_answer
 pred={}
 for q in items:
  for m in methods:
   key=(q['dataset'],q['question_id'],m);r=by[key];assert r['seed']==42 and r['latent_steps']==10
   value=parse_answer(q,r['answer_text'],extract_gsm8k_answer);assert value==r['prediction'];pred[key]=value==normalized_gold(q)
 groups={ds:[q for q in items if q['dataset']==ds] for ds in ['gpqa','mmlu_pro','gsm8k']}
 groups.update(gpqa_diamond=[q for q in groups['gpqa'] if q['is_diamond']],gpqa_non_diamond=[q for q in groups['gpqa'] if not q['is_diamond']])
 result=dict(complete=True,existing_keys=2880,new_keys=2055,combined_keys=4935,datasets={},accuracy_computed_after_both_raw_archives=True,old_results_modified=False,scope='Fixed705 same questions; original225 GPQA development pool; new comparisons exploratory',billing={k:load(p/'formal_audit_verified.json') for k,p in runs.items()})
 for name,qs in groups.items():
  metrics={}
  for m in methods:
   rs=[by[(q['dataset'],q['question_id'],m)] for q in qs];correct=sum(pred[(q['dataset'],q['question_id'],m)] for q in qs)
   metrics[m]=dict(correct=correct,total=len(qs),accuracy=correct/len(qs),Wilson95CI=ns['wilson'](correct,len(qs)),parse_failures=sum(not r['parse_valid'] for r in rs),cap_hits=sum(r['generated_tokens']==8192 for r in rs),actual_KV_bytes_by_handoff=[sum(r['actual_kv_bytes_by_handoff'][h] for r in rs) for h in range(3)],cumulative_KV_retained_ratio=sum(r['actual_cumulative_relay_bytes'] for r in rs)/sum(sum(r['full_reference_bytes_by_original_positions']) for r in rs))
  tests={a+'_vs_'+b:ns['exact_mcnemar']([pred[(q['dataset'],q['question_id'],a)] for q in qs],[pred[(q['dataset'],q['question_id'],b)] for q in qs]) for a,b in itertools.combinations(methods,2)}
  agreement={m:sum(by[(q['dataset'],q['question_id'],m)]['shared_planner_kv_sha']==by[(q['dataset'],q['question_id'],'Full')]['shared_planner_kv_sha'] for q in qs) for m in methods}
  result['datasets'][name]=dict(n=len(qs),methods=metrics,exploratory_pairwise_comparisons=tests,Planner_SHA_agreement_with_Full=agreement)
 report=['# 七方法705题完整同题汇总','','既有2880原答冻结复用，新增2055；不同run只读同题比较，Planner SHA一致数单列；不声称跨run共享RID。GPQA来自已看过的225开发范围，Diamond106/非Diamond119为子集，不相加。所有新增比较探索性。','','| 题集 | Full | receiver | GGG | RRR | TTT | H | L |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for name,qs in groups.items():report.append('|'+name+'|'+'|'.join(f"{result['datasets'][name]['methods'][m]['correct']}/{len(qs)} ({result['datasets'][name]['methods'][m]['accuracy']:.2%})" for m in methods)+'|')
 report+=['','H/L是发送者latent注意力、每角色新prompt32、无末8强制，典型46/88/130；receiver为32+末8，GGG/RRR/TTT110/120/130，早期预算不同。Full完整原KV。真实KV字节与累计比见JSON，不当作实测通信或GPU/FLOP加速。']
 for p in runs.values():save(p/'final_summary.json',result);(p/'final_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
 print(json.dumps(dict(complete=True,new_keys=2055,combined_keys=4935)))

if __name__=='__main__':main()
