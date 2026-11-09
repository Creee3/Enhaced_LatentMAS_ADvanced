"""Frozen post-completion five-method same-question programmatic summary."""
import ast,hashlib,itertools,json,math,sys
from pathlib import Path
from common import HERE,SCIENCE,METHODS,load,save,records,verify,sha,rows
from task_adapter import parse_answer,normalized_gold

def main():
 verify();assert load(HERE/'formal_audit_verified.json')['passed'] and load(HERE/'formal_archive_verified.json')['passed']
 items=load(HERE/'questions.json');rs=records();assert len(rs)==1152
 ref=load(HERE/'reference_lock.json');base=Path(ref['run'])
 for n,m in ref['files'].items():assert (base/n).stat().st_size==m['bytes'] and sha(base/n)==m['sha256']
 old=[r for n in ref['files'] if n.endswith('results.jsonl') for r in rows(base/n)];assert len(old)==1728
 assert sha(base/'score.py')==load(base/'run_lock.json')['files']['score.py']['sha256']
 ns=dict(math=math);tree=ast.parse((base/'score.py').read_text());defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['exact_mcnemar','wilson']];assert len(defs)==2
 exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),'frozen_statistics','exec'),ns)
 sys.path.insert(0,str(SCIENCE));from author_parser import extract_gsm8k_answer
 five=['Full','receiver','GGG','RRR','TTT'];by={(r['index'],r['method']):r for r in rs+old};assert len(by)==576*5
 pred={}
 for (i,m),r in by.items():
  assert (r['dataset'],r['question_id'])==(items[i]['dataset'],items[i]['question_id'])
  answer=parse_answer(items[i],r['answer_text'],extract_gsm8k_answer);assert answer==r['prediction']
  pred[(i,m)]=answer==normalized_gold(items[i])
 groups={d:[x['index'] for x in items if x['dataset']==d] for d in ['gpqa','mmlu_pro','gsm8k']};groups['gpqa_diamond']=[x['index'] for x in items if x['dataset']=='gpqa' and x['is_diamond']];groups['gpqa_non_diamond']=[x['index'] for x in items if x['dataset']=='gpqa' and not x['is_diamond']]
 result=dict(complete=True,new_keys=1152,reference_keys=1728,combined_keys=2880,comparison_scope='same fixed questions; Full/receiver share a new Planner; GGG/RRR/TTT are read-only previous run',datasets={},billing_new=load(HERE/'formal_audit_verified.json'),old_results_modified=False,accuracy_computed_after_archive=True)
 for name,ix in groups.items():
  methods={}
  for m in five:
   correct=sum(pred[(i,m)] for i in ix);methods[m]=dict(correct=correct,total=len(ix),accuracy=correct/len(ix),Wilson95CI=ns['wilson'](correct,len(ix)),parse_failures=sum(not by[(i,m)]['parse_valid'] for i in ix),cap_hits=sum(by[(i,m)]['generated_tokens']==8192 for i in ix),actual_KV_bytes_by_handoff=[sum(by[(i,m)]['actual_kv_bytes_by_handoff'][h] for i in ix) for h in range(3)],cumulative_KV_retained_ratio=sum(by[(i,m)]['actual_cumulative_relay_bytes'] for i in ix)/sum(sum(by[(i,m)]['full_reference_bytes_by_original_positions']) for i in ix))
  comparisons={a+'_vs_'+b:ns['exact_mcnemar']([pred[(i,a)] for i in ix],[pred[(i,b)] for i in ix]) for a,b in itertools.combinations(five,2)}
  agreement=sum(by[(i,'Full')]['shared_planner_kv_sha']==by[(i,'GGG')]['shared_planner_kv_sha'] for i in ix)
  result['datasets'][name]=dict(n=len(ix),methods=methods,exploratory_pairwise_comparisons=comparisons,old_new_Planner_SHA_agreement=agreement)
 save(HERE/'final_summary.json',result)
 report=['# 同题576题 Full / 原receiver补跑与五方法汇总','','Full与原receiver本次各576条；GGG/RRR/TTT只读复用上一完整run，零重跑。新增两臂共享Planner；跨run前缀SHA一致数量见JSON，不谎称跨run共享RID。所有新增比较是探索性，GPQA为已看过开发题；Diamond是其子集。','','| 题集 | Full | 原receiver | GGG | RRR | TTT |','|---|---:|---:|---:|---:|---:|']
 for name in groups:report.append('|'+name+'|'+'|'.join(f"{result['datasets'][name]['methods'][m]['correct']}/{len(groups[name])} ({result['datasets'][name]['methods'][m]['accuracy']:.2%})" for m in five)+'|')
 report+=['','原receiver每个角色仅新prompt选32（末8保护），保留已有历史/latent/sink，典型46/88/130行；GGG/RRR/TTT为110/120/130，不能称每次交接预算相同。Full保留全部原KV，无额外probe调用。实际KV字节不是通信/GPU/FLOP实测加速。',f"新增去重RID {result['billing_new']['unique_RIDs']}，实际位置 {result['billing_new']['actual_positions']}，自有预留0。"]
 (HERE/'final_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8');print(json.dumps(dict(complete=True,new_keys=1152,combined_keys=2880)))

if __name__=='__main__':main()
