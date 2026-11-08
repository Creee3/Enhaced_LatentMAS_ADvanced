"""Frozen post-completion scoring only; no model or semantic answer review."""
import hashlib,json,math,sys
from common import HERE,ROOT,SCIENCE,METHODS,load,save,records,verify
from task_adapter import parse_answer,normalized_gold
def exact_mcnemar(a,b):
 win=sum(x and not y for x,y in zip(a,b));lose=sum(y and not x for x,y in zip(a,b));n=win+lose;p=min(1.0,2*sum(math.comb(n,k) for k in range(min(win,lose)+1))/2**n) if n else 1.0
 return dict(first_only=win,second_only=lose,net_correct=win-lose,accuracy_difference=(win-lose)/len(a),two_sided_exact_p=p)
def wilson(correct,n):
 z=1.959963984540054;p=correct/n;center=(p+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
 return [max(0,center-half),min(1,center+half)]
def main():
 source=verify();assert load(HERE/'formal_audit_verified.json')['passed'] and load(HERE/'formal_archive_verified.json')['passed'];allrows=records();assert len(allrows)==1728
 sys.path.insert(0,str(SCIENCE));from author_parser import extract_gsm8k_answer
 items=load(HERE/'questions.json');pred={};byrow={}
 for r in allrows:
  item=items[r['index']];p=parse_answer(item,r['answer_text'],extract_gsm8k_answer);assert p==r['prediction'];pred[(r['index'],r['method'])]=bool(p is not None and p==normalized_gold(item));byrow[(r['index'],r['method'])]=r
 assert len(pred)==1728
 result=dict(complete=True,source=source,total_cases=576,total_keys=1728,datasets={},primary_comparison='RRR versus GGG; three dataset tests, Holm family alpha .05',accuracy_computed_after_archive=True,old_answers_modified=False)
 groups={ds:[x['index'] for x in items if x['dataset']==ds] for ds in ['gpqa','mmlu_pro','gsm8k']}
 groups.update(gpqa_diamond=[x['index'] for x in items if x['dataset']=='gpqa' and x['is_diamond']],gpqa_non_diamond=[x['index'] for x in items if x['dataset']=='gpqa' and not x['is_diamond']])
 for name,ix in groups.items():
  table={m:dict(correct=sum(pred[(i,m)] for i in ix),total=len(ix),accuracy=sum(pred[(i,m)] for i in ix)/len(ix),parse_failures=sum(not byrow[(i,m)]['parse_valid'] for i in ix),cap_hits=sum(byrow[(i,m)]['generated_tokens']==8192 for i in ix),generated_tokens=sum(byrow[(i,m)]['generated_tokens'] for i in ix),actual_KV_bytes_by_handoff=[sum(byrow[(i,m)]['actual_kv_bytes_by_handoff'][h] for i in ix) for h in range(3)],cumulative_KV_retained_ratio=sum(byrow[(i,m)]['actual_cumulative_relay_bytes'] for i in ix)/sum(sum(byrow[(i,m)]['full_reference_bytes_by_original_positions']) for i in ix),judger_seconds=sum(byrow[(i,m)]['timing']['judger_seconds'] for i in ix),reconstructed_standalone_seconds=sum(byrow[(i,m)]['timing']['reconstructed_standalone_seconds'] for i in ix),reconstructed_not_independently_timed=True) for m in METHODS}
  tests={a+'_vs_'+b:exact_mcnemar([pred[(i,a)] for i in ix],[pred[(i,b)] for i in ix]) for a,b in [('RRR','GGG'),('RRR','TTT'),('GGG','TTT')]};result['datasets'][name]=dict(n=len(ix),methods=table,comparisons=tests,subgroup_not_independent=name.startswith('gpqa_'))
  for m in METHODS:table[m]['Wilson_95CI']=wilson(table[m]['correct'],len(ix))
  import numpy as np
  strata={}
  for i in ix:strata.setdefault(items[i]['category'],[]).append(i)
  for pair_id,(a,b) in enumerate([('RRR','GGG'),('RRR','TTT'),('GGG','TTT')]):
   rng=np.random.default_rng(20261108+list(groups).index(name)*10+pair_id);boot=np.zeros(10000)
   for indices in strata.values():
    diff=np.array([int(pred[(i,a)])-int(pred[(i,b)]) for i in indices]);boot+=rng.choice(diff,size=(10000,len(indices)),replace=True).sum(-1)
   tests[a+'_vs_'+b]['paired_question_stratified_bootstrap_95CI']=np.quantile(boot/len(ix),[.025,.975]).tolist()
 primary=sorted((result['datasets'][d]['comparisons']['RRR_vs_GGG']['two_sided_exact_p'],d) for d in ['gpqa','mmlu_pro','gsm8k']);adjusted=0
 for j,(p,d) in enumerate(primary):
  adjusted=max(adjusted,min(1,p*(3-j)));test=result['datasets'][d]['comparisons']['RRR_vs_GGG'];test.update(Holm_adjusted_p=adjusted,Holm_reject_alpha_05=adjusted<=.05)
 result['mmlu_pro_categories']={cat:{m:dict(correct=sum(pred[(x['index'],m)] for x in items if x['dataset']=='mmlu_pro' and x['category']==cat),total=sum(x['dataset']=='mmlu_pro' and x['category']==cat for x in items)) for m in METHODS} for cat in sorted({x['category'] for x in items if x['dataset']=='mmlu_pro'})}
 result['billing']=load(HERE/'formal_audit_verified.json');save(HERE/'final_summary.json',result)
 report=['# 全交接GGG/RRR/TTT 576题完整结果','', '全部1728键完成、结构/费用/退出及原始归档核验后，由冻结程序评分。GPQA是已看过开发题；新题集固定样本不代表模型训练未见。无R-shuffle，本轮不证明真实角色边界因果特异性。','', '| 范围 | GGG | RRR | TTT |','|---|---:|---:|---:|']
 for ds in groups:
  a=result['datasets'][ds];report.append('|'+ds+'|'+ '|'.join(f"{a['methods'][m]['correct']}/{a['n']} ({a['methods'][m]['accuracy']:.2%})" for m in METHODS)+'|')
 report+=['','主比较RRR对GGG采用双侧精确McNemar，三个题集Holm控制alpha .05；其余比较与子集描述单独标注，不把不显著当等价。','', 'KV按三次交接实际字节累计/相同原位置Full字节参考汇总；无Full生成臂，字节比不代表GPU/FLOP/网络实测收益。重建单臂计时不是独立实测延迟。','',f"去重RID {result['billing']['unique_RIDs']}，实际位置 {result['billing']['actual_positions']}，新增unknown {result['billing']['unknown_added']}，自有预留0。"]
 (HERE/'final_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8');print(json.dumps(dict(complete=True,keys=1728,scoring_finished=True)))
if __name__=='__main__':main()
