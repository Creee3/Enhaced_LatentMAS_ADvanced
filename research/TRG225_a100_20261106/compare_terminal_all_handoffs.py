"""Offline six-method comparison on identical 96 GPQA questions, no inference."""
import hashlib,importlib.util,json,math,statistics
from pathlib import Path
ROOT=Path('D:/liuxin_workspace/ICML2027/advanced_MOC/experiments')
OLD=ROOT/'latentmas_TRG225_a100_20261106/completed_verified_0e54558e885e/run'
NEW=ROOT/'latentmas_rolewise_TRG576_20261108/verified_formal_final_438ac46bb046'

def rows(p):return [json.loads(b) for b in p.read_bytes().split(b'\n') if b.strip()]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def duration(t):
 s=round(t);return f'{s//3600:02d}:{s%3600//60:02d}:{s%60:02d}'
def mcnemar(first,second):
 b=sum(a and not c for a,c in zip(first,second));c=sum(c and not a for a,c in zip(first,second));n=b+c
 return dict(new_only=b,old_only=c,net=b-c,two_sided_exact_p=min(1,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1,exploratory=True)

def main():
 refs=[]
 for base in [OLD,NEW]:
  manifest=load(base.parent/'manifest.json' if base==OLD else base/'manifest.json')
  files=[base/'questions.json']+list((base/'attempts').glob('*/results.jsonl'))
  if base==OLD:files+=[base/'reused_results.jsonl',base/'author_parser.py']
  for p in files:
   prefix=base.parent if base==OLD else base;entry=manifest[p.relative_to(prefix).as_posix()];assert sha(p)==entry['sha256'];refs.append(dict(file=str(p),sha256=sha(p)))
 qs=[q for q in load(NEW/'questions.json') if q['dataset']=='gpqa'];oldqs={q['question_id']:q for q in load(OLD/'questions.json')};ids={q['question_id'] for q in qs};assert len(ids)==96
 assert all(oldqs[q['question_id']]['question']==q['question'] and oldqs[q['question_id']]['gold']==q['gold'] and oldqs[q['question_id']]['is_diamond']==q['is_diamond'] for q in qs)
 gold={q['question_id']:str(q['gold']).upper() for q in qs}
 a=rows(OLD/'reused_results.jsonl')+[r for f in (OLD/'attempts').glob('*/results.jsonl') for r in rows(f)];a=[r for r in a if r['question_id'] in ids]
 b=[r for f in (NEW/'attempts').glob('*/results.jsonl') for r in rows(f) if r['dataset']=='gpqa' and r['question_id'] in ids];assert len(a)==len(b)==288
 spec=importlib.util.spec_from_file_location('author_parser',OLD/'author_parser.py');parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
 assert sha(OLD/'author_parser.py')=='92163f3a5389c4a7e25868ff931fc919703eb1e500ef74dc03b4810ab6f0225d'
 by={(r['question_id'],r['method']):r for r in a+b};assert len(by)==576;pred={}
 for key,r in by.items():
  raw=parser.extract_gsm8k_answer(r['answer_text'])
  v=parser.normalize_answer(raw) if len(r['method'])==1 else (None if raw is None else raw.strip().upper())
  if len(r['method'])>1 and v not in ['A','B','C','D']:v=None
  assert v==r['prediction'];pred[key]=bool(v) and parser.normalize_answer(v)==parser.normalize_answer(gold[key[0]])
 result=dict(passed=True,model_calls=0,questions=96,diamond=48,nonDiamond=48,question_text_gold_subset_exact_match=True,dataset_name_normalized=True,methods={},comparisons={},verified_inputs=refs,timing_caveat='Historical recorded reconstructed standalone times; different concurrency/CPU/MPS regimes; not isolated method speedups')
 order=['T','TTT','R','RRR','G','GGG'];sorted_ids=sorted(ids)
 for m in order:
  z=[r for r in a+b if r['method']==m];assert len(z)==96
  scopes={}
  for name,sub in [('all96',z),('Diamond48',[r for r in z if r['is_diamond']]),('nonDiamond48',[r for r in z if not r['is_diamond']])]:scopes[name]=dict(correct=sum(pred[(r['question_id'],m)] for r in sub),n=len(sub))
  times=[r['timing']['reconstructed_standalone_seconds'] for r in z]
  result['methods'][m]=dict(scopes=scopes,mean_reconstructed_seconds=statistics.mean(times),sum_reconstructed_seconds=sum(times),mean_judger_seconds=statistics.mean(r['timing']['judger_seconds'] for r in z),mean_selector_seconds=statistics.mean(r['timing']['selector_and_gather_seconds'] if len(m)==1 else sum(x['selector_and_gather_seconds'] for x in r['timing']['components']) for r in z),mean_generated_tokens=statistics.mean(r['generated_tokens'] for r in z))
 for m in ['T','R','G']:result['comparisons'][m*3+'_vs_'+m]=mcnemar([pred[(i,m*3)] for i in sorted_ids],[pred[(i,m)] for i in sorted_ids])
 report=['# 末次交接与三次交接：同题GPQA96六方法对比','','旧225题仅取新576题GPQA子集的相同96题，逐题核对qid、原文、gold和Diamond标签一致。旧dataset标签按共同qid归一化，禁止直接将225题全体与96题相比较。冻结作者解析器重算576条答案，全部与原prediction一致；无新推理。','','| 方法 | 修改范围 | 全部96 | Diamond48 | 非Diamond48 | 平均单臂重建时间 | 单臂重建累计 |','|---|---|---:|---:|---:|---:|---:|']
 for m in order:
  z=result['methods'][m];cells=[f"{v['correct']}/{v['n']} ({v['correct']/v['n']:.2%})" for v in z['scopes'].values()]
  report.append('|'+m+'|'+('仅Refiner→Judger' if len(m)==1 else '三次交接')+'|'+'|'.join(cells)+'|'+duration(z['mean_reconstructed_seconds'])+'|'+duration(z['sum_reconstructed_seconds'])+'|')
 report+=['','| 探索性同题比较 | 新方法独有正确 | 旧方法独有正确 | 净正确差 | 双侧精确McNemar p |','|---|---:|---:|---:|---:|']
 for name,v in result['comparisons'].items():report.append(f"|{name}|{v['new_only']}|{v['old_only']}|{v['net']:+d}|{v['two_sided_exact_p']:.6f}|")
 report+=['','解释限制：旧方案前两次原receiver通常46/88行；新三次交接通常110/120/130行，故算法作用位置与前两跳预算同时变化，不能隔离为角色方法扩展的因果收益。三对跨run比较是事后探索性原始p，不是预登记确认性检验。','', '两轮CPU亲和/MPS/并发配置不同，生成长度也不同。耗时是各自历史重建口径，共享前缀计入单臂；不是独立实测延迟或GPU忙碌时间，累计存在并发重叠，不能由此宣称三次压缩比一次更快。','','旧225题整体生产窗口15:53:19不等于这96题连续独立执行时间；新576题整体墙钟也不能当这96题的墙钟。未记录独立96题生产run，因此不编造96题独立总墙钟。']
 text='\n'.join(report)+'\n';out=ROOT/'latentmas_rolewise_TRG576_20261108/comparison_terminal_all96_20261109';out.mkdir(exist_ok=True);(out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');(out/'comparison.md').write_text(text,encoding='utf-8');Path(__file__).with_name('terminal_vs_all96_20261109.md').write_text(text,encoding='utf-8')
 print(json.dumps(dict(passed=True,questions=96,keys=576,model_calls=0,comparisons=result['comparisons'],report=str(out/'comparison.md')),ensure_ascii=False))
if __name__=='__main__':main()
