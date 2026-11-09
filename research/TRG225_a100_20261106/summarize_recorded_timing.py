"""Read-only retrospective consistency/timing summary; no inference calls."""
import hashlib,importlib.util,json,statistics,datetime
from pathlib import Path
PRIVATE=Path('D:/liuxin_workspace/ICML2027/advanced_MOC/experiments/latentmas_TRG225_a100_20261106')
BASE=PRIVATE/'completed_verified_0e54558e885e'
HERE=Path(__file__).resolve().parent

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(b) for b in p.read_bytes().split(b'\n') if b.strip()] if p.exists() else []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def duration(s):
 s=round(s);return f'{s//3600:02d}:{s%3600//60:02d}:{s%60:02d}'
def stamp(s):return datetime.datetime.fromtimestamp(s,datetime.timezone(datetime.timedelta(hours=8))).isoformat()

def main():
 manifest=load(BASE/'manifest.json');verified=[]
 def check(path):
  rel=path.relative_to(BASE).as_posix();meta=manifest[rel]
  assert sha(path)==meta['sha256']
  expected=meta.get('bytes',meta.get('size'));assert expected is None or path.stat().st_size==expected
  verified.append(dict(file=rel,sha256=sha(path)))
 run=BASE/'run';inputs=[run/'questions.json',run/'queue.json',run/'workers.json',run/'reused_results.jsonl',run/'author_parser.py']+list((run/'attempts').glob('*/results.jsonl'))
 for path in inputs:check(path)
 summary=load(PRIVATE/'final_summary.json');scorer=module('locked_summary',HERE/'final_summary.py');assert sha(HERE/'final_summary.py')==summary['scorer_sha256']
 parser=module('locked_author_parser',run/'author_parser.py');assert sha(run/'author_parser.py')=='92163f3a5389c4a7e25868ff931fc919703eb1e500ef74dc03b4810ab6f0225d'
 reused=rows(run/'reused_results.jsonl');new=[v for f in (run/'attempts').glob('*/results.jsonl') for v in rows(f)];data=reused+new;assert len(data)==675 and len(reused)==4
 questions=load(run/'questions.json');correct=scorer.scores(data,questions,parser.extract_gsm8k_answer,parser.normalize_answer)
 result=dict(passed=True,model_calls=0,scope='A100 last-handoff-only GPQA225',new_answers=len(new),reused_answers=len(reused),verified_inputs=verified,accuracy={},timing={},production={},phases={},shared={})
 subsets={'all225':questions,'Diamond106':[q for q in questions if q['is_diamond']],'nonDiamond119':[q for q in questions if not q['is_diamond']]}
 for scope,qs in subsets.items():
  ids={(q['dataset'],q['question_id']) for q in qs};metrics={};timings={}
  for m in ['T','R','G']:
   selected=[r for r in data if r['method']==m and (r['dataset'],r['question_id']) in ids];n=len(selected);ncorrect=sum(correct[scorer.key(r)] for r in selected)
   assert ncorrect==summary['scopes'][scope]['methods'][m]['correct'] and n==summary['scopes'][scope]['methods'][m]['questions']
   metrics[m]=dict(correct=ncorrect,n=n,accuracy=ncorrect/n)
   keys=['prefix_instrumented_wall_seconds','required_shared_judger_probe_seconds','selector_and_gather_seconds','independent_tensor_audit_seconds','position_io_seconds','judger_seconds','reconstructed_standalone_seconds']
   timings[m]=dict(n=n,sums={k:sum(r['timing'][k] for r in selected) for k in keys},means={k:statistics.mean(r['timing'][k] for r in selected) for k in keys},median_standalone_seconds=statistics.median(r['timing']['reconstructed_standalone_seconds'] for r in selected),generated_tokens=sum(r['generated_tokens'] for r in selected))
   for r in selected:
    t=r['timing'];expected=sum(t[k] for k in ['prefix_instrumented_wall_seconds','required_shared_judger_probe_seconds','selector_and_gather_seconds','judger_seconds'])
    assert abs(expected-t['reconstructed_standalone_seconds'])<1e-6 and t['reconstructed'] and t['cuda_sync_both_boundaries']
  result['accuracy'][scope]=metrics;result['timing'][scope]=timings
 shared={}
 for r in data:
  rid=tuple(r['shared_request_ids']);value=tuple(r['timing'][k] for k in ['prefix_instrumented_wall_seconds','required_shared_judger_probe_seconds'])
  assert rid not in shared or shared[rid]==value;shared[rid]=value
 assert len(shared)==227
 result['shared']=dict(actual_prefix_groups=227,reason='225 main-run prefixes + 2 pilot prefixes; two reused T/G pairs and new R recompute their own prefix',prefix_wall_seconds=sum(v[0] for v in shared.values()),judger_probe_seconds=sum(v[1] for v in shared.values()))
 branch=sum(sum(r['timing'][k] for k in ['selector_and_gather_seconds','independent_tensor_audit_seconds','position_io_seconds','judger_seconds']) for r in data)
 result['shared']['tracked_components_seconds_without_shared_double_count']=branch+sum(sum(v) for v in shared.values())
 for phase in ['run','reused_pilot','prior_A100_failure']:
  base=BASE/phase;attempts=[a for a in (base/'attempts').iterdir() if a.is_dir()];starts=[int(a.name.rsplit('_',1)[1])/1e9 for a in attempts];events=[];loads=[];repeat=[]
  for a in attempts:
   for n in ['trace.jsonl','repeat_checks.jsonl']:
    f=a/n
    if f.exists():check(f)
   events+=rows(a/'trace.jsonl');repeat+=rows(a/'repeat_checks.jsonl')
   for n in ['exit.json','failure.json','model_ready.json']:
    f=a/n
    if f.exists():check(f);v=load(f);events.append(v)
    if f.exists() and n=='model_ready.json':loads.append(v['model_load_hash_prepare_seconds'])
  end=max(v['unix'] for v in events if 'unix' in v);start=min(starts)
  result['phases'][phase]=dict(recorded_attempt_start_unix=start,latest_recorded_event_unix=end,start_Beijing=stamp(start),end_Beijing=stamp(end),elapsed_seconds=end-start,worker_loads=len(loads),sum_model_hash_load_prepare_seconds=sum(loads),instrumented_repeat_count=len(repeat),sum_instrumented_repeat_wall_seconds=sum(v['actual_repeat_instrumented_wall'] for v in repeat))
 q=load(run/'queue.json');workers=load(run/'workers.json');start=result['phases']['run']['recorded_attempt_start_unix'];last_commit=max(t['ended'] for t in q['tasks']);exits=[load(f)['unix'] for f in (run/'attempts').glob('*/exit.json')]
 result['production']=dict(first_worker_attempt_start_Beijing=stamp(start),last_task_commit_Beijing=stamp(last_commit),last_worker_exit_Beijing=stamp(max(exits)),to_last_commit_seconds=last_commit-start,to_last_worker_exit_seconds=max(exits)-start,worker_count=len(workers),initial_GPU1_workers=4,later_GPU0_workers=4)
 allstart=min(v['recorded_attempt_start_unix'] for v in result['phases'].values());result['all_stages_chronological_window']=dict(start_Beijing=stamp(allstart),end_Beijing=stamp(max(exits)),elapsed_seconds=max(exits)-allstart,includes_gaps_and_failed_prefix_setup=True)
 out=PRIVATE/'retrospective_timing_20261109';out.mkdir(exist_ok=True);(out/'timing_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 report=['# A100 GPQA225：末次交接T/R/G结果与耗时复盘','','离线核对已验证归档中的675条结果（本run新增671、A100两题pilot复用4），使用原冻结解析器重算正确性，全部与冻结final_summary一致；未调用模型、未改原答。','','| 范围 | T | R | G |','|---|---:|---:|---:|']
 for scope in subsets:report.append('|'+scope+'|'+'|'.join(f"{result['accuracy'][scope][m]['correct']}/{result['accuracy'][scope][m]['n']} ({result['accuracy'][scope][m]['accuracy']:.2%})" for m in ['T','R','G'])+'|')
 report+=['','主比较R对T净多对10题，精确双侧McNemar p=0.220328；R对G各114题但各自独有正确24题，不构成等价结论。Diamond是225题的子集。','','| 方法（各225题） | Judger生成累计 | 选择+gather累计 | 重建单臂总计 | 平均单臂总时间 | 平均生成token |','|---|---:|---:|---:|---:|---:|']
 for m in ['T','R','G']:
  z=result['timing']['all225'][m];report.append(f"|{m}|{duration(z['sums']['judger_seconds'])}|{duration(z['sums']['selector_and_gather_seconds'])}|{duration(z['sums']['reconstructed_standalone_seconds'])}|{duration(z['means']['reconstructed_standalone_seconds'])}|{z['generated_tokens']/225:.1f}|")
 production=result['production'];report+=['',f"主run实际执行窗口：北京时间 {production['first_worker_attempt_start_Beijing']} 至最后worker退出 {production['last_worker_exit_Beijing']}，共 **{duration(production['to_last_worker_exit_seconds'])}**；从首worker attempt目录命名时间起算，含本run模型加载及并发运行，不含此前pilot/前缀失败准备和最终归档复核。最后一题提交窗口为{duration(production['to_last_commit_seconds'])}。",'',f"若从前置A100前缀失败的首worker启动记录算起，包含pilot、间隔和准备的整个日历窗口为 **{duration(result['all_stages_chronological_window']['elapsed_seconds'])}**，不是净推理时间。",'', '| 阶段 | worker份数 | 模型SHA/加载/准备累计 | 记录的逐题执行累计 |','|---|---:|---:|---:|']
 for phase,z in result['phases'].items():report.append(f"|{phase}|{z['worker_loads']}|{duration(z['sum_model_hash_load_prepare_seconds'])}|{duration(z['sum_instrumented_repeat_wall_seconds']) if z['instrumented_repeat_count'] else '未产生完整逐题记录'}|")
 report+=['','计时口径：重建单臂总时间=共享prefix墙钟+共享Judger probe+本臂selector/gather+本臂Judger；不是三种方法独立运行的实测延迟，也没有包含独立张量审计、位置写盘等所有管理开销。每方法表内共享prefix重复计入，是假设该臂单独执行的重建口径；不可三列相加当整轮实际耗时。',f"共享RID去重后实际prefix组227（主run225组+pilot2组）；prefix累计{duration(result['shared']['prefix_wall_seconds'])}、共享Judger probe累计{duration(result['shared']['judger_probe_seconds'])}。全部答案已计时组件去共享重复后的累计为{duration(result['shared']['tracked_components_seconds_without_shared_double_count'])}；仍包含并发重叠，非GPU忙碌小时或实际墙钟。",'','当时先GPU1四进程、后增加GPU0四进程，且中途纠正CPU亲和性分配；这是该历史run的时间记录，不能当硬件峰值基准或直接预测当前新run速度。','', '模型加载单列，勿再加到主run实际窗口上。4条pilot答案使用其真实历史计时，不伪装为本run新生成。前置失败没有完整repeat计时，不能把缺失记录写成零成本。','', '最终归档SHA：0e54558e885eeafb998ef90a5a70fd5d7d0864802ddb15c1165bf7657ee187c4。输入逐文件SHA核对及精确秒数见同目录timing_summary.json。']
 text='\n'.join(report)+'\n';(out/'timing_report.md').write_text(text,encoding='utf-8');(HERE/'timing_report_20261109.md').write_text(text,encoding='utf-8')
 print(json.dumps(dict(passed=True,accuracy_consistent=True,keys=675,production_elapsed=duration(production['to_last_worker_exit_seconds']),all_stage_window=duration(result['all_stages_chronological_window']['elapsed_seconds']),methods={m:{'judger':duration(result['timing']['all225'][m]['sums']['judger_seconds']),'selection':duration(result['timing']['all225'][m]['sums']['selector_and_gather_seconds']),'standalone_reconstructed':duration(result['timing']['all225'][m]['sums']['reconstructed_standalone_seconds']),'mean_standalone':duration(result['timing']['all225'][m]['means']['reconstructed_standalone_seconds'])} for m in ['T','R','G']},report=str(out/'timing_report.md')),ensure_ascii=False))
if __name__=='__main__':main()
