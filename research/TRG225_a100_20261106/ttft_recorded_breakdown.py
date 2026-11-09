"""Recorded timing decomposition; deliberately never invents missing TTFT."""
import importlib.util,json,statistics,hashlib
from pathlib import Path
ROOT=Path('D:/liuxin_workspace/ICML2027/advanced_MOC/experiments')
OLD=ROOT/'latentmas_TRG225_a100_20261106/completed_verified_0e54558e885e'
NEW=ROOT/'latentmas_rolewise_TRG576_20261108/verified_formal_final_438ac46bb046'

def rows(p):return [json.loads(b) for b in p.read_bytes().split(b'\n') if b.strip()]
def main():
 ids={q['question_id'] for q in json.loads((NEW/'questions.json').read_text()) if q['dataset']=='gpqa'}
 a=rows(OLD/'run/reused_results.jsonl')+[r for p in (OLD/'run/attempts').glob('*/results.jsonl') for r in rows(p)]
 b=[r for p in (NEW/'attempts').glob('*/results.jsonl') for r in rows(p)]
 selected=[r for r in a if r['question_id'] in ids]+[r for r in b if r['dataset']=='gpqa' and r['question_id'] in ids];assert len(selected)==576
 result=dict(passed=True,model_calls=0,same_questions=96,exact_TTFT_available=False,reason='Engine has only per-request elapsed and throttled one-second native_forward_progress; no synchronized first-sampled-token event or complete request scheduling/tokenization timestamps',methods={})
 order=['T','TTT','R','RRR','G','GGG']
 for m in order:
  rs=[r for r in selected if r['method']==m];assert len(rs)==96;assert {r['judger_receipt']['initial_past_positions'] for r in rs}=={130}
  times=[r['timing'] for r in rs];pre=[t['reconstructed_standalone_seconds']-t['judger_seconds'] for t in times];sel=[t['selector_and_gather_seconds'] if len(m)==1 else sum(c['selector_and_gather_seconds'] for c in t['components']) for t in times]
  z=dict(n=96,mean_pre_Judger_seconds=statistics.mean(pre),mean_TRG_selector_gather_seconds=statistics.mean(sel),mean_other_pre_Judger_seconds=statistics.mean(x-y for x,y in zip(pre,sel)),mean_Judger_seconds=statistics.mean(t['judger_seconds'] for t in times),mean_reconstructed_total_seconds=statistics.mean(t['reconstructed_standalone_seconds'] for t in times),mean_generated_tokens=statistics.mean(r['generated_tokens'] for r in rs),aggregate_generated_tokens_per_Judger_wall_second=sum(r['generated_tokens'] for r in rs)/sum(t['judger_seconds'] for t in times),initial_Judger_past_rows=130)
  if len(m)==1:
   z['available_preparation_components']=dict(latent_P_C_R_seconds=statistics.mean(sum(c['latent_seconds'] for c in t['prefix_components']) for t in times),early_receiver_probes_seconds=statistics.mean(sum(c.get('required_probe_seconds',0) for c in t['prefix_components']) for t in times),early_original_receiver_compression_seconds=statistics.mean(sum(c.get('compression_outer_seconds',0) for c in t['prefix_components']) for t in times),shared_Judger_probe_seconds=statistics.mean(t['required_shared_judger_probe_seconds'] for t in times))
  else:
   z['available_preparation_components']=dict(shared_Planner_first_Critic_probe_seconds=statistics.mean(t['shared_prefix_and_first_probe_seconds'] for t in times),Critic_Refiner_latent_seconds=statistics.mean(sum(c.get('receiver_latent_seconds',0) for c in t['components']) for t in times),remaining_two_probes_seconds=statistics.mean(sum(c['required_probe_seconds'] for c in t['components'][1:]) for t in times))
  known=sum(z['available_preparation_components'].values())+z['mean_TRG_selector_gather_seconds'];z['mean_remaining_preparation_overhead_seconds']=z['mean_pre_Judger_seconds']-known
  assert z['mean_remaining_preparation_overhead_seconds']>-1e-6
  result['methods'][m]=z
 wanted={r['judger_receipt']['reservation_id'] for r in selected};first={};starts={}
 for folder in [OLD/'run/attempts',OLD/'reused_pilot/attempts',NEW/'attempts']:
  for path in folder.glob('*/trace.jsonl'):
   for line in path.read_bytes().split(b'\n'):
    if b'"request_started"' not in line and b'"native_forward_progress"' not in line:continue
    x=json.loads(line);rid=x.get('reservation_id')
    if rid not in wanted:continue
    if x['event']=='request_started':starts[rid]=x['unix']
    elif rid not in first:first[rid]=dict(unix=x['unix'],completed_forwards=x['completed_forwards'])
 observed=[x['completed_forwards'] for x in first.values()];result['trace_sampling_evidence']=dict(Judger_requests=len(wanted),requests_with_progress=len(first),median_completed_forwards_at_first_progress=statistics.median(observed),min_completed_forwards_at_first_progress=min(observed),max_completed_forwards_at_first_progress=max(observed),first_progress_is_not_first_token=True)
 result['differences']={}
 for m in ['T','R','G']:
  old,new=result['methods'][m],result['methods'][m*3];delta=old['mean_reconstructed_total_seconds']-new['mean_reconstructed_total_seconds'];judge_delta=old['mean_Judger_seconds']-new['mean_Judger_seconds'];result['differences'][m*3+'_vs_'+m]=dict(total_seconds_saved_observed=delta,Judger_seconds_saved_observed=judge_delta,Judger_fraction_of_observed_difference=judge_delta/delta,causal_attribution_available=False)
 report=['# 同题GPQA96：记录的耗时拆分与TTFT缺口','','没有精确TTFT记录。当前Engine只有每请求整体elapsed_seconds及最多每秒一次native_forward_progress，没有首个采样token完成时间戳或逐请求排队/分词/采样事件。不能把首个forward进度或Judger前准备称为TTFT。','','| 方法 | 进入Judger前准备 | 其中T/R/G选位+gather | 其余前缀/probe/开销 | Judger全生成 | 重建整臂 | 平均输出token | token/Judger秒 |','|---|---:|---:|---:|---:|---:|---:|---:|']
 for m,z in result['methods'].items():report.append(f"|{m}|{z['mean_pre_Judger_seconds']:.2f}s|{z['mean_TRG_selector_gather_seconds']:.2f}s|{z['mean_other_pre_Judger_seconds']:.2f}s|{z['mean_Judger_seconds']:.2f}s|{z['mean_reconstructed_total_seconds']:.2f}s|{z['mean_generated_tokens']:.1f}|{z['aggregate_generated_tokens_per_Judger_wall_second']:.2f}|")
 report+=['','旧T/R/G的前两次原receiver压缩计在“其余”列；新TTT/RRR/GGG的三次新选择均计入选位列，不能将这两列直接作纯算法消融。','', '两轮所有这96题的Judger初始历史KV都为130行。TTT相对T的总差中约88%落在Judger；RRR相对R约88%；GGG相对G约96%。这说明“多次压缩更快”不是由这张表支持的结论。运行配置、CPU亲和、并发/MPS与输出长度均混杂，不能定量归因于其中一项。RRR输出token略多于R，依然Judger更快，单靠长度不能解释。','','阶段组件细表（平均秒）如下；标注shared的成本在原run实际共享，仅独立单臂重建中计入。三跳first probe已在shared内，不能再把components[0].required_probe重复加一次。']
 for m,z in result['methods'].items():report+=['',m+': '+', '.join(k+'='+f'{v:.4f}' for k,v in z['available_preparation_components'].items())+f", unallocated_preparation_overhead={z['mean_remaining_preparation_overhead_seconds']:.4f}"]
 report+=['',f"Trace证据：{len(first)}/{len(wanted)}个Judger请求有进度；首条进度时已完成forward中位数{statistics.median(observed):.1f}，范围{min(observed)}–{max(observed)}。进度事件没有保证首token采样或CUDA同步，不能还原真实TTFT。",'', '完整warm请求TTFT应分：调度/等待→分词及准备→Planner latent→第一次probe及选位→Critic latent→第二次probe及选位→Refiner latent→第三次probe及选位→Judger prefill→首token采样/可用。当前精确缺少调度、Judger prefill/采样边界、首token事件。模型冷加载另记，不能混入warm TTFT。', '同题分支原本依次执行，独立单臂重建不含排在前面的其他臂等待。当前冻结实验不加热插埋点；未来独立性能验证应在首个采样token可用处记录单调时钟/CUDA同步边界，再在相同MPS/并发/CPU/固定生成长度下作对照；不为取数重跑科学结果。']
 out=ROOT/'latentmas_rolewise_TRG576_20261108/comparison_terminal_all96_20261109';(out/'ttft_recorded_breakdown.json').write_text(json.dumps(result,indent=2),encoding='utf-8');text='\n'.join(report)+'\n';(out/'ttft_recorded_breakdown.md').write_text(text,encoding='utf-8');Path(__file__).with_name('ttft_breakdown_20261109.md').write_text(text,encoding='utf-8')
 print(json.dumps(dict(passed=True,exact_TTFT_available=False,trace_sampling=result['trace_sampling_evidence'],differences=result['differences'],report=str(out/'ttft_recorded_breakdown.md'))))
if __name__=='__main__':main()
