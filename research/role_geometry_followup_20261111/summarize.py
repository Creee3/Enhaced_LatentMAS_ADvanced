"""Final-only per-dataset, per-question diagnostic summary; zero model calls."""
import json,sys,numpy as np
from common import HERE,load,save,records,rows,verify
def summary():
 assert load(HERE/'formal_audit_verified.json')['passed'] and load(HERE/'formal_archive_verified.json')['passed'];verify()
 rr=records();assert len(rr)==120;groups={}
 for row in rr:groups.setdefault(row['dataset_group'],[]).append(row)
 rng=np.random.default_rng(42);report=dict(complete=True,questions=120,requests=2520,model_forward_calls=0,independent_unit='question',exploratory=True,no_pooled_primary_effect=True,no_accuracy_evaluated=True,bootstrap=dict(replicates=1000,seed=42),datasets={})
 md=['# 角色任务干预与Full位置补充诊断','', '每数据组独立；主指标为角色间子空间距离减同角色改写距离，正值表示角色间比同角色表述间更分离。仅解释为固定历史/位置下的任务指令效应。K/V、prompt/latent及三历史分别报告。区间为按题bootstrap，不进行显著性或多重比较结论。','']
 for group,rrg in groups.items():
  ds=dict(n=len(rrg),contexts={});assert len(rrg)==20
  md+=['## '+group,'','|历史条件|KV|类型|同角色改写距离|不同角色距离|差值|差值95%区间|','|---|---|---|---:|---:|---:|---|']
  for c in range(3):
   ctx={}
   for kind in ['prompt','latent']:
    data=np.array([[r['contexts'][c]['effects'][kind][k] for k in ['within_role','between_role','between_minus_within']] for r in rrg]);assert data.shape==(20,3,2)
    ci=np.quantile(data[rng.integers(0,20,size=(1000,20))].mean(1),[.025,.975],axis=0);means=data.mean(0)
    ctx[kind]={kv:dict(within_mean=float(means[0,i]),between_mean=float(means[1,i]),effect_mean=float(means[2,i]),effect_ci95=[float(ci[0,2,i]),float(ci[1,2,i])]) for i,kv in enumerate(['K','V'])}
    for i,kv in enumerate(['K','V']):md.append(f'|{c}|{kv}|{kind}|{means[0,i]:.5f}|{means[1,i]:.5f}|{means[2,i]:.5f}|[{ci[0,2,i]:.5f}, {ci[1,2,i]:.5f}]|')
   ds['contexts'][str(c)]=ctx
  ds['full_windows']={}
  for stage,role in enumerate(['planner','critic','refiner']):
   x=np.array([r['native'][stage]['metrics_mean'] for r in rrg]);assert x.shape==(20,3,2,3)
   ds['full_windows'][role]=dict(axes=['window_early_middle_late','K_or_V','entropy_rank_participation_rank_numerical_rank'],mean=x.mean(0).tolist(),late_minus_early_entropy_rank_mean=(x[:,2,:,0]-x[:,0,:,0]).mean(0).tolist(),template_included=True)
  report['datasets'][group]=ds;md+=['']
 save(HERE/'final_diagnostics_summary.json',report);(HERE/'final_diagnostics_summary.md').write_text('\n'.join(md),encoding='utf-8');return dict(complete=True,groups=len(groups),questions=120,model_forward_calls=0)
if __name__=='__main__':print(json.dumps(summary()))
