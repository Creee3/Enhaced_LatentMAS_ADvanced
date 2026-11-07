"""Aggregate-only local analysis of a completed, private engineering test."""
import json,statistics as st
from pathlib import Path
PRIVATE=Path('D:/liuxin_workspace/ICML2027/advanced_MOC/experiments/real_kv_batch_20261108')
def main():
 x=json.loads((PRIVATE/'result.json').read_bytes());assert x['passed']
 metrics={};ref={}
 for mode in ['serial','batch']:
  vals=x[mode]['results'];metrics[mode]=dict(total_tokens_s=st.mean(512/(r['end']-r['start']) for r in vals),wall_seconds=[r['end']-r['start'] for r in vals],peak_allocated_GiB=max(q['peak_bytes'] for r in vals for q in r['records'])/2**30)
 for r in x['serial']['results']:
  for rec in r['records']:
   for k,i in enumerate(rec['indices']):
    if i in ref:assert ref[i]==rec['tokens'][k],'Sequential reference not repeatable'
    ref[i]=rec['tokens'][k]
 comparisons=[]
 for mode,runs in [('batch',[x['batch']]),('process',x['processes'])]:
  for item in runs:
   for r in item['results']:
    for rec in r['records']:
     for k,i in enumerate(rec['indices']):
      tok=rec['tokens'][k];same=tok==ref[i]
      comparisons.append(dict(mode=mode,repeat=r['repeat'],packet=i,matched=same,first_different_token=next((p for p,(a,b) in enumerate(zip(tok,ref[i])) if a!=b),None)))
 walls=[];peaks=[]
 for repeat in range(2):
  runs=[p['results'][repeat] for p in x['processes']];walls.append(max(r['end'] for r in runs)-min(r['start'] for r in runs));peaks.append(sum(r['records'][0]['peak_bytes'] for r in runs)/2**30)
 metrics['four_processes']=dict(total_tokens_s=st.mean(512/w for w in walls),wall_seconds=walls,sum_process_peak_allocated_GiB=max(peaks))
 result=dict(passed=True,engineering_only=True,metrics=metrics,batch_vs_four_process_throughput_ratio=metrics['batch']['total_tokens_s']/metrics['four_processes']['total_tokens_s'],token_comparisons=comparisons,actual_positions=x['actual_positions'],all_tokens_match=all(c['matched'] for c in comparisons))
 (PRIVATE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
