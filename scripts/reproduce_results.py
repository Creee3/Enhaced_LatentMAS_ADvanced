"""Rebuild all sixteen cells from answer-free metrics, without inference."""
import argparse
import csv
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
METHODS=('full','headobf','jive','receiver_attention_sink_recent')

def boolean(value):
    if value not in ('True','False'):raise ValueError('Expected an explicit CSV boolean')
    return value=='True'

def rebuild(path=None):
    path=Path(path) if path else ROOT/'results/gpqa_main448/case_metrics.csv'
    cohort=json.loads((ROOT/'provenance/cohort_ids.json').read_text(encoding='utf-8'))
    membership={(x['dataset'],x['question_id']):x['is_diamond'] for x in cohort}
    assert len(cohort)==len(membership)==448 and sum(membership.values())==198
    expected={(d,q,s,m) for d,q in membership for s in (10,40) for m in METHODS}
    with path.open(encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
    seen=set();items=[]
    for row in raw:
        key=(row['dataset'],row['question_id'],int(row['latent_steps']),row['method'])
        if key in seen:raise ValueError('Duplicate metric key')
        if key not in expected:raise ValueError('Unexpected metric key')
        seen.add(key);diamond=boolean(row['is_diamond'])
        if diamond!=membership[key[:2]]:raise ValueError('Diamond membership mismatch')
        kv=int(row['kv_bytes'])
        if kv<=0 or kv%147456:raise ValueError('Invalid BF16 KV bytes')
        if key[3]!='full' and kv!=(130 if key[2]==10 else 220)*147456:raise ValueError('Compressed budget mismatch')
        if row['source'] not in ('new','reused'):raise ValueError('Unknown result origin')
        items.append(dict(key=key,correct=boolean(row['correct']),diamond=diamond,kv=kv,source=row['source']))
    if seen!=expected:raise ValueError('Missing metric keys')
    assert sum(x['source']=='new' for x in items)==2164
    assert sum(x['source']=='reused' for x in items)==1420
    cells=[]
    for scope,n in [('main448',448),('Diamond198',198)]:
        for steps in (10,40):
            subset=[x for x in items if x['key'][2]==steps and (scope=='main448' or x['diamond'])]
            fullbytes=sum(x['kv'] for x in subset if x['key'][3]=='full')
            for method in METHODS:
                group=[x for x in subset if x['key'][3]==method];assert len(group)==n
                correct=sum(x['correct'] for x in group);total=sum(x['kv'] for x in group);retention=100*total/fullbytes
                cells.append(dict(scope=scope,latent_steps=steps,method=method,n=n,correct=correct,accuracy_pct=100*correct/n,mean_kv_bytes=total/n,total_kv_bytes=total,kv_retention_pct=retention,kv_reduction_pct=100-retention))
    return cells

def check(cells):
    summary=json.loads((ROOT/'results/gpqa_main448/summary.json').read_text(encoding='utf-8'))
    assert summary['total_unique']==3584 and not summary['AI_scoring'] and not summary['baseline_rerun']
    expected={(x['scope'],x['latent_steps'],x['method']):x for x in summary['cells']}
    assert len(cells)==len(expected)==16
    for cell in cells:
        old=expected[cell['scope'],cell['latent_steps'],cell['method']]
        for name,value in cell.items():
            if isinstance(value,float):assert math.isclose(value,old[name],rel_tol=1e-12,abs_tol=1e-8),(name,value,old[name])
            else:assert value==old[name],(name,value,old[name])

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--check',action='store_true');ap.add_argument('--output',type=Path);args=ap.parse_args()
    cells=rebuild()
    if args.check:check(cells)
    if args.output:args.output.write_text(json.dumps(cells,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,unique=3584,cells=len(cells),summary_checked=args.check,model_calls=0)))

if __name__=='__main__':main()
