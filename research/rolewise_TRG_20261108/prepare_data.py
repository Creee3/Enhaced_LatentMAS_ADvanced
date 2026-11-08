"""Pin public dataset bytes and choose fixed ID-hash cohorts; no model calls."""
import hashlib,json
from pathlib import Path
import requests
import pyarrow.parquet as pq
PRIVATE=Path('D:/liuxin_workspace/ICML2027/advanced_MOC/experiments/latentmas_rolewise_TRG576_20261108')
SALT='rolewise_TRG576_s42_20261108'
def digest(b):return hashlib.sha256(b).hexdigest()
def rank(dataset,qid):return digest((SALT+'|'+dataset+'|'+str(qid)).encode())
def fetch(repo,rev,name):
 url=f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}'
 out=PRIVATE/'source_data'/repo.replace('/','__')/name;out.parent.mkdir(parents=True,exist_ok=True)
 if not out.exists():
  response=requests.get(url,timeout=90);response.raise_for_status();out.write_bytes(response.content)
 info=requests.get(f'https://huggingface.co/api/datasets/{repo}/revision/{rev}',params={'blobs':'true'},timeout=30);info.raise_for_status();meta=info.json();assert meta['sha']==rev
 member=next(x for x in meta['siblings'] if x['rfilename']==name);sha=digest(out.read_bytes());expected=member.get('lfs',{}).get('sha256')
 if expected:assert sha==expected
 return pq.read_table(out).to_pylist(),dict(repository=repo,revision=rev,file=name,sha256=sha,bytes=out.stat().st_size,official_lfs_sha_verified=bool(expected),url=url)
def main():
 PRIVATE.mkdir(exist_ok=True)
 assert not (PRIVATE/'questions.json').exists(),'Cohort already frozen; do not silently resample'
 source=PRIVATE/'frozen_source_reference/questions.json';old=json.loads(source.read_text(encoding='utf-8'));items=[]
 for diamond in [False,True]:
  pool=sorted([x for x in old if x['is_diamond']==diamond],key=lambda x:rank('gpqa',x['question_id']))
  assert len(pool)>=48
  for x in pool[:48]:items.append(dict(dataset='gpqa',question_id=x['question_id'],question=x['question'],gold=x['gold'],is_diamond=diamond,category='diamond' if diamond else 'non_diamond',n_options=4))
 mmlu,ms=fetch('TIGER-Lab/MMLU-Pro','b189ec765aa7ed75c8acfea42df31fdae71f97be','data/test-00000-of-00001.parquet')
 categories=sorted({x['category'] for x in mmlu});assert len(categories)==14
 for cat in categories:
  pool=sorted([x for x in mmlu if x['category']==cat],key=lambda x:rank('mmlu_pro',x['question_id']));assert len(pool)>=20
  for x in pool[:20]:
   opts=x['options'];assert 2<=len(opts)<=10 and 0<=x['answer_index']<len(opts) and x['answer']==chr(65+x['answer_index'])
   question=x['question'].strip()+'\n\nChoices:\n'+'\n'.join(f'({chr(65+i)}) {o}' for i,o in enumerate(opts))
   items.append(dict(dataset='mmlu_pro',question_id='mmlu_pro_test_'+str(x['question_id']),question=question,gold=x['answer'],category=cat,n_options=len(opts),is_diamond=False,source_row_id=x['question_id']))
 gsm,gs=fetch('openai/gsm8k','740312add88f781978c0658806c59bc2815b9866','main/test-00000-of-00001.parquet');assert len(gsm)==1319
 for ix in sorted(range(len(gsm)),key=lambda i:rank('gsm8k',i))[:200]:
  x=gsm[ix];assert '####' in x['answer'];gold=x['answer'].rsplit('####',1)[1].strip()
  items.append(dict(dataset='gsm8k',question_id=f'gsm8k_test_{ix:04d}',question=x['question'],gold=gold,category='math',n_options=0,is_diamond=False,source_row_id=ix))
 items.sort(key=lambda x:(['gpqa','mmlu_pro','gsm8k'].index(x['dataset']),rank(x['dataset'],x['question_id'])))
 assert len(items)==576 and len({(x['dataset'],x['question_id']) for x in items})==576
 for i,x in enumerate(items):x['index']=i
 smoke=[next(x['index'] for x in items if x['dataset']=='gpqa' and x['is_diamond']==d) for d in [False,True]]+[next(x['index'] for x in items if x['dataset']==ds) for ds in ['mmlu_pro','gsm8k']]
 raw=(json.dumps(items,ensure_ascii=False,indent=2)+'\n').encode();(PRIVATE/'questions.json').write_bytes(raw)
 counts={ds:sum(x['dataset']==ds for x in items) for ds in ['gpqa','mmlu_pro','gsm8k']}
 manifest=dict(passed=True,selection='SHA256 fixed ID ranking within predeclared strata, no performance selection',salt=SALT,counts=counts,total=576,target_keys=1728,gpqa_diamond=48,gpqa_non_diamond=48,mmlu_categories={c:20 for c in categories},sources=[dict(dataset='gpqa',source_run='TRG225_a100_s42_l10_20261106T155639Z',file_sha256=digest(source.read_bytes()),already_development_data=True),ms,gs],questions_sha256=digest(raw),smoke_indices=smoke,smoke_keys_reusable_only_after_structural_archive_gate=True,model_forward_calls=0)
 (PRIVATE/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(counts=counts,total=576,smoke_cases=len(smoke),sources_verified=True,model_forward_calls=0)))
if __name__=='__main__':main()
