"""CPU-only exact native tokenization and worst-case all-branch billing proof."""
import ast,copy,hashlib,json,runpy,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoTokenizer
from common import HERE,ROOT,SCIENCE,BASE,load,save,sha
from task_adapter import messages,normalized_gold
def main():
 items=load(HERE/'questions.json');manifest=load(HERE/'data_manifest.json');assert len(items)==576 and sha(HERE/'questions.json')==manifest['questions_sha256']
 for entry in manifest['sources']:
  if 'repository' in entry:
   path=HERE/'source_data'/entry['repository'].replace('/','__')/entry['file'];assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256']
  else:assert sha(SCIENCE/'questions.json')==entry['file_sha256']
 lock=load(BASE/'model_lock.json');tok=AutoTokenizer.from_pretrained(lock['path'],local_files_only=True);build=runpy.run_path(str(BASE/'vendor/LatentMAS/prompts.py'))['build_agent_message_sequential_latent_mas']
 tree=ast.parse((BASE/'vendor/LatentMAS/models.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelWrapper');names=['render_chat','prepare_chat_batch'];defs=[copy.deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(defs)==2
 module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='TokenizerOnly',bases=[],keywords=[],body=defs,decorator_list=[])],type_ignores=[]));ns=dict(torch=torch);exec(compile(module,'exact_native_tokenizer_methods','exec'),ns);core=ns['TokenizerOnly']();core.tokenizer=tok;core.device=torch.device('cpu')
 proofs=[];upper=0;max_context=0
 for item in items:
  assert normalized_gold(item) is not None
  if item['dataset']!='gsm8k':assert normalized_gold(item) in [chr(65+i) for i in range(item['n_options'])]
  lengths={};hashes={}
  for role in ['planner','critic','refiner','judger']:
   ms=messages(role,item,build,lock['path']);_,ids,mask,_=core.prepare_chat_batch([ms],add_generation_prompt=True);assert bool(mask.all())
   lengths[role]=ids.shape[-1];hashes[role]=hashlib.sha256(json.dumps(ids[0].tolist(),separators=(',',':')).encode()).hexdigest()
  p,c,r,j=[lengths[x] for x in ['planner','critic','refiner','judger']]
  # Shared planner+first receiver probe; each branch has C/probe/R/probe/J.
  cost=p+4*c+6*r+6*j+70+3*8192;context=p+c+r+j+30+8192
  assert context<=32768
  upper+=cost;max_context=max(max_context,context);proofs.append(dict(index=item['index'],dataset=item['dataset'],question_id=item['question_id'],prompt_lengths=lengths,prompt_token_sha=hashes,upper_positions=cost,max_original_context=context,native_requests=17))
 b=load(ROOT/'cumulative_budget.json');charged=sum(b[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']);assert charged+upper<=115000000
 out=dict(passed=True,total_cases=576,target_keys=1728,methods=['GGG','RRR','TTT'],upper_positions=upper,max_original_context=max_context,model_forward_calls=0,model_constructed=False,tokenizer_source_ast_sha=hashlib.sha256(ast.dump(ast.Module(body=defs,type_ignores=[]),include_attributes=False).encode()).hexdigest(),question_sha=sha(HERE/'questions.json'),model_lock_sha=sha(BASE/'model_lock.json'),budget_before={k:b[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']},maximum_total=charged+upper,proofs=proofs)
 save(HERE/'preflight.json',out);print(json.dumps({k:out[k] for k in ['passed','total_cases','target_keys','upper_positions','max_original_context','model_forward_calls','maximum_total']}))
if __name__=='__main__':main()
