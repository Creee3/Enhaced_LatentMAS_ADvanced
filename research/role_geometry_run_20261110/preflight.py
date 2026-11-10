"""Exact native tokenizer-only budget proof; never instantiate a model."""
import ast,copy,hashlib,json,runpy,sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
from common import HERE,ROOT,SCIENCE,BASE,load,save,sha,METHODS
from task_adapter import messages,normalized_gold

def main():
 items=load(HERE/'questions.json');plan=load(HERE/'plan.json');manifest=load(HERE/'data_manifest.json')
 assert len(items)==120 and manifest['questions_sha256']==sha(HERE/'questions.json')
 lock=load(BASE/'model_lock.json');tok=AutoTokenizer.from_pretrained(lock['path'],local_files_only=True)
 build=runpy.run_path(str(HERE/'dependencies/LatentMAS/prompts.py'))['build_agent_message_sequential_latent_mas']
 tree=ast.parse((BASE/'vendor/LatentMAS/models.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelWrapper');defs=[copy.deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['render_chat','prepare_chat_batch']]
 module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='TokenizerOnly',bases=[],keywords=[],body=defs,decorator_list=[])],type_ignores=[]));ns=dict(torch=torch);exec(compile(module,'frozen_native_tokenizer','exec'),ns);core=ns['TokenizerOnly']();core.tokenizer=tok;core.device=torch.device('cpu')
 proofs=[];upper=0;maxctx=0;M=len(METHODS)
 for item in items:
  lengths={};hashes={}
  for role in ['planner','critic','refiner','judger']:
   ms=messages(role,item,build,lock['path']);_,ids,mask,_=core.prepare_chat_batch([ms],add_generation_prompt=True);assert bool(mask.all())
   lengths[role]=ids.shape[-1];hashes[role]=hashlib.sha256(json.dumps(ids[0].tolist(),separators=(',',':')).encode()).hexdigest()
  p,c,r,j=[lengths[x] for x in ['planner','critic','refiner','judger']]
  # One shared Planner; 12 independent C/R latent pairs + Judger generations.
  # First C full-state probe is shared, then per-trajectory R/J predeletion probes.
  # Common Full-state receiver intervention additionally probes 11 compact variants
  # at hops2/3. Hop1 uses actual first native prefills for the same comparison.
  chain=p+10+M*(c+10+r+10+j+8192)
  conditional=c+M*(r+j)
  common=(M-1)*(r+j)
  cost=chain+conditional+common
  context=p+c+r+j+30+8192;assert context<=32768
  upper+=cost;maxctx=max(maxctx,context)
  proofs.append(dict(index=item['index'],dataset_group=item['dataset_group'],question_id=item['question_id'],prompt_lengths=lengths,prompt_token_sha=hashes,upper_positions=cost,chain_upper=chain,conditional_probe_upper=conditional,common_state_probe_upper=common,max_original_context=context,native_requests=1+3*M+1+2*M+2*(M-1)))
 budget=load(ROOT/'cumulative_budget.json');charged=sum(budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']);passed=charged+upper<=115000000
 out=dict(passed=passed,total_cases=120,target_keys=1440,methods=METHODS,upper_positions=upper,max_original_context=maxctx,model_forward_calls=0,model_constructed=False,maximum_total=charged+upper,headroom=115000000-charged,question_sha=sha(HERE/'questions.json'),model_lock_sha=sha(BASE/'model_lock.json'),tokenizer_source_ast_sha=hashlib.sha256(ast.dump(ast.Module(body=defs,type_ignores=[]),include_attributes=False).encode()).hexdigest(),budget_before={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']},proofs=proofs)
 save(HERE/'preflight.json',out);print(json.dumps({k:out[k] for k in ['passed','total_cases','target_keys','upper_positions','max_original_context','maximum_total','headroom','model_forward_calls']}));assert passed,'Budget proof failed; no model was constructed'
if __name__=='__main__':main()
