"""Tokenizer-only exact positions, natural paraphrase matching and total budget."""
import ast,copy,hashlib,itertools,json,sys,runpy
from pathlib import Path
from common import HERE,ROOT,SCIENCE,BASE,OLD,load,save,sha
from prompts import ROLES,instruction_candidates,intervention_messages
def digest(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
def main():
 import torch
 from transformers import AutoTokenizer
 sys.path.insert(0,str(OLD));from task_adapter import messages
 lock=load(BASE/'model_lock.json');tok=AutoTokenizer.from_pretrained(lock['path'],local_files_only=True)
 tree=ast.parse((BASE/'vendor/LatentMAS/models.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ModelWrapper')
 defs=[copy.deepcopy(n) for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['render_chat','prepare_chat_batch']]
 module=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='TokenizerOnly',bases=[],keywords=[],body=defs,decorator_list=[])],type_ignores=[]))
 ns=dict(torch=torch);exec(compile(module,'frozen_native_tokenizer','exec'),ns);core=ns['TokenizerOnly']();core.tokenizer=tok;core.device=torch.device('cpu')
 def encode(ms):
  ps,ids,mask,_=core.prepare_chat_batch([ms],add_generation_prompt=True);assert bool(mask.all());return ps[0],ids[0].tolist()
 candidates=instruction_candidates();lengths={r:{} for r in ROLES};placeholder={'question':'What is 2 plus 2?'}
 for r in ROLES:
  for text in candidates[r]:
   _,ids=encode(intervention_messages(placeholder,text));lengths[r].setdefault(len(ids),[]).append(text)
 shared=set.intersection(*(set(n for n,v in lengths[r].items() if len(v)>=2) for r in ROLES))
 assert shared, 'No natural equal-length two-paraphrase set: '+str({r:{n:len(v) for n,v in z.items()} for r,z in lengths.items()})
 target=min(shared);instructions={r:lengths[r][target][:2] for r in ROLES};save(HERE/'matched_instructions.json',dict(instructions=instructions,placeholder_prompt_length=target,natural_no_padding=True))
 build=runpy.run_path(str(OLD/'dependencies/LatentMAS/prompts.py'))['build_agent_message_sequential_latent_mas']
 items=load(HERE/'questions.json');assert len(items)==120 and [x['index'] for x in items]==list(range(120))
 proofs=[];upper=0;max_context=0
 for item in items:
  native=[]
  for role in ROLES:
   _,ids=encode(messages(role,item,build,lock['path']));native.append(dict(role=role,prompt_len=len(ids),token_sha=digest(ids)))
  branches=[]
  for role in ROLES:
   for variant,text in enumerate(instructions[role]):
    rendered,ids=encode(intervention_messages(item,text));encoded=tok(rendered,add_special_tokens=False,return_offsets_mapping=True);assert encoded['input_ids']==ids
    start=rendered.index(item['question']);end=start+len(item['question']);ix=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if a>=start and b<=end and b>a]
    assert len(ix)>=32 and ix==list(range(ix[0],ix[-1]+1)),(item['index'],len(ix))
    branches.append(dict(role=role,variant=variant,prompt_len=len(ids),token_sha=digest(ids),question_start=ix[0],question_end=ix[-1]+1,question_token_sha=digest(ids[ix[0]:ix[-1]+1])))
  assert len({(x['prompt_len'],x['question_start'],x['question_end'],x['question_token_sha']) for x in branches})==1,'Unmatched shared question positions'
  histories=[0,native[0]['prompt_len']+10,sum(x['prompt_len']+10 for x in native[:2])]
  cost=sum(x['prompt_len']+10 for x in native)+3*sum(x['prompt_len']+10 for x in branches)
  context=max(sum(x['prompt_len']+10 for x in native),max(histories)+branches[0]['prompt_len']+10);assert context<=32768
  upper+=cost;max_context=max(max_context,context);proofs.append(dict(index=item['index'],native=native,branches=branches,histories=histories,upper_positions=cost,request_count=21,max_context=context))
 budget=load(ROOT/'cumulative_budget.json');charged=sum(budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens'])
 result=dict(passed=charged+upper<=115000000,model_forward_calls=0,questions=120,total_requests=2520,upper_positions=upper,max_context=max_context,maximum_total=charged+upper,budget_before={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']},question_sha=sha(HERE/'questions.json'),proofs=proofs)
 save(HERE/'preflight.json',result);print(json.dumps({k:result[k] for k in ['passed','model_forward_calls','questions','total_requests','upper_positions','max_context','maximum_total']}));assert result['passed'],'Budget proof failed'
if __name__=='__main__':main()
