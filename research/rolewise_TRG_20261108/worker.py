"""Native B1 GGG/RRR/TTT worker; shared computation stops at the first fork."""
import argparse,ast,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
from common import HERE,ROOT,SCIENCE,BASE,METHODS,load,save,append,sha,verify,identity,claim,commit,hold
sys.path[:0]=[str(HERE),str(SCIENCE),str(BASE)]
from runtime_common import Ledger,seed_all
from task_adapter import args_for,messages,parse_answer
from relay_select import select_relay

def rng_sha(torch):
 return hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()+b''.join(x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all())).hexdigest()
def kv_sha(torch,xs):
 h=hashlib.sha256()
 for pair in xs:
  for t in pair:h.update(str((tuple(t.shape),str(t.dtype))).encode());h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
 return h.hexdigest()

def main(phase,slot,out):
 task=None;out.mkdir(parents=True,exist_ok=False);save(out/'loading.json',dict(identity=identity(os.getpid()),phase=phase,slot=slot,unix=time.time()))
 try:
  source=verify();assert load(HERE/'plan.json')['methods']==METHODS
  assert os.environ.get('LATENTMAS_SHARED_LEDGER')==str(ROOT/'cumulative_budget.json') and (ROOT/'cumulative_budget.json').exists()
  import torch,numpy as np,torch.nn.functional as F
  import experiment_worker as ew
  from stage_ops import pairs,make_cache,clone_cache,observe
  from author_parser import extract_gsm8k_answer
  torch.set_num_threads(1);allowed=sorted(os.sched_getaffinity(0))
  try:os.sched_setaffinity(0,{allowed[slot%len(allowed)]})
  except OSError:os.sched_setaffinity(0,os.sched_getaffinity(0))
  original_check=Ledger.check
  def check(self):
   original_check(self)
   if (HERE/'STOP').exists():raise RuntimeError('user STOP requested')
  Ledger.check=check
  start=time.monotonic();runner=ew.Runner(str(out),torch.device('cuda:0'));engine=runner.engine;model=engine.model
  engine.config.update(experiment_id=HERE.name,request_timeout_seconds=1800);engine.phase=HERE.name+':'+phase
  matrix,target=engine.core._latent_realign_matrices[id(model)];model_files=[x for x in load(BASE/'model_lock.json')['files']]
  ready=dict(identity=identity(os.getpid()),source=source,phase=phase,slot=slot,model_initialization_seconds=time.monotonic()-start,weight_files=len(model_files),bf16_cuda=all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters()),parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),identity_norm=torch.equal(matrix,torch.eye(matrix.shape[0],dtype=matrix.dtype,device=matrix.device)),target_norm=float(target),affinity=sorted(os.sched_getaffinity(0)),unix=time.time())
  assert ready['weight_files']==15 and ready['bf16_cuda'] and ready['parameter_bytes']==16381470720 and ready['identity_norm'] and abs(ready['target_norm']-1.3758320808410645)<1e-6
  save(out/'model_ready.json',ready)
  items=load(HERE/'questions.json');pre={x['index']:x for x in load(HERE/'preflight.json')['proofs']};current_item=None;expected_role=[None]
  native_builder=ew.PROMPTS['build_agent_message_sequential_latent_mas']
  def prompt(role,question,args):
   assert current_item is not None and question==current_item['question'];expected_role[0]=role
   return messages(role,current_item,native_builder,runner.model_name)
  ew.stage_prompt=prompt;ew.args_for=args_for;native_prepare=engine.prepare
  def prepare(ms):
   p,ids,mask=native_prepare(ms);proof=pre[current_item['index']];role=expected_role[0]
   assert ids.shape[-1]==proof['prompt_lengths'][role]
   assert hashlib.sha256(json.dumps(ids[0].cpu().tolist(),separators=(',',':')).encode()).hexdigest()==proof['prompt_token_sha'][role]
   return p,ids,mask
  engine.prepare=prepare
  tree=ast.parse((SCIENCE/'cloud_worker.py').read_text());oldmain=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');names=['timed','context','stage','probe','judge'];defs=[n for n in ast.walk(oldmain) if isinstance(n,ast.FunctionDef) and n.name in names];assert len(defs)==5
  ns=dict(globals(),**locals(),current=dict(repeat=1));exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),'frozen_native_stage_probe_judge','exec'),ns)
  ast_sha=hashlib.sha256(ast.dump(ast.Module(body=defs,type_ignores=[]),include_attributes=False).encode()).hexdigest();save(out/'native_functions.json',dict(names=names,ast_sha256=ast_sha,source_file_sha256=sha(SCIENCE/'cloud_worker.py')))
  while not (HERE/'STOP').exists():
   task=claim(phase,out)
   if task is None:break
   verify();current_item=items[task['index']];item=current_item;ns['current']=dict(repeat=1);seed_all(42);torch.cuda.reset_peak_memory_stats();case_start=time.monotonic()
   case_dir=out/'positions'/f"case{item['index']:04d}";case_dir.mkdir(parents=True,exist_ok=True);engine.phase=HERE.name+':'+phase+':shared'
   save(out/'current.json',dict(index=item['index'],dataset=item['dataset'],question_id=item['question_id'],phase='shared_planner'))
   (planner,prec,_),planner_seconds=ns['timed'](lambda:ns['stage'](item,'planner',None,0))
   (captured_first,probe_first),probe_seconds=ns['timed'](lambda:ns['probe'](item,'critic',planner,prec['original_end'],True))
   prefix_sha=kv_sha(torch,pairs(planner));rng=(torch.random.get_rng_state(),torch.cuda.get_rng_state_all());fork_rng_sha=rng_sha(torch)
   common_ids=[prec['reservation_id'],probe_first['reservation_id']];shared_seconds=time.monotonic()-case_start
   order=METHODS[item['index']%3:]+METHODS[:item['index']%3]
   for method in order:
    if method not in task['methods']:continue
    torch.random.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);assert rng_sha(torch)==fork_rng_sha
    assert kv_sha(torch,pairs(planner))==prefix_sha
    mode=method[0];engine.phase=HERE.name+':'+phase+':'+method;arm_start=time.monotonic();cache=planner;mapping=None;stages=[prec];transfers=[];proof_ids=list(common_ids);components=[];position_files=[]
    for hop,receiver in enumerate(['critic','refiner','judger']):
     sender=['planner','critic','refiner'][hop]
     save(out/'current.json',dict(index=item['index'],dataset=item['dataset'],question_id=item['question_id'],method=method,phase='select_for_'+receiver))
     if hop==0:captured=captured_first;probe_receipt=probe_first;probe_time=probe_seconds
     else:
      (captured,probe_receipt),probe_time=ns['timed'](lambda:ns['probe'](item,receiver,cache,stages[-1]['original_end'],True));proof_ids.append(probe_receipt['reservation_id'])
     before_rng=rng_sha(torch);selected,select_seconds=ns['timed'](lambda:select_relay(pairs(cache),mapping,stages,captured,mode));assert rng_sha(torch)==before_rng
     new_pairs,mapping,indices,diagnostics,transfer=selected;compact=make_cache(new_pairs)
     filename=case_dir/(method+f'_handoff{hop+1}.npz');np.savez_compressed(filename,positions=mapping.cpu().numpy(),storage_indices=indices.cpu().numpy())
     entry=dict(file=str(filename.relative_to(out)),file_sha256=sha(filename),position_sha=transfer['position_sha'],shape=list(mapping.shape),handoff=hop+1);position_files.append(entry)
     transfer.update(sender=sender,receiver=receiver,required_probe_RID=probe_receipt['reservation_id'],diagnostics=diagnostics,local_cache_bytes_before=sum(t.numel()*t.element_size() for p in pairs(cache) for t in p));transfers.append(transfer)
     components.append(dict(handoff=hop+1,required_probe_seconds=probe_time,selector_and_gather_seconds=select_seconds,probe_shared_across_methods=hop==0))
     cache=compact;del new_pairs,selected,indices,compact
     if hop<2:
      (cache,rec,_),seconds=ns['timed'](lambda:ns['stage'](item,receiver,cache,stages[-1]['original_end']));stages.append(rec);proof_ids.append(rec['reservation_id']);components[-1]['receiver_latent_seconds']=seconds
     if hop>0:del captured
    judge_rng_sha=rng_sha(torch);save(out/'current.json',dict(index=item['index'],dataset=item['dataset'],question_id=item['question_id'],method=method,phase='judger'))
    (answer,jrec),judge_seconds=ns['timed'](lambda:ns['judge'](item,cache,stages[-1]['original_end'],method));proof_ids.append(jrec['reservation_id'])
    assert len(set(proof_ids))==len(proof_ids)==7 and kv_sha(torch,pairs(planner))==prefix_sha
    pred=parse_answer(item,answer,extract_gsm8k_answer)
    row=dict(dataset=item['dataset'],question_id=item['question_id'],index=item['index'],is_diamond=item['is_diamond'],category=item['category'],method=method,latent_steps=10,seed=42,status='ok',phase=phase,answer_text=answer,prediction=pred,parse_valid=pred is not None,generated_tokens=len(jrec['generated_ids']),token_sha=hashlib.sha256(json.dumps(jrec['generated_ids']).encode()).hexdigest(),judger_receipt=jrec,shared_request_ids=common_ids,request_ids=proof_ids,shared_planner_kv_sha=prefix_sha,fork_rng_sha=fork_rng_sha,judge_rng_sha=judge_rng_sha,prefix_stages=stages,transfers=transfers,position_files=position_files,actual_kv_bytes_by_handoff=[x['actual_kv_bytes'] for x in transfers],actual_cumulative_relay_bytes=sum(x['actual_kv_bytes'] for x in transfers),full_reference_bytes_by_original_positions=[x['full_bytes_from_original_positions'] for x in transfers],source_lock_sha=source['run_lock_sha256'],native_functions_ast_sha=ast_sha,adapter_sha=sha(HERE/'task_adapter.py'),timing=dict(shared_prefix_and_first_probe_seconds=shared_seconds,components=components,judger_seconds=judge_seconds,arm_instrumented_seconds=time.monotonic()-arm_start,reconstructed_standalone_seconds=shared_seconds+time.monotonic()-arm_start,reconstructed_not_independently_timed=True,cuda_sync_both_boundaries=True,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    commit(task,row,out);save(out/'last_commit.json',dict(index=item['index'],method=method,unix=time.time()))
    del cache,mapping,transfers,stages,position_files;gc.collect();torch.cuda.empty_cache()
   append(out/'case_done.jsonl',dict(index=item['index'],seconds=time.monotonic()-case_start,unix=time.time()));del planner,captured_first;gc.collect();torch.cuda.empty_cache();task=None
  save(out/'exit.json',dict(normal=True,phase=phase,identity=identity(os.getpid()),unix=time.time()))
 except BaseException as e:
  save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),identity=identity(os.getpid()),current_task=task,unix=time.time()));hold(task,e);raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['engineering','formal'],required=True);p.add_argument('--slot',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.phase,a.slot,Path(a.out))
