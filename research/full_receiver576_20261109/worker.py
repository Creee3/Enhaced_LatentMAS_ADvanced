"""Frozen-native Full and original 32-per-role receiver, all three handoffs."""
import ast,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
from common import HERE,ROOT,SCIENCE,BASE,METHODS,load,save,append,sha,verify,identity,claim,commit,hold
sys.path[:0]=[str(HERE),str(SCIENCE),str(BASE)]
from runtime_common import Ledger,seed_all
from task_adapter import args_for,messages,parse_answer

def kv_sha(torch,xs):
 h=hashlib.sha256()
 for pair in xs:
  for t in pair:h.update(str((tuple(t.shape),str(t.dtype))).encode());h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
 return h.hexdigest()

def rng_sha(torch):return hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()+b''.join(x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all())).hexdigest()

def relay(cache,mapping,rec,mode,attention,compress,pairs,torch):
 old=pairs(cache);h=old[0][0].shape[-2]-rec['prompt_len']-10
 fresh=torch.arange(rec['original_before'],rec['original_end'],device=old[0][0].device).view(1,1,-1).expand(36,8,-1)
 all_map=fresh.clone() if mapping is None else torch.cat([mapping,fresh],-1)
 result,pos,native=compress(cache,mapping,role=rec['role'],steps=10,prompt_len=rec['prompt_len'],original_before=rec['original_before'],method='full' if mode=='Full' else 'receiver_attention_sink_recent',attention=attention)
 ix=torch.searchsorted(all_map.contiguous(),pos.contiguous())
 for l,(before,after) in enumerate(zip(old,pairs(result))):
  for a,b in zip(before,after):assert torch.equal(torch.gather(a,2,ix[l][None,:,:,None].expand_as(b)),b)
 retained=pos.shape[-1];receipt=dict(method=mode,retained=retained,original_end=rec['original_end'],original_RoPE=True,
  source_k_equal_measured=True,source_v_equal_measured=True,history_kept=h,source_prompt_len=rec['prompt_len'],sender=rec['role'],
  actual_kv_bytes=sum(t.numel()*t.element_size() for p in pairs(result) for t in p),full_bytes_from_original_positions=rec['original_end']*147456,
  native=native,position_sha=hashlib.sha256(pos.cpu().contiguous().numpy().tobytes()).hexdigest())
 assert receipt['actual_kv_bytes']==retained*147456
 return result,pos,ix,receipt

def main(phase,slot,out):
 task=None;out.mkdir(parents=True,exist_ok=False);save(out/'loading.json',dict(identity=identity(os.getpid()),phase=phase,slot=slot,unix=time.time()))
 try:
  source=verify();assert load(HERE/'plan.json')['methods']==METHODS
  assert os.environ.get('LATENTMAS_SHARED_LEDGER')==str(ROOT/'cumulative_budget.json')
  import torch,numpy as np,torch.nn.functional as F
  import experiment_worker as ew
  from stage_ops import pairs,make_cache,clone_cache,observe,compress
  from author_parser import extract_gsm8k_answer
  torch.set_num_threads(1);allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[slot%len(allowed)]})
  original_check=Ledger.check
  def check(self):
   original_check(self)
   if (HERE/'STOP').exists():raise RuntimeError('user STOP requested')
  Ledger.check=check
  start=time.monotonic();runner=ew.Runner(str(out),torch.device('cuda:0'));engine=runner.engine;model=engine.model
  engine.config.update(experiment_id=HERE.name,request_timeout_seconds=1800);engine.phase=HERE.name+':'+phase
  matrix,target=engine.core._latent_realign_matrices[id(model)]
  ready=dict(identity=identity(os.getpid()),source=source,phase=phase,slot=slot,model_initialization_seconds=time.monotonic()-start,weight_files=len(load(BASE/'model_lock.json')['files']),bf16_cuda=all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters()),parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),identity_norm=torch.equal(matrix,torch.eye(matrix.shape[0],dtype=matrix.dtype,device=matrix.device)),target_norm=float(target),unix=time.time())
  assert ready['weight_files']==15 and ready['bf16_cuda'] and ready['parameter_bytes']==16381470720 and ready['identity_norm'] and abs(ready['target_norm']-1.3758320808410645)<1e-6
  save(out/'model_ready.json',ready)
  items=load(HERE/'questions.json');pre={x['index']:x for x in load(HERE/'preflight.json')['proofs']};current_item=None;expected_role=[None];native_builder=ew.PROMPTS['build_agent_message_sequential_latent_mas']
  def prompt(role,question,args):
   assert question==current_item['question'];expected_role[0]=role;return messages(role,current_item,native_builder,runner.model_name)
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
   save(out/'current.json',dict(index=item['index'],phase='shared_planner'))
   (planner,prec,_),_=ns['timed'](lambda:ns['stage'](item,'planner',None,0))
   prefix_sha=kv_sha(torch,pairs(planner));rng=(torch.random.get_rng_state(),torch.cuda.get_rng_state_all());fork_rng=rng_sha(torch);shared_seconds=time.monotonic()-case_start
   order=METHODS[item['index']%2:]+METHODS[:item['index']%2]
   for method in order:
    if method not in task['methods']:continue
    torch.random.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);assert rng_sha(torch)==fork_rng
    cache=clone_cache(planner);mapping=None;stages=[prec];proof_ids=[prec['reservation_id']];transfers=[];position_files=[];components=[];arm_start=time.monotonic()
    for hop,receiver in enumerate(['critic','refiner','judger']):
     save(out/'current.json',dict(index=item['index'],method=method,phase='select_for_'+receiver));attention=None;probe_seconds=0
     if method=='receiver':
      before_rng=rng_sha(torch);(attention,probe_receipt),probe_seconds=ns['timed'](lambda:ns['probe'](item,receiver,cache,stages[-1]['original_end'],False));assert rng_sha(torch)==before_rng;proof_ids.append(probe_receipt['reservation_id'])
     (cache,mapping,ix,transfer),select_seconds=ns['timed'](lambda:relay(cache,mapping,stages[-1],method,attention,compress,pairs,torch))
     filename=case_dir/(method+f'_handoff{hop+1}.npz');np.savez_compressed(filename,positions=mapping.cpu().numpy(),storage_indices=ix.cpu().numpy());position_files.append(dict(file=str(filename.relative_to(out)),file_sha256=sha(filename),position_sha=transfer['position_sha'],handoff=hop+1));transfer['handoff']=hop+1;transfers.append(transfer);components.append(dict(required_probe_seconds=probe_seconds,selector_and_gather_seconds=select_seconds));del attention,ix
     if hop<2:
      (cache,rec,_),seconds=ns['timed'](lambda:ns['stage'](item,receiver,cache,stages[-1]['original_end']));stages.append(rec);proof_ids.append(rec['reservation_id']);components[-1]['receiver_latent_seconds']=seconds
    save(out/'current.json',dict(index=item['index'],method=method,phase='judger'));engine.phase=HERE.name+':'+phase+':'+method
    (answer,jrec),judge_seconds=ns['timed'](lambda:ns['judge'](item,cache,stages[-1]['original_end'],method));proof_ids.append(jrec['reservation_id'])
    assert len(proof_ids)==(4 if method=='Full' else 7) and len(set(proof_ids))==len(proof_ids) and kv_sha(torch,pairs(planner))==prefix_sha
    pred=parse_answer(item,answer,extract_gsm8k_answer)
    row=dict(dataset=item['dataset'],question_id=item['question_id'],index=item['index'],is_diamond=item['is_diamond'],category=item['category'],method=method,latent_steps=10,seed=42,status='ok',phase=phase,answer_text=answer,prediction=pred,parse_valid=pred is not None,generated_tokens=len(jrec['generated_ids']),token_sha=hashlib.sha256(json.dumps(jrec['generated_ids']).encode()).hexdigest(),judger_receipt=jrec,shared_request_ids=[prec['reservation_id']],request_ids=proof_ids,shared_planner_kv_sha=prefix_sha,fork_rng_sha=fork_rng,prefix_stages=stages,transfers=transfers,position_files=position_files,actual_kv_bytes_by_handoff=[x['actual_kv_bytes'] for x in transfers],actual_cumulative_relay_bytes=sum(x['actual_kv_bytes'] for x in transfers),full_reference_bytes_by_original_positions=[x['full_bytes_from_original_positions'] for x in transfers],source_lock_sha=source['run_lock_sha256'],native_functions_ast_sha=ast_sha,adapter_sha=sha(HERE/'task_adapter.py'),timing=dict(shared_prefix_seconds=shared_seconds,components=components,judger_seconds=judge_seconds,arm_instrumented_seconds=time.monotonic()-arm_start,reconstructed_standalone_seconds=shared_seconds+time.monotonic()-arm_start,reconstructed_not_independently_timed=True,cuda_sync_both_boundaries=True,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    commit(task,row,out);save(out/'last_commit.json',dict(index=item['index'],method=method,unix=time.time()));del cache,mapping,transfers,stages,position_files;gc.collect();torch.cuda.empty_cache()
   append(out/'case_done.jsonl',dict(index=item['index'],seconds=time.monotonic()-case_start,unix=time.time()));del planner;gc.collect();torch.cuda.empty_cache();task=None
  save(out/'exit.json',dict(normal=True,phase=phase,identity=identity(os.getpid()),unix=time.time()))
 except BaseException as e:
  save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),identity=identity(os.getpid()),current_task=task,unix=time.time()));hold(task,e);raise

if __name__=='__main__':
 p=__import__('argparse').ArgumentParser();p.add_argument('--phase',choices=['engineering','formal'],required=True);p.add_argument('--slot',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.phase,a.slot,Path(a.out))
