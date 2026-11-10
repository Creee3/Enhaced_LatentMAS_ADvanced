"""Resident twelve-method scientific worker; stages retain frozen native AST."""
import ast,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
from common import HERE,ROOT,SCIENCE,BASE,METHODS,load,save,append,sha,verify,identity,claim,commit,hold
sys.path[:0]=[str(HERE),str(SCIENCE),str(BASE)]
from runtime_common import Ledger,seed_all
from task_adapter import args_for,messages,parse_answer,load_frozen_builder
from controls import stable_seed

def kv_sha(torch,xs):
 h=hashlib.sha256()
 for pair in xs:
  for t in pair:h.update(str((tuple(t.shape),str(t.dtype))).encode());h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
 return h.hexdigest()
def rng_sha(torch):return hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()+b''.join(x.cpu().numpy().tobytes() for x in torch.cuda.get_rng_state_all())).hexdigest()
def snapshot_rng(torch):return (torch.random.get_rng_state(),torch.cuda.get_rng_state_all())
def restore_rng(torch,rng):torch.random.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1])

def main(slot,out):
 task=None;out.mkdir(parents=True,exist_ok=False);save(out/'loading.json',dict(identity=identity(os.getpid()),slot=slot,unix=time.time()))
 try:
  source=verify();assert os.environ.get('LATENTMAS_SHARED_LEDGER')==str(ROOT/'cumulative_budget.json')
  import torch,numpy as np,torch.nn.functional as F
  import experiment_worker as ew
  from stage_ops import pairs,make_cache,clone_cache,observe
  from method_ops import select_method
  from relay_select import provenance_map,masks
  from diagnostics import geometry,readout,save_compact_npz
  from model_ops import probe,capture_prefill,fidelity
  torch.set_num_threads(1);allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[slot%len(allowed)]})
  original_check=Ledger.check
  def check(self):
   original_check(self)
   if (HERE/'STOP').exists():raise RuntimeError('user STOP requested')
  Ledger.check=check
  ew.args_for=args_for
  begin=time.monotonic();runner=ew.Runner(str(out),torch.device('cuda:0'));engine=runner.engine;model=engine.model
  engine.config.update(experiment_id=HERE.name,request_timeout_seconds=1800);engine.phase=HERE.name
  matrix,target=engine.core._latent_realign_matrices[id(model)]
  ready=dict(identity=identity(os.getpid()),source=source,slot=slot,model_initialization_seconds=time.monotonic()-begin,weight_files=len(load(BASE/'model_lock.json')['files']),bf16_cuda=all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters()),parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),identity_norm=torch.equal(matrix,torch.eye(matrix.shape[0],dtype=matrix.dtype,device=matrix.device)),target_norm=float(target),unix=time.time())
  assert ready['weight_files']==15 and ready['bf16_cuda'] and ready['parameter_bytes']==16381470720 and ready['identity_norm'] and abs(ready['target_norm']-1.3758320808410645)<1e-6
  save(out/'model_ready.json',ready)
  items=load(HERE/'questions.json');pre={x['index']:x for x in load(HERE/'preflight.json')['proofs']};current_item=None;expected_role=[None];native_builder=load_frozen_builder(HERE)
  def prompt(role,question,args):
   assert question==current_item['question'];expected_role[0]=role;return messages(role,current_item,native_builder,runner.model_name)
  ew.stage_prompt=prompt;native_prepare=engine.prepare
  def prepare(ms):
   p,ids,mask=native_prepare(ms);proof=pre[current_item['index']];role=expected_role[0]
   assert ids.shape[-1]==proof['prompt_lengths'][role] and hashlib.sha256(json.dumps(ids[0].cpu().tolist(),separators=(',',':')).encode()).hexdigest()==proof['prompt_token_sha'][role]
   return p,ids,mask
  engine.prepare=prepare
  tree=ast.parse((SCIENCE/'cloud_worker.py').read_text());oldmain=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');names=['timed','context','stage','judge'];defs=[n for n in ast.walk(oldmain) if isinstance(n,ast.FunctionDef) and n.name in names];assert len(defs)==4
  ns=dict(globals(),**locals(),current=dict(repeat=1));exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),'frozen_native_stage_judge','exec'),ns)
  ast_sha=hashlib.sha256(ast.dump(ast.Module(body=defs,type_ignores=[]),include_attributes=False).encode()).hexdigest();save(out/'native_functions.json',dict(names=names,ast_sha256=ast_sha,source_file_sha256=sha(SCIENCE/'cloud_worker.py')))
  def diagnostic(path,arrays,metadata):
   receipt=save_compact_npz(path,arrays,metadata=metadata);receipt['file']=str(path.relative_to(out));receipt.pop('path');return receipt
  def observed(item,receiver,cache,absolute,kind,detail=True):
   before_rng=snapshot_rng(torch);before_sha=rng_sha(torch)
   try:
    result=probe(engine,model,item,receiver,cache,absolute,prepare_messages=lambda role,item:prompt(role,item['question'],None),observe=observe,pairs=pairs,clone_cache=clone_cache,detail=detail,kind=kind)
    assert rng_sha(torch)==before_sha,'Diagnostic forward unexpectedly consumed sampling RNG'
    return result
   finally:restore_rng(torch,before_rng)
  def selection(cache,mapping,stages,method,sender,captured,seed):
   rng0=snapshot_rng(torch)
   try:return select_method(cache,mapping,stages,method,sender,captured,seed,pairs,make_cache)
   finally:restore_rng(torch,rng0)
  save(out/'worker_ready_to_claim.json',dict(identity=identity(os.getpid()),unix=time.time()))
  while not (HERE/'STOP').exists() and not (HERE/'MANAGEMENT_HOLD').exists():
   phase='formal' if (HERE/'engineering_release.json').exists() else 'engineering'
   task=claim(phase,out)
   if task is None:
    q=load(HERE/'queue.json');eligible=q['tasks'] if phase=='formal' else [t for t in q['tasks'] if t['index'] in q['engineering_indices']]
    if phase=='formal' and not any(t['state'] in ['pending','running'] for t in eligible):break
    save(out/'current.json',dict(phase='waiting_'+phase,unix=time.time()));time.sleep(2);continue
   verify();current_item=items[task['index']];item=current_item;ns['current']=dict(repeat=1);seed_all(42);torch.cuda.reset_peak_memory_stats();case_start=time.monotonic();common_diag_ids=[];common_files=[]
   case_dir=out/'positions'/f"case{item['index']:04d}";case_dir.mkdir(parents=True,exist_ok=True);engine.phase=HERE.name+':'+phase+':shared'
   save(out/'current.json',dict(index=item['index'],phase='shared_planner',unix=time.time()))
   with capture_prefill(model) as planner_features:
    (planner,prec,planner_attention),planner_seconds=ns['timed'](lambda:ns['stage'](item,'planner',None,0))
   prec['native_prefill_seconds']=planner_features['prefill_seconds'];prec['native_latent_decode_seconds']=max(0,planner_seconds-planner_features['prefill_seconds']);del planner_features
   (first_probe,first_receipt,first_features),first_probe_seconds=ns['timed'](lambda:observed(item,'critic',planner,prec['original_end'],'shared_first_receiver_probe',True))
   prefix_sha=kv_sha(torch,pairs(planner));rng=snapshot_rng(torch);fork_rng=rng_sha(torch);shared_seconds=time.monotonic()-case_start
   for method in METHODS:
    if method not in task['methods']:continue
    restore_rng(torch,rng);assert rng_sha(torch)==fork_rng
    cache=clone_cache(planner);mapping=None;stages=[prec];proof_ids=[prec['reservation_id'],first_receipt['reservation_id']];transfers=[];position_files=[];diagnostic_files=[];components=[];fidelities=[];arm_start=time.monotonic();sender_attention=planner_attention
    for hop,receiver in enumerate(['critic','refiner','judger']):
     save(out/'current.json',dict(index=item['index'],method=method,phase='diagnose_for_'+receiver,unix=time.time()));seed=stable_seed(item['question_id'],hop+1)
     if hop==0:captured,probe_receipt,ref_features=first_probe,first_receipt,first_features;probe_seconds=0
     else:
      (captured,probe_receipt,ref_features),probe_seconds=ns['timed'](lambda:observed(item,receiver,cache,stages[-1]['original_end'],'method_predeletion_probe',True));proof_ids.append(probe_receipt['reservation_id'])
     old_xs=pairs(cache);allmap=provenance_map(mapping,stages,36,8,old_xs[0][0].device);source_before=kv_sha(torch,old_xs)
     (geo,geo_meta),geo_seconds=ns['timed'](lambda:geometry(old_xs,allmap,stages,seed=seed));diagnostic_files.append(diagnostic(case_dir/(method+f'_hop{hop+1}_geometry.npz'),geo,dict(method=method,hop=hop+1,source_cache_sha=source_before,**geo_meta)))
     if method=='Full':
      # On Full's exact state, evaluate all selector objectives and all later-hop
      # receiving prefills. These probes are diagnostic costs, never answer keys.
      for local_method in METHODS:
       compact,pos_l,ix_l,rec_l=selection(cache,mapping,stages,local_method,sender_attention,captured,seed)
       labels=torch.stack([masks(allmap[l],stages,captured[l][0].shape[-1])[0] for l in range(36)])
       arr=readout(captured,ix_l,pairs(compact),labels,allmap.shape[-1]);meta=dict(analysis='same_full_state',method=local_method,hop=hop+1,source_cache_sha=source_before,position_sha=rec_l['position_sha'],actual_kv_bytes=rec_l['actual_kv_bytes'])
       if hop>0 and local_method!='Full':
        (_,lr,lf),local_probe_seconds=ns['timed'](lambda:observed(item,receiver,compact,stages[-1]['original_end'],'same_full_state_'+local_method,False));common_diag_ids.append(lr['reservation_id']);arr.update({'receiver_'+k:v for k,v in fidelity(ref_features,lf).items()});meta.update(receiver_probe_rid=lr['reservation_id'],receiver_probe_seconds=local_probe_seconds);del lf
       common_files.append(diagnostic(case_dir/(f'common_hop{hop+1}_'+local_method+'.npz'),arr,meta));del compact,pos_l,ix_l,arr
      assert kv_sha(torch,old_xs)==source_before
     (new_cache,new_mapping,ix,transfer),select_seconds=ns['timed'](lambda:selection(cache,mapping,stages,method,sender_attention,captured,seed))
     assert kv_sha(torch,old_xs)==source_before
     labels=torch.stack([masks(allmap[l],stages,captured[l][0].shape[-1])[0] for l in range(36)])
     arr=readout(captured,ix,pairs(new_cache),labels,allmap.shape[-1]);diagnostic_files.append(diagnostic(case_dir/(method+f'_hop{hop+1}_readout.npz'),arr,dict(analysis='own_trajectory_same_state',method=method,hop=hop+1,source_cache_sha=source_before,position_sha=transfer['position_sha'])))
     path=case_dir/(method+f'_handoff{hop+1}.npz');np.savez_compressed(path,positions=new_mapping.cpu().numpy(),storage_indices=ix.cpu().numpy());position_files.append(dict(file=str(path.relative_to(out)),file_sha256=sha(path),position_sha=transfer['position_sha'],handoff=hop+1));transfers.append(transfer)
     cache,mapping=new_cache,new_mapping;del old_xs,allmap,captured,ix,arr;gc.collect()
     components.append(dict(required_probe_seconds=probe_seconds,geometry_seconds=geo_seconds,selector_and_gather_seconds=select_seconds))
     with capture_prefill(model) as actual_features:
      if hop<2:
       (cache,rec,sender_attention),seconds=ns['timed'](lambda:ns['stage'](item,receiver,cache,stages[-1]['original_end']));stages.append(rec);proof_ids.append(rec['reservation_id']);components[-1]['receiver_latent_seconds']=seconds;components[-1]['receiver_prefill_seconds']=actual_features['prefill_seconds'];components[-1]['receiver_latent_decode_seconds']=max(0,seconds-actual_features['prefill_seconds'])
      else:
       save(out/'current.json',dict(index=item['index'],method=method,phase='judger',unix=time.time()));engine.phase=HERE.name+':'+phase+':'+method
       (answer,jrec),judge_seconds=ns['timed'](lambda:ns['judge'](item,cache,stages[-1]['original_end'],method));proof_ids.append(jrec['reservation_id'])
     metrics=fidelity(ref_features,actual_features);fidelities.append(dict(**{k:v.tolist() for k,v in metrics.items()},native_prefill_seconds=actual_features['prefill_seconds']));diagnostic_files.append(diagnostic(case_dir/(method+f'_hop{hop+1}_receiver.npz'),metrics,dict(analysis='own_trajectory_same_state',first_hop_common_prefix=hop==0,method=method,hop=hop+1,receiver=receiver,decode_position=0,reference_probe_rid=probe_receipt['reservation_id'],native_prefill_rid=rec['reservation_id'] if hop<2 else jrec['reservation_id'])))
     del ref_features,actual_features;gc.collect();torch.cuda.empty_cache()
    judge_prefill_seconds=fidelities[-1]['native_prefill_seconds']
    assert len(proof_ids)==7 and len(set(proof_ids))==7 and kv_sha(torch,pairs(planner))==prefix_sha
    pred=parse_answer(item,answer)
    row=dict(global_index=item['global_index'],dataset=item['dataset'],dataset_group=item['dataset_group'],question_id=item['question_id'],index=item['index'],is_diamond=item['is_diamond'],category=item['category'],method=method,latent_steps=10,seed=42,status='ok',phase=phase,answer_text=answer,prediction=pred,parse_valid=pred is not None,generated_tokens=len(jrec['generated_ids']),token_sha=hashlib.sha256(json.dumps(jrec['generated_ids']).encode()).hexdigest(),judger_receipt=jrec,shared_request_ids=[prec['reservation_id'],first_receipt['reservation_id']],request_ids=proof_ids,shared_planner_kv_sha=prefix_sha,fork_rng_sha=fork_rng,prefix_stages=stages,transfers=transfers,position_files=position_files,diagnostic_files=diagnostic_files,receiver_fidelity=fidelities,common_state_diagnostic_request_ids=list(common_diag_ids),common_state_diagnostic_files=list(common_files),actual_kv_bytes_by_handoff=[x['actual_kv_bytes'] for x in transfers],actual_cumulative_relay_bytes=sum(x['actual_kv_bytes'] for x in transfers),source_lock_sha=source['run_lock_sha256'],native_functions_ast_sha=ast_sha,timing=dict(shared_prefix_seconds=shared_seconds,shared_first_probe_seconds=first_probe_seconds,components=components,judger_seconds=judge_seconds,judger_prefill_seconds=judge_prefill_seconds,judger_decode_seconds=max(0,judge_seconds-judge_prefill_seconds),judger_generated_tokens=len(jrec['generated_ids']),arm_instrumented_seconds=time.monotonic()-arm_start,reconstructed_standalone_seconds=shared_seconds+time.monotonic()-arm_start,reconstructed_not_independently_timed=True,cuda_sync_both_boundaries=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))
    commit(task,row,out);save(out/'last_commit.json',dict(index=item['index'],method=method,unix=time.time()));del cache,mapping,transfers,stages,position_files;gc.collect();torch.cuda.empty_cache()
   append(out/'case_done.jsonl',dict(index=item['index'],seconds=time.monotonic()-case_start,common_diagnostic_request_ids=common_diag_ids,common_diagnostic_files=common_files,unix=time.time()));del planner,planner_attention,first_probe,first_features;gc.collect();torch.cuda.empty_cache();task=None
  save(out/'exit.json',dict(normal=True,identity=identity(os.getpid()),unix=time.time()))
 except BaseException as e:
  save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),identity=identity(os.getpid()),current_task=task,unix=time.time()));hold(task,e);raise

if __name__=='__main__':
 p=__import__('argparse').ArgumentParser();p.add_argument('--slot',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.slot,Path(a.out))
