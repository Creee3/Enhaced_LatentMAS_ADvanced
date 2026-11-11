"""Resident diagnostic-only worker. No Judger or final answer generation."""
import gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
from common import HERE,ROOT,BASE,SCIENCE,OLD,load,save,sha,identity,claim,commit,hold,verify
sys.path[:0]=[str(SCIENCE),str(BASE)]
from prompts import ROLES,intervention_messages
from measure import sample_indices,describe,compare,role_effect,windows,overlap
def main(slot,out):
 task=None;out.mkdir(parents=True,exist_ok=False);save(out/'loading.json',dict(identity=identity(os.getpid()),slot=slot,unix=time.time()))
 try:
  source=verify();assert os.environ.get('LATENTMAS_SHARED_LEDGER')==str(ROOT/'cumulative_budget.json')
  import torch,numpy as np,runpy
  import experiment_worker as ew
  from runtime_common import Ledger,seed_all
  from stage_ops import pairs,clone_cache,observe
  sys.path.insert(0,str(OLD));from task_adapter import args_for,messages
  ew.args_for=args_for;torch.set_num_threads(1);allowed=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed[slot%len(allowed)]})
  original_check=Ledger.check
  def check(self):
   original_check(self)
   if (HERE/'STOP').exists():raise RuntimeError('user STOP requested')
  Ledger.check=check;started=time.monotonic();runner=ew.Runner(str(out),torch.device('cuda:0'));engine=runner.engine;model=engine.model
  engine.config.update(experiment_id=HERE.name,request_timeout_seconds=1800);engine.phase=HERE.name
  matrix,target=engine.core._latent_realign_matrices[id(model)]
  ready=dict(identity=identity(os.getpid()),slot=slot,source=source,model_load_seconds=time.monotonic()-started,weight_files=len(load(BASE/'model_lock.json')['files']),bf16_cuda=all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters()),parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),identity_norm=torch.equal(matrix,torch.eye(matrix.shape[0],dtype=matrix.dtype,device=matrix.device)),target_norm=float(target),unix=time.time())
  assert ready['weight_files']==15 and ready['bf16_cuda'] and ready['parameter_bytes']==16381470720 and ready['identity_norm'] and abs(ready['target_norm']-1.3758320808410645)<1e-6
  save(out/'model_ready.json',ready);items=load(HERE/'questions.json');proofs=load(HERE/'preflight.json')['proofs'];instructions=load(HERE/'matched_instructions.json')['instructions'];build=runpy.run_path(str(OLD/'dependencies/LatentMAS/prompts.py'))['build_agent_message_sequential_latent_mas']
  def rng():return (torch.random.get_rng_state(),torch.cuda.get_rng_state_all())
  def restore(r):torch.random.set_rng_state(r[0]);torch.cuda.set_rng_state_all(r[1])
  def rng_sha():
   r=rng();return hashlib.sha256(r[0].numpy().tobytes()+b''.join(x.cpu().numpy().tobytes() for x in r[1])).hexdigest()
  def cache_sha(cache):
   h=hashlib.sha256()
   for pair in pairs(cache):
    for t in pair:h.update(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
   return h.hexdigest()
  def sampled(cache,ix):
   xs=pairs(cache);assert len(xs)==36
   return torch.stack([torch.stack([t[0,:,ix,:].detach().cpu() for t in pair]) for pair in xs])
  def as_float(t):return t.float().numpy()
  def write_npz(path,arr):
   np.savez_compressed(path,**arr);return dict(file=str(path.relative_to(out)),sha256=sha(path),bytes=path.stat().st_size)
  def stage(item,msg,cache,absolute,expected,kind):
   prompt,ids,_=engine.prepare(msg);assert len(ids[0])==expected['prompt_len'] and hashlib.sha256(json.dumps(ids[0].cpu().tolist(),separators=(',',':')).encode()).hexdigest()==expected['token_sha']
   assert (0 if cache is None else cache.get_seq_length())==absolute
   with observe(model,absolute,latent=False,prefix=pairs(cache)) as proof:
    new,rec,_=engine.latent(msg,10,dict(dataset=item['dataset'],question_id=item['question_id'],kind=kind),past=cache)
   assert proof['forwards']==11 and proof['positions']==rec['actual_positions']==expected['prompt_len']+10 and proof['sdpa_checks']==396
   rec.update(proof={k:proof[k] for k in ['forwards','positions','sdpa_checks']},original_before=absolute,original_end=absolute+expected['prompt_len']+10,prompt_len=expected['prompt_len']);return new,rec
  while not (HERE/'STOP').exists() and not (HERE/'MANAGEMENT_HOLD').exists():
   phase='formal' if (HERE/'engineering_release.json').exists() else 'engineering';task=claim(phase,out)
   if task is None:
    q=load(HERE/'queue.json')
    if phase=='formal' and not any(t['state'] in ['pending','running'] for t in q['tasks']):break
    save(out/'current.json',dict(phase='waiting_'+phase,unix=time.time()));time.sleep(2);continue
   verify();item=items[task['index']];pre=proofs[item['index']];seed_all(42);folder=out/f"case{item['index']:04d}";folder.mkdir();receipts=[];files=[];context_results=[];native_results=[];cache=None;absolute=0;case_start=time.monotonic()
   for context_id,native_role in enumerate(ROLES):
    prefix_hash=cache_sha(cache);r=rng();fork=rng_sha();save(out/'current.json',dict(index=item['index'],phase='role_interventions',context=context_id,unix=time.time()))
    prompt_samples=[];latent_samples=[];prompt_desc=[];latent_desc=[];raw_prompt=[];raw_latent=[];branch_info=[]
    for b,bproof in enumerate(pre['branches']):
     role,variant=bproof['role'],bproof['variant'];restore(r);assert rng_sha()==fork
     copy=clone_cache(cache);new,rec=stage(item,intervention_messages(item,instructions[role][variant]),copy,absolute,bproof,'role_task_intervention');receipts.append(rec)
     pix=sample_indices(bproof['question_end']-bproof['question_start'],32)+absolute+bproof['question_start'];lix=np.arange(absolute+bproof['prompt_len'],absolute+bproof['prompt_len']+10)
     p=sampled(new,pix.tolist());l=sampled(new,lix.tolist());pf=as_float(p);lf=as_float(l);pd=describe(pf);ld=describe(lf)
     prompt_samples.append(pf);latent_samples.append(lf);prompt_desc.append(pd);latent_desc.append(ld);raw_prompt.append(p.view(torch.uint16).numpy());raw_latent.append(l.view(torch.uint16).numpy())
     branch_info.append(dict(role=role,variant=variant,reservation_id=rec['reservation_id'],fork_rng_sha=fork,prompt_positions=pix.tolist(),latent_positions=lix.tolist(),question_token_sha=bproof['question_token_sha']))
     del new,copy,p,l,pf,lf;gc.collect();torch.cuda.empty_cache()
    assert cache_sha(cache)==prefix_hash and absolute==pre['histories'][context_id]
    arrays={};effects={}
    for kind,samples,desc in [('prompt',prompt_samples,prompt_desc),('latent',latent_samples,latent_desc)]:
     pp=compare(samples,desc);eff=role_effect(pp);effects[kind]={k:np.nanmean(v,axis=(0,2)).tolist() for k,v in eff.items()}
     arrays[kind+'_metrics']=np.stack([x[0] for x in desc]);arrays[kind+'_basis']=np.stack([x[1] for x in desc]);arrays[kind+'_basis_rank']=np.stack([x[2] for x in desc]);arrays[kind+'_pair_subspace_distance']=np.stack([x['subspace_distance'] for x in pp]);arrays[kind+'_pair_relative_l2']=np.stack([x['relative_l2'] for x in pp]);arrays[kind+'_effect']=eff['between_minus_within']
    arrays['prompt_kv_bf16_bits']=np.stack(raw_prompt);arrays['latent_kv_bf16_bits']=np.stack(raw_latent);arrays['pair_same_role']=np.array([x['same_role'] for x in pp]);arrays['pair_indices']=np.array([[x['i'],x['j']] for x in pp])
    arrays['metadata_json']=np.array(json.dumps(dict(index=item['index'],context=context_id,prefix_sha=prefix_hash,branches=branch_info,axes='branch,layer,K_or_V,head,sampled_position,dimension',kv_dtype='native_bfloat16_uint16_bits',feature_dimension=128),ensure_ascii=False));files.append(write_npz(folder/f'context{context_id}.npz',arrays));context_results.append(dict(context=context_id,prefix_positions=absolute,prefix_sha=prefix_hash,effects=effects))
    del arrays,prompt_samples,latent_samples,prompt_desc,latent_desc,raw_prompt,raw_latent;restore(r)
    save(out/'current.json',dict(index=item['index'],phase='native_full_'+native_role,unix=time.time()))
    cache,rec=stage(item,messages(native_role,item,build,runner.model_name),cache,absolute,pre['native'][context_id],'native_full_reconstruction');receipts.append(rec)
    # Equal, non-overlapping windows of this role's original prompt. Remove sink
    # on Planner, retain wording as an explicitly observational position control.
    start=absolute+(4 if context_id==0 else 0);available=rec['prompt_len']-(4 if context_id==0 else 0);ww=windows(available);wm=[];wb=[];wr=[];bits=[]
    for w in ww:
     t=sampled(cache,(w+start).tolist());m,b,rank=describe(as_float(t));wm.append(m);wb.append(b);wr.append(rank);bits.append(t.view(torch.uint16).numpy())
    ov=np.stack([overlap(wb[i],wr[i],wb[j],wr[j]) for i,j in [(0,1),(0,2),(1,2)]])
    arr=dict(window_metrics=np.stack(wm),window_basis=np.stack(wb),window_basis_rank=np.stack(wr),window_overlap=ov,window_kv_bf16_bits=np.stack(bits),positions=np.stack(ww)+start,metadata_json=np.array(json.dumps(dict(role=native_role,index=item['index'],prompt_len=rec['prompt_len'],sampling='early_middle_late_equal_rows',template_included=True))))
    files.append(write_npz(folder/f'full_{native_role}_windows.npz',arr));native_results.append(dict(role=native_role,positions=rec['original_end'],window_rows=len(ww[0]),metrics_mean=np.mean(arr['window_metrics'],axis=(1,3)).tolist()));absolute=rec['original_end'];del arr,wm,wb,wr,bits;gc.collect();torch.cuda.empty_cache()
   assert len(receipts)==21 and len({r['reservation_id'] for r in receipts})==21 and sum(r['actual_positions'] for r in receipts)==pre['upper_positions']
   row=dict(index=item['index'],question_id=item['question_id'],dataset_group=item['dataset_group'],status='ok',phase=phase,request_ids=[r['reservation_id'] for r in receipts],receipts=receipts,diagnostic_files=files,contexts=context_results,native=native_results,run_lock_sha256=source['run_lock_sha256'],seconds=time.monotonic()-case_start,actual_positions=sum(r['actual_positions'] for r in receipts),no_judger_or_final_answers=True)
   commit(task,row,out);save(out/'last_commit.json',dict(index=item['index'],unix=time.time(),requests=21));task=None;del cache;gc.collect();torch.cuda.empty_cache()
  save(out/'exit.json',dict(normal=True,identity=identity(os.getpid()),unix=time.time()))
 except BaseException as e:
  save(out/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),identity=identity(os.getpid()),task=task,unix=time.time()));hold(task,e);raise
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--slot',type=int,required=True);p.add_argument('--out',required=True);a=p.parse_args();main(a.slot,Path(a.out))
