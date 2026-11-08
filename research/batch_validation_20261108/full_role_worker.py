"""Prepared four-case T/R/G smoke entry; frozen scientific functions by AST."""
import argparse,ast,gc,hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025');RUN=ROOT/'evidence/TRG225_a100_s42_l10_20261106T155639Z';BASE=ROOT/'evidence/gpqa_repair_8096_20261028T161254Z/bundle/runtime'
OUT=Path(__file__).resolve().parent;sys.path[:0]=[str(RUN),str(BASE)]
from runtime_common import Ledger,append,sha,seed_all
def main(index):
 import torch,numpy as np,torch.nn.functional as F
 import experiment_worker as ew
 from stage_ops import observe,pairs,compress,make_cache,clone_cache
 from selector_dispatch import select
 from cloud_common import verify
 source=verify();plan=json.loads((OUT/'launch_plan.json').read_text());assert plan['status']=='VALIDATED_FOUR_CASE_SMOKE' and plan['methods']==['T','R','G'] and plan['case_indices']==[0,1,2,3]
 out=OUT/f'case{index}';out.mkdir();items=json.loads((RUN/'questions.json').read_text());item=items[index]
 names=['args_for','timed','context','stage','probe','judge','final_select'];tree=ast.parse((RUN/'cloud_worker.py').read_text());oldmain=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');defs=[n for n in ast.walk(oldmain) if isinstance(n,ast.FunctionDef) and n.name in names];assert len(defs)==7
 ns=dict(globals(),**locals());exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),'frozen_native_science','exec'),ns);ew.args_for=ns['args_for']
 runner=ew.Runner(str(out),torch.device('cuda:0'));engine=runner.engine;model=engine.model;engine.config.update(experiment_id=OUT.name,request_timeout_seconds=1800);engine.phase='four_case_full_role_smoke';ns.update(runner=runner,engine=engine,model=model,current=dict(repeat=1))
 matrix,target=engine.core._latent_realign_matrices[id(model)];assert torch.equal(matrix,torch.eye(matrix.shape[0],device=matrix.device)) and abs(float(target)-1.3758320808410645)<1e-6
 (out/'model_ready.json').write_text(json.dumps(dict(pid=os.getpid(),source=source,bf16_cuda=all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters()),parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),identity_norm=True,target_norm=float(target))))
 seed_all(42);cache=None;mapping=None;absolute=0;stages=[];transfers=[];shared=[]
 for k,role in enumerate(['planner','critic','refiner']):
  cache,rec,_=ns['stage'](item,role,cache,absolute);stages.append(rec);shared.append(rec['reservation_id']);absolute=rec['original_end']
  if k<2:
   attention,probe=ns['probe'](item,['critic','refiner'][k],cache,absolute);shared.append(probe['reservation_id']);cache,mapping,transfer=compress(cache,mapping,role=role,steps=10,prompt_len=rec['prompt_len'],original_before=rec['original_before'],method='receiver_attention_sink_recent',attention=attention);transfers.append(transfer)
 captured,probe=ns['probe'](item,'judger',cache,absolute,True);shared.append(probe['reservation_id']);rng=(torch.random.get_rng_state(),torch.cuda.get_rng_state_all())
 order=['T','R','G'];offset=index%3;order=order[offset:]+order[:offset]
 for mode in order:
  compact,position,ix,diagnostics,_=ns['final_select'](cache,mapping,stages,captured,mode)
  for l,(old,new) in enumerate(zip(pairs(cache),pairs(compact))):
   for a,b in zip(old,new):assert torch.equal(torch.gather(a,2,ix[l][None,:,:,None].expand_as(b)),b)
  assert sum(t.numel()*t.element_size() for p in pairs(compact) for t in p)==19169280 and bool(((position<4).sum(-1)==4).all())
  for s in stages:assert bool((((position>=s['original_end']-10)&(position<s['original_end'])).sum(-1)==10).all())
  s=stages[-1];last=s['original_before']+s['prompt_len']-8;assert bool((((position>=last)&(position<last+8)).sum(-1)==8).all())
  torch.random.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1]);start=time.monotonic();answer,receipt=ns['judge'](item,compact,absolute,mode)
  np.savez_compressed(out/(mode+'_positions.npz'),positions=position.cpu().numpy())
  append(out/'results.jsonl',dict(engineering_only=True,packet=index,method=mode,seed=42,latent_steps=10,cap=8192,answer_text=answer,receipt=receipt,token_sha=hashlib.sha256(json.dumps(receipt['generated_ids']).encode()).hexdigest(),position_sha=hashlib.sha256(position.cpu().numpy().tobytes()).hexdigest(),shared_request_ids=shared,prefix_transfers=transfers,actual_kv_bytes=19169280,seconds=time.monotonic()-start,diagnostics=diagnostics))
  del compact;gc.collect();torch.cuda.empty_cache()
 (out/'done.json').write_text(json.dumps(dict(passed=True,methods=order,accuracy_computed=False)))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,choices=range(4),required=True);a=ap.parse_args()
 try:main(a.index)
 except BaseException as e:(OUT/f'failure_case{a.index}.json').write_text(json.dumps(dict(error=repr(e),pid=os.getpid())));raise
