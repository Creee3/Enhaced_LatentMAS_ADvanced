"""Authorized G-only matched225 attention control; frozen prefix and gather."""
import argparse, contextlib, gc, hashlib, json, os, sys, time, traceback
from pathlib import Path
from types import SimpleNamespace

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
BASE = ROOT/'evidence/gpqa_repair_8096_20261028T161254Z/bundle'
sys.path.insert(0,str(BASE/'runtime'))
from cloud_common import save, append, load, sha, identity, qlock, all_rows, key
from queue_logic import choose

def verify(snapshots=False):
    lock=load(RUN/'run_lock.json')
    for name,entry in lock['files'].items():
        assert sha(RUN/name)==entry['sha256'], name
    if snapshots:
        for name,entry in lock['snapshots'].items():
            assert sha(RUN/name)==entry['sha256'], name
    assert sha(BASE/'source_lock.json')==lock['base_source_lock_sha256']
    for name,entry in load(BASE/'source_lock.json')['files'].items():
        assert sha(BASE/name)==entry['sha256'], name
    return dict(passed=True,run_sha=sha(RUN/'run_lock.json'),bundle_files=37)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    def record(name,x):save(out/name,dict(unix=time.time(),identity=identity(os.getpid()),**x))
    count=0;current=None;current_task=None
    record('loading.json',dict(phase='hash_and_model_loading'))
    try:
        source=verify(snapshots=True);plan=load(RUN/'plan.json');items=load(RUN/'questions.json')
        assert plan['methods']==['G']
        references=[json.loads(x) for x in (RUN/'baseline_results.jsonl').read_bytes().split(b'\n') if x]
        reference_T={r['question_id']:r for r in references if r['method']=='T'}
        assert len(reference_T)==len(items)==225
        assert os.environ['LATENTMAS_SHARED_LEDGER']==str(ROOT/'cumulative_budget.json')
        assert (ROOT/'cumulative_budget.json').exists()
        import numpy as np
        import torch
        import torch.nn.functional as F
        import experiment_worker as ew
        from runtime_common import Ledger,seed_all
        from stage_ops import pairs,clone_cache,make_cache,observe,compress
        from attention_selector import select
        from author_parser import extract_gsm8k_answer,normalize_answer
        torch.set_num_threads(2)
        old_check=Ledger.check
        def check(self):
            old_check(self)
            if (RUN/'STOP').exists():raise RuntimeError('user STOP requested')
        Ledger.check=check
        def args_for(ds,name,method='latent_mas'):
            return SimpleNamespace(task='gpqa',model_name=name,prompt='sequential',think=False,
                method=method,text_mas_context_length=-1,latent_space_realign=False)
        ew.args_for=args_for
        started=time.monotonic();runner=ew.Runner(str(out),torch.device('cuda:0'))
        engine=runner.engine;model=engine.model
        engine.config.update(experiment_id=RUN.name,request_timeout_seconds=1800);engine.phase=RUN.name
        matrix,target=engine.core._latent_realign_matrices[id(model)]
        ready=dict(source=source,model_load_hash_prepare_seconds=time.monotonic()-started,
            verified_weight_files=len(load(BASE/'runtime/model_lock.json')['files']),
            bf16_cuda=all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters()),
            parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            identity_norm=torch.equal(matrix,torch.eye(matrix.shape[0],dtype=matrix.dtype,device=matrix.device)),
            target_norm=float(target),device=torch.cuda.get_device_name(0))
        assert ready['verified_weight_files']==15 and ready['bf16_cuda'] and ready['parameter_bytes']==16381470720
        assert ready['identity_norm'] and not engine.core.args.latent_space_realign
        record('model_ready.json',ready)
        def timed(fn):
            torch.cuda.synchronize();start=time.monotonic();value=fn();torch.cuda.synchronize()
            return value,time.monotonic()-start
        def context(item,**kw):
            return dict(dataset=item['dataset'],question_id=item['question_id'],repeat=current['repeat'],**kw)
        def stage(item,role,cache,absolute):
            messages=ew.stage_prompt(role,item['question'],args_for(item['dataset'],runner.model_name))
            _,ids,_=engine.prepare(messages);p=int(ids.shape[-1])
            with observe(model,absolute,latent=True,prefix=pairs(cache)) as proof:
                cache,rec,_=engine.latent(messages,10,context(item,receiver=role,kind='shared_prefix'),past=cache)
            assert proof['positions']==rec['actual_positions'] and proof['forwards']==11
            rec.update(role=role,prompt_len=p,original_before=absolute,original_end=absolute+p+10,
                       proof=dict(forwards=proof['forwards'],sdpa_checks=proof['sdpa_checks']))
            return cache,rec,proof['attention_steps']
        def probe(item,receiver,cache,absolute,detail=False):
            messages=ew.stage_prompt(receiver,item['question'],args_for(item['dataset'],runner.model_name))
            prompt,ids,mask=engine.prepare(messages);before=cache.get_seq_length();copy=clone_cache(cache)
            full_mask=torch.cat([torch.ones(1,before,dtype=mask.dtype,device=mask.device),mask],-1)
            cp=torch.arange(before,before+ids.shape[-1],device=ids.device);captured=[]
            assert absolute+ids.shape[-1]<32768
            with observe(model,absolute,latent=False,prefix=pairs(cache),capture_queries=0 if detail else 16) as proof:
                previous=F.scaled_dot_product_attention
                def capture(q,k,v,*aa,**kw):
                    output=previous(q,k,v,*aa,**kw)
                    if q.shape[1]!=k.shape[1]:k=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
                    take=min(16,q.shape[-2]);score=q[...,-take:,:].float() @ k.float().transpose(-1,-2)
                    score*=kw.get('scale') or q.shape[-1]**-.5
                    mask=kw.get('attn_mask',aa[0] if aa else None)
                    assert not kw.get('is_causal',aa[2] if len(aa)>2 else False)
                    if mask is not None:
                        if mask.shape[-2]>1:mask=mask[...,-take:,:]
                        score=score.masked_fill(~mask,-torch.inf) if mask.dtype==torch.bool else score+mask.float()
                    heads=model.config.num_key_value_heads;group=q.shape[1]//heads
                    attn=score.softmax(-1).reshape(heads,group*take,-1).detach()
                    vv=v[0]
                    if vv.shape[0]!=heads:
                        grouped=vv.reshape(heads,group,vv.shape[-2],vv.shape[-1])
                        assert torch.equal(grouped,grouped[:,:1].expand_as(grouped))
                        vv=grouped[:,0]
                    captured.append((attn,vv.detach()))
                    return output
                if detail:F.scaled_dot_product_attention=capture
                try:
                    engine.invoke(int(ids.shape[-1]),lambda:model(input_ids=ids,attention_mask=full_mask,
                        past_key_values=copy,cache_position=cp,use_cache=True,return_dict=True),
                        context(item,receiver=receiver,kind='required_receiver_probe'),
                        dict(prompt=prompt,input_ids=ids[0].cpu().tolist(),past_positions=before),expected_past=copy)
                finally:F.scaled_dot_product_attention=previous
            rec=dict(engine.last_receipt)
            assert proof['forwards']==1 and proof['positions']==rec['actual_positions']
            if detail:
                assert len(captured)==36
                return captured,rec
            layers=[]
            for a,n in proof['query_attention']:
                a=a/n
                a=a.reshape(1,model.config.num_key_value_heads,-1,a.shape[-1]).mean(2)
                layers.append(a[0])
            return dict(kind='receiver_probe',layers=layers),rec
        def judge(item,cache,absolute,mode):
            prompt,ids,mask=engine.prepare(ew.stage_prompt('judger',item['question'],args_for(item['dataset'],runner.model_name)))
            before=cache.get_seq_length();cap=8192
            assert absolute+ids.shape[-1]+cap<=32768
            full_mask=torch.cat([torch.ones(1,before,dtype=mask.dtype,device=mask.device),mask],-1)
            cp=torch.arange(before,before+ids.shape[-1],device=ids.device)
            with observe(model,absolute,latent=False,prefix=pairs(cache)) as proof:
                value=engine.invoke(int(ids.shape[-1])+cap,lambda:model.generate(input_ids=ids,
                    attention_mask=full_mask,past_key_values=cache,cache_position=cp,max_new_tokens=cap,
                    temperature=.6,top_p=.95,top_k=20,do_sample=True,pad_token_id=engine.tokenizer.pad_token_id,
                    return_dict_in_generate=True,output_scores=False),context(item,receiver='Judger',method=mode),
                    dict(prompt=prompt,input_ids=ids[0].cpu().tolist(),past_positions=before,max_new_tokens=cap),expected_past=cache)
            rec=dict(engine.last_receipt);assert rec['actual_positions']==proof['positions']
            gen=value.sequences[0,ids.shape[-1]:]
            answer=engine.tokenizer.decode(gen,skip_special_tokens=True).strip()
            assert 0<len(gen)<=cap
            rec['position_proof']={k:proof[k] for k in ['forwards','positions','sdpa_checks']}
            return answer,rec
        def final_select(cache,mapping,stages,captured,mode):
            ref=stages[-1];xs=pairs(cache);dev=xs[0][0].device;heads=8;total=xs[0][0].shape[-2]
            fresh=torch.arange(ref['original_before'],ref['original_end'],device=dev)
            allmap=torch.cat([mapping,fresh[None,None,:].expand(36,heads,-1)],-1)
            chosen=[];reports=[]
            for layer,(a,v) in enumerate(captured):
                pos=allmap[layer];n=a.shape[-1]
                labels=torch.zeros(heads,n,dtype=torch.long,device=dev)
                protected=torch.ones(heads,n,dtype=torch.bool,device=dev)
                for r,s in enumerate(stages,1):
                    labels[:,:total][(pos>=s['original_before'])&(pos<s['original_end'])]=r
                    prompt=(pos>=max(4,s['original_before']))&(pos<s['original_before']+s['prompt_len'])
                    protected[:,:total][prompt]=False
                labels[:,:total][pos<4]=0
                candidate=~protected;mandatory=torch.zeros_like(protected)
                mandatory[:,:total]=(pos>=ref['original_before']+ref['prompt_len']-8)&(pos<ref['original_before']+ref['prompt_len'])
                keep,report=select(a,v,labels,protected,candidate,mandatory,96,mode)
                ix=torch.stack([torch.where(x[:total])[0] for x in keep])
                assert ix.shape==(8,130)
                chosen.append(ix);reports.append(report)
            indices=torch.stack(chosen)
            result=make_cache(tuple(tuple(torch.gather(t,2,indices[l][None,:,:,None].expand(1,8,130,t.shape[-1])).contiguous()
                for t in pair) for l,pair in enumerate(xs)))
            return result,torch.gather(allmap,2,indices),indices,reports,allmap
        while True:
            with qlock():
                queue=load(RUN/'queue.json');task=choose(queue['tasks'])
                if task is not None:
                    task.update(state='running',owner_pid=os.getpid(),attempt=str(out),started=time.time())
                    save(RUN/'queue.json',queue);current_task=dict(task)
            if task is None:break
            item=items[task['item_index']];repeat=1
            if True:
                current=dict(question_id=item['question_id'],repeat=repeat);record('current.json',current)
                verify();seed_all(42);torch.cuda.reset_peak_memory_stats();wall_start=time.monotonic()
                cache=None;mapping=None;absolute=0;stages=[];prefix_times=[];prefix_transfers=[];shared_rids=[]
                for idx,role in enumerate(['planner','critic','refiner']):
                    (cache,rec,latent_a),seconds=timed(lambda:stage(item,role,cache,absolute))
                    del latent_a
                    stages.append(rec);shared_rids.append(rec['reservation_id']);absolute=rec['original_end']
                    segment=dict(role=role,latent_seconds=seconds)
                    if idx<2:
                        (attention,pr),pt=timed(lambda:probe(item,['critic','refiner'][idx],cache,absolute))
                        shared_rids.append(pr['reservation_id'])
                        (cache,mapping,transfer),ct=timed(lambda:compress(cache,mapping,role=role,steps=10,
                            prompt_len=rec['prompt_len'],original_before=rec['original_before'],method='receiver_attention_sink_recent',attention=attention))
                        del attention
                        segment.update(required_probe_seconds=pt,compression_outer_seconds=ct)
                        prefix_transfers.append(transfer)
                    prefix_times.append(segment)
                prefix_seconds=time.monotonic()-wall_start
                reference=reference_T[item['question_id']]
                assert [x['selected_position_sha256'] for x in prefix_transfers]==[x['selected_position_sha256'] for x in reference['prefix_transfers']], 'T prefix position mismatch; preserve scene'
                assert [[x[k] for k in ['original_before','original_end','prompt_len']] for x in stages]==[[x[k] for k in ['original_before','original_end','prompt_len']] for x in reference['prefix_stages']], 'T stage length mismatch'
                record('prefix_match.json',dict(**current,matched_T_prefix_positions=True))
                (captured,pr),probe_seconds=timed(lambda:probe(item,'judger',cache,absolute,True))
                shared_rids.append(pr['reservation_id'])
                rng=(torch.random.get_rng_state(),torch.cuda.get_rng_state_all())
                order=['G'];order=order[repeat-1:]+order[:repeat-1]
                selections={}
                for mode in order:
                    if mode not in current_task['methods']:continue
                    record('current.json',dict(**current,method=mode,phase='selector'))
                    (compact,position,indices,diagnostics,allmap),selection_seconds=timed(lambda:final_select(cache,mapping,stages,captured,mode))
                    def validate():
                        for l,(old,new) in enumerate(zip(pairs(cache),pairs(compact))):
                            for original,actual in zip(old,new):
                                assert torch.equal(torch.gather(original,2,indices[l][None,:,:,None].expand_as(actual)),actual)
                        assert bool(((position<4).sum(-1)==4).all())
                        for s in stages:
                            assert bool((((position>=s['original_end']-10)&(position<s['original_end'])).sum(-1)==10).all())
                    _,audit_seconds=timed(validate)
                    actual_kv_bytes=sum(t.numel()*t.element_size() for pair in pairs(compact) for t in pair)
                    assert actual_kv_bytes==19169280
                    selections[mode]=position.detach().cpu()
                    pos_sha=hashlib.sha256(position.cpu().contiguous().numpy().tobytes()).hexdigest()
                    io_start=time.monotonic()
                    with (out/(item['question_id']+'_r'+str(repeat)+'_'+mode+'_positions.npz')).open('wb') as f:
                        np.savez_compressed(f,positions=position.cpu().numpy(),storage_indices=indices.cpu().numpy())
                    io_seconds=time.monotonic()-io_start
                    torch.random.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1])
                    record('current.json',dict(**current,method=mode,phase='judger'))
                    (answer,jrec),judge_seconds=timed(lambda:judge(item,compact,absolute,mode))
                    pred=normalize_answer(extract_gsm8k_answer(answer))
                    token_ids=jrec['generated_ids']
                    row=dict(**current,dataset=item['dataset'],is_diamond=item['is_diamond'],gold=item['gold'],correct=bool(pred and pred==normalize_answer(str(item['gold']))),method=mode,status='ok',engineering_only=False,latent_steps=10,seed=42,
                        answer_text=answer,prediction=pred,matched_T_prefix_positions=True,parser_sha=sha(RUN/'author_parser.py'),generated_tokens=len(token_ids),
                        token_sha=hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
                        selected_position_sha=pos_sha,retained_positions=130,retained_kv_bytes=actual_kv_bytes,
                        source_k_equal_measured=True,source_v_equal_measured=True,protected_latent_sink=True,
                        positions_shape=list(position.shape),original_end=absolute,full_bytes_from_original_positions=absolute*147456,
                        selector_diagnostics=diagnostics,shared_request_ids=shared_rids,judger_receipt=jrec,
                        prefix_stages=stages,prefix_transfers=prefix_transfers,
                        timing=dict(prefix_instrumented_wall_seconds=prefix_seconds,prefix_components=prefix_times,
                            required_shared_judger_probe_seconds=probe_seconds,selector_and_gather_seconds=selection_seconds,
                            independent_tensor_audit_seconds=audit_seconds,position_io_seconds=io_seconds,
                            judger_seconds=judge_seconds,reconstructed_standalone_seconds=prefix_seconds+probe_seconds+selection_seconds+judge_seconds,
                            reconstructed=True,cuda_sync_both_boundaries=True,audit_hooks_present=True,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                    with qlock():
                        assert key(row) not in {key(x) for x in all_rows()}, 'duplicate new answer'
                        append(out/'results.jsonl',row)
                    count+=1
                    record('last_commit.json',dict(**current,method=mode,count=count))
                    del compact,position,indices,allmap;gc.collect()
                # Diagnostic only: compare per-head retained ORIGINAL coordinates.
                overlap={}
                for x,y in [('T','R'),('R','R-shuffle')]:
                    if x not in selections or y not in selections:continue
                    vals=[]
                    for a,b in zip(selections[x].reshape(-1,130),selections[y].reshape(-1,130)):
                        aa=set(a.tolist());bb=set(b.tolist());vals.append(len(aa&bb)/len(aa|bb))
                    overlap[x+'__'+y]=dict(mean=float(np.mean(vals)),median=float(np.median(vals)))
                append(out/'repeat_checks.jsonl',dict(**current,jaccard=overlap,actual_repeat_instrumented_wall=time.monotonic()-wall_start))
                del captured,cache,mapping,selections;gc.collect();torch.cuda.empty_cache()
                with qlock():
                    queue=load(RUN/'queue.json');task=next(t for t in queue['tasks'] if t['id']==current_task['id'])
                    assert task['owner_pid']==os.getpid()
                    assert {(item['dataset'],item['question_id'],10,m) for m in task['methods']} <= {key(x) for x in all_rows()}
                    task.update(state='complete',ended=time.time());save(RUN/'queue.json',queue)
                record('last_task_commit.json',dict(task=current_task['id']))
                current_task=None
        record('exit.json',dict(completed=True,committed=count,released_idle_tail=True))
    except BaseException as exc:
        if current_task is not None:
            with qlock():
                queue=load(RUN/'queue.json');task=next(t for t in queue['tasks'] if t['id']==current_task['id'])
                task.update(state='failed_held',failure=repr(exc),ended=time.time());save(RUN/'queue.json',queue)
        record('failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),current=current,committed=count))
        record('exit.json',dict(completed=False,committed=count,error=repr(exc)))
        raise

if __name__=='__main__':main()
