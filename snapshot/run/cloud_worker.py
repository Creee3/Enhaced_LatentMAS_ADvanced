import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
from cloud_common import *
sys.path.insert(0,str(BASE/'runtime'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);current=None;count=0
    def receipt(name,x):save(out/name,dict(unix=time.time(),identity=identity(os.getpid()),**x))
    receipt('loading.json',dict(gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),phase='source_preflight'))
    try:
        source=verify();plan=load(RUN/'plan.json');items=load(RUN/'questions.json')
        assert os.environ['LATENTMAS_SHARED_LEDGER']==str(ROOT/'cumulative_budget.json')
        import numpy as np
        import torch
        import experiment_worker as ew
        from runtime_common import Ledger,seed_all
        from stage_ops import pairs,clone_cache,observe,compress,load_obf
        ROLES=('planner','critic','refiner')
        from author_parser import extract_gsm8k_answer,normalize_answer
        torch.set_num_threads(2)
        original_check=Ledger.check
        def check(self):
            original_check(self)
            if (RUN/'STOP').exists():raise RuntimeError('user STOP requested')
        Ledger.check=check
        def args_for(ds,model_name,method='latent_mas'):
            return SimpleNamespace(task='gpqa' if ds.startswith('gpqa') else 'gsm8k',model_name=model_name,prompt='sequential',think=False,method=method,text_mas_context_length=-1,latent_space_realign=False)
        ew.args_for=args_for
        runner=ew.Runner(str(out),torch.device('cuda:0'));engine=runner.engine;model=engine.model
        engine.config.update(experiment_id=RUN.name,request_timeout_seconds=900);engine.phase=RUN.name
        matrix,target=engine.core._latent_realign_matrices[id(model)]
        ready=dict(source=source,weights_verified=len(load(BASE/'runtime/model_lock.json')['files']),
            full_bf16_cuda=all(p.dtype==torch.bfloat16 and p.device.type=='cuda' for p in model.parameters()),
            parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),matrix_is_identity=torch.equal(matrix,torch.eye(matrix.shape[0],device=matrix.device,dtype=matrix.dtype)),
            alignment='norm_only',target_norm=float(target),seed=42,latent_lengths=[10,40],output_caps=plan['output_caps'])
        assert ready['weights_verified']==15 and ready['full_bf16_cuda'] and ready['parameter_bytes']==16381470720 and ready['matrix_is_identity'] and not engine.core.args.latent_space_realign
        receipt('model_ready.json',ready)
        def npz(path,**arrays):
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
            tmp.replace(path)
        def stage(item,role,cache,absolute,steps,method):
            before=0 if cache is None else cache.get_seq_length()
            messages=ew.stage_prompt(role,item['question'],args_for(item['dataset'],runner.model_name))
            _,ids,_=engine.prepare(messages);prompt=int(ids.shape[-1])
            if absolute+prompt+steps>32768:raise RuntimeError('original context cap')
            with observe(model,absolute,latent=True,prefix=pairs(cache)) as proof:
                past,rec,_=engine.latent(messages,steps,dict(dataset=item['dataset'],question_id=item['question_id'],receiver=role,method=method),past=cache)
            assert proof['forwards']==steps+1 and len(proof['attention_steps'])==steps and proof['positions']==rec['actual_positions']
            rec.update(role=role,prompt_len=prompt,storage_before=before,original_before=absolute,original_end=absolute+prompt+steps,latent_steps=steps,
                position_proof=dict(forwards=proof['forwards'],positions=proof['positions'],sdpa_checks=proof['sdpa_checks']))
            return past,rec,proof['attention_steps']
        def judge(item,cache,absolute,steps,method):
            cap=plan['output_caps'][item['dataset']]
            messages=ew.stage_prompt('judger',item['question'],args_for(item['dataset'],runner.model_name))
            prompt,ids,mask=engine.prepare(messages);before=cache.get_seq_length()
            if absolute+ids.shape[-1]+cap>32768:raise RuntimeError('original Judger cap')
            full_mask=torch.cat([torch.ones(1,before,dtype=mask.dtype,device=mask.device),mask],-1)
            cp=torch.arange(before,before+ids.shape[-1],device=ids.device)
            with observe(model,absolute,latent=False,prefix=pairs(cache)) as proof:
                result=engine.invoke(int(ids.shape[-1])+cap,
                    lambda:model.generate(input_ids=ids,attention_mask=full_mask,past_key_values=cache,cache_position=cp,max_new_tokens=cap,temperature=.6,top_p=.95,top_k=20,do_sample=True,pad_token_id=engine.tokenizer.pad_token_id,return_dict_in_generate=True,output_scores=False),
                    dict(dataset=item['dataset'],question_id=item['question_id'],receiver='Judger',method=method,latent_steps=steps),
                    dict(prompt=prompt,input_ids=ids[0].cpu().tolist(),past_positions=before,original_history=absolute,max_new_tokens=cap),expected_past=cache)
            rec=dict(engine.last_receipt);assert proof['positions']==rec['actual_positions'] and proof['forwards']==rec['native_forward_calls']
            gen=result.sequences[0,ids.shape[-1]:];answer=engine.tokenizer.decode(gen,skip_special_tokens=True).strip()
            rec['position_proof']=dict(forwards=proof['forwards'],positions=proof['positions'],sdpa_checks=proof['sdpa_checks'],original_history=absolute,storage_history=before)
            return answer,rec
        def receiver_probe(item,receiver,cache,absolute,steps,method):
            """Run a metered, no-generation receiver prompt probe on a cache clone.

            The probe reads the receiver's last 16 prompt queries and returns
            per-layer/head attention over the current producer cache. It is a
            new method's measured cost; it never changes the producer cache or
            the frozen Full/HeadOBF/JIVE results.
            """
            messages=ew.stage_prompt(receiver,item['question'],args_for(item['dataset'],runner.model_name))
            prompt,ids,mask=engine.prepare(messages)
            before=cache.get_seq_length()
            probe_cache=clone_cache(cache)
            if absolute+ids.shape[-1]>32768:raise RuntimeError('receiver probe context cap')
            full_mask=torch.cat([torch.ones(1,before,dtype=mask.dtype,device=mask.device),mask],-1)
            cp=torch.arange(before,before+ids.shape[-1],device=ids.device)
            with observe(model,absolute,latent=False,prefix=pairs(cache),capture_queries=16) as proof:
                engine.invoke(int(ids.shape[-1]),
                    lambda:model(input_ids=ids,attention_mask=full_mask,past_key_values=probe_cache,
                        cache_position=cp,use_cache=True,return_dict=True),
                    dict(dataset=item['dataset'],question_id=item['question_id'],receiver=receiver,
                         method=method,latent_steps=steps,kind='receiver_attention_probe'),
                    dict(prompt=prompt,input_ids=ids[0].cpu().tolist(),past_positions=before,
                         probe_query_count=min(16,int(ids.shape[-1]))),expected_past=probe_cache)
            layers=[]
            for a,count in proof['query_attention']:
                a=a/count
                # GQA probe scores are reduced to KV-head scores used by cache rows.
                if a.shape[1]!=model.config.num_key_value_heads:
                    a=a.reshape(1,model.config.num_key_value_heads,
                                a.shape[1]//model.config.num_key_value_heads,a.shape[-1]).mean(2)
                layers.append(a[0])
            if len(layers)!=len(model.model.layers):raise RuntimeError('receiver probe layer count drift')
            return dict(kind='receiver_probe',layers=layers,receiver=receiver,
                        probe_query_count=min(16,int(ids.shape[-1])),prompt_len=int(ids.shape[-1]),
                        probe_forwards=proof['forwards'],probe_positions=proof['positions'])
        while True:
            with qlock():
                q=load(RUN/'queue.json');task=choose(q['tasks']);finished=terminal(q['tasks'])
                if task:
                    task.update(state='running',owner_pid=os.getpid(),attempt=str(out),gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),started=time.time())
                    save(RUN/'queue.json',q);current=dict(task)
            if not task:
                # No pending task can appear after all have been claimed. Release
                # idle resident models in the tail instead of holding the GPUs.
                if finished or not any(t['state']=='pending' for t in q['tasks']):break
                time.sleep(2);continue
            verify();receipt('current_task.json',current)
            item=items[current['item_index']];steps=current['steps']
            assert item['split']=='development' and 'gold' in item
            seed_all(42);planner,prec,patt=stage(item,'planner',None,0,steps,'shared_planner')
            rng=(torch.random.get_rng_state(),torch.cuda.get_rng_state_all())
            methods=list(current['methods'])
            for method in methods:
                result_key=(item['dataset'],item['question_id'],steps,method)
                with qlock():already={key(r) for r in all_rows()}
                if result_key in already:continue
                torch.random.set_rng_state(rng[0]);torch.cuda.set_rng_state_all(rng[1])
                cache=clone_cache(planner);mapping=None;stages=[prec];transfers=[];past_ranges=[];probe_receipts=[]
                obf=load_obf() if method=='headobf' else None
                started=time.monotonic();selection_seconds=0.
                for index,role in enumerate(ROLES):
                    if index==0:rec=prec;attention=patt
                    else:
                        cache,rec,attention=stage(item,role,cache,absolute,steps,method);stages.append(rec)
                    absolute=rec['original_end'];sel_start=time.monotonic()
                    if method=='receiver_attention_sink_recent':
                        receiver=('critic','refiner','judger')[index]
                        attention=receiver_probe(item,receiver,cache,absolute,steps,method)
                        probe_receipts.append({k:v for k,v in attention.items() if k!='layers'})
                    cache,mapping,transfer=compress(cache,mapping,role=role,steps=steps,prompt_len=rec['prompt_len'],original_before=rec['original_before'],method=method,attention=attention,obf=obf,past_ranges=past_ranges)
                    selection_seconds+=time.monotonic()-sel_start
                    transfers.append(transfer)
                    past_ranges.append((rec['original_before'],rec['original_before']+rec['prompt_len']))
                    npz(out/(item['question_id']+'_l'+str(steps)+'_'+method+'_'+role+'_positions.npz'),positions=mapping.detach().cpu().numpy())
                    if index:del attention
                history=int(cache.get_seq_length());jstart=time.monotonic()
                answer,jrec=judge(item,cache,absolute,steps,method);judge_seconds=time.monotonic()-jstart
                stages.append(jrec);ids=jrec['generated_ids'];cap=plan['output_caps'][item['dataset']]
                eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos or [])
                pred=normalize_answer(extract_gsm8k_answer(answer));gold=normalize_answer(str(item['gold']))
                assert 0<len(ids)<=cap
                if method!='full':
                    assert [t['retained'] for t in transfers]==[4+k*(32+steps) for k in [1,2,3]]
                    assert all(t['source_k_equal'] and t['protected_latent_and_sink'] for t in transfers)
                if method=='jive':
                    assert all(t['source_v_equal'] and t['external_questions_used']==0 for t in transfers)
                    assert transfers[0]['decompositions'][0]['available_views']==1
                    assert all(d['available_views']==2 for d in transfers[1]['decompositions'])
                    assert all(d['available_views']==3 for d in transfers[2]['decompositions'])
                row=dict(dataset=item['dataset'],question_id=item['question_id'],latent_steps=steps,method=method,
                    seed=42,alignment='norm_only',answer_text=answer,prediction=pred,gold=gold,
                    correct=bool(pred and pred==gold),max_new_tokens=cap,generated_tokens=len(ids),
                    truncated=len(ids)>=cap and ids[-1] not in eos,stages=stages,transfers=transfers,
                    retained_positions=history,source=source,worker_pid=os.getpid(),external_questions_used=0,
                    calibration_model_used=False,algorithm='within_question_JIVE_geometry' if method=='jive' else method,
                    timing=dict(case_arm_seconds=time.monotonic()-started,selection_seconds=selection_seconds,judge_seconds=judge_seconds),
                    protocol='current question only; receiver prompt attention probe; actual retained past prompts; all latent preserved',
                    receiver_probes=probe_receipts,
                    status='ok',parser=plan['author_parser_revision'])
                with qlock():
                    if result_key in {key(r) for r in all_rows()}:raise RuntimeError('duplicate answer')
                    append(out/'results.jsonl',row)
                count+=1;receipt('last_commit.json',dict(key=list(result_key),count=count,task=current['id']))
                receipt('formal_'+method+'.json',dict(passed=True,current_attempt_key=list(result_key),model_ready=ready,
                    source=source,transfers=transfers,judger_position_proof=jrec['position_proof'],external_questions_used=0))
                del cache,mapping
            del planner,patt
            with qlock():
                q=load(RUN/'queue.json');t=next(t for t in q['tasks'] if t['id']==current['id'])
                assert t['owner_pid']==os.getpid()
                required={(item['dataset'],item['question_id'],steps,m) for m in methods}
                assert required<={key(r) for r in all_rows()}
                t.update(state='complete',ended=time.time());save(RUN/'queue.json',q)
            receipt('last_task_commit.json',dict(task=current['id'],kind='direct_development_case'))
            current=None;gc.collect()
        q=load(RUN/'queue.json')
        receipt('exit.json',dict(completed=all(t['state']=='complete' for t in q['tasks']),committed=count,
            released_idle_tail=not any(t['state']=='pending' for t in q['tasks']),blocked_by_held=any(t['state']=='failed_held' for t in q['tasks'])))
    except BaseException as e:
        receipt('failure.json',dict(error=repr(e),task=current))
        if current:
            with qlock():
                q=load(RUN/'queue.json');t=next(t for t in q['tasks'] if t['id']==current['id'])
                t.update(state='failed_held',failure=repr(e),ended=time.time());save(RUN/'queue.json',q)
        receipt('exit.json',dict(completed=False,error=repr(e),committed=count));raise

if __name__=='__main__':main()

