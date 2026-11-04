"""One explicitly authorized CPU feasibility answer, never a formal G/baseline row."""
import hashlib,json,os,resource,sys,time,traceback
from pathlib import Path

RUN=Path(__file__).resolve().parent
ROOT=RUN.parents[1]
BASE=ROOT/'evidence/gpqa_repair_8096_20261028T161254Z/bundle'
sys.path.insert(0,str(BASE/'runtime'))
from runtime_common import save,append,sha,verify_source,Ledger


def trace(event,**values):append(RUN/'trace.jsonl',dict(unix=time.time(),event=event,**values))


def main():
    started=time.monotonic()
    try:
        assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
        save(RUN/'state.json',dict(phase='verify_sources_and_weights',pid=os.getpid()))
        source=verify_source();lock=json.loads((RUN/'run_lock.json').read_text())
        assert sha(RUN/'worker.py')==lock['worker_sha256']
        assert sha(RUN/'question.json')==lock['question_sha256']
        model_lock=json.loads((BASE/'runtime/model_lock.json').read_text())
        for name,entry in model_lock['files'].items():
            p=Path(model_lock['path'])/name
            assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],name
            trace('model_file_verified',file=name)
        assert len(model_lock['files'])==15
        save(RUN/'state.json',dict(phase='import_and_load_cpu_model',pid=os.getpid()))
        import torch
        torch.set_num_threads(16);torch.set_num_interop_threads(1)
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from inference_engine import Engine
        from author_parser import extract_gsm8k_answer,normalize_answer
        model_start=time.monotonic()
        tokenizer=AutoTokenizer.from_pretrained(model_lock['path'],local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(model_lock['path'],local_files_only=True,
            torch_dtype=torch.bfloat16,device_map={'':'cpu'},attn_implementation='sdpa').eval()
        load_seconds=time.monotonic()-model_start
        assert all(p.device.type=='cpu' and p.dtype==torch.bfloat16 for p in model.parameters())
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
        assert parameter_bytes==16381470720
        config=dict(temperature=.6,top_p=.95,top_k=20,max_total_context_positions=32768,
            request_timeout_seconds=300,seed=42,experiment_id=RUN.name,
            budget=dict(ledger_path=str(ROOT/'cumulative_budget.json'),cumulative_position_limit=115000000))
        # Retain original engine metering; only add a stop check for this new run.
        original_check=Ledger.check
        def check(self):
            original_check(self)
            if (RUN/'STOP').exists():raise RuntimeError('CPU pilot STOP requested')
        Ledger.check=check
        engine=Engine(config,trace,RUN/'raw',model=model,tokenizer=tokenizer)
        engine.phase='cpu_single_question_feasibility'
        item=json.loads((RUN/'question.json').read_text(encoding='utf-8'))
        messages=[dict(role='system',content='You are a helpful assistant.'),dict(role='user',
            content='Choose the correct option for the following question. Output only the final answer as \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}. Do not provide an explanation.\n\n'+item['question'])]
        prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=tokenizer(prompt,return_tensors='pt',add_special_tokens=False)
        n=int(inputs['input_ids'].shape[-1]);cap=64
        assert n+cap<=32768
        save(RUN/'preflight.json',dict(model_forward_calls=0,input_tokens=n,cap=cap,bound=n+cap,
            parameter_bytes=parameter_bytes,device='cpu',dtype='bfloat16',threads=16,
            enable_thinking=False,hash_files=15,bundle_files=len(source['files']),load_seconds=load_seconds))
        progress=[]
        def timing_hook(module,args,kwargs,output):progress.append(time.monotonic())
        handle=model.register_forward_hook(timing_hook,with_kwargs=True)
        torch.manual_seed(42)
        inference_start=time.monotonic()
        save(RUN/'state.json',dict(phase='single_question_generation',pid=os.getpid()))
        with torch.inference_mode():
            value=engine.invoke(n+cap,lambda:model.generate(**inputs,max_new_tokens=cap,
                do_sample=True,temperature=.6,top_p=.95,top_k=20,pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,output_scores=False),
                dict(dataset=item['dataset'],question_id=item['question_id'],method='cpu_feasibility_direct_answer',device='cpu'),
                dict(prompt=prompt,input_ids=inputs['input_ids'][0].tolist(),past_positions=0,max_new_tokens=cap))
        elapsed=time.monotonic()-inference_start;handle.remove()
        ids=value.sequences[0,n:].tolist()
        assert ids==engine.last_receipt['generated_ids'] and 0<len(ids)<=64
        answer=tokenizer.decode(ids,skip_special_tokens=True).strip()
        pred=normalize_answer(extract_gsm8k_answer(answer))
        eos=model.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos or [])
        hit_cap=len(ids)==cap and ids[-1] not in eos
        result=dict(completed=True,engineering_only=True,formal_G_result=False,model='Qwen3-8B',
            device='cpu',dtype='bfloat16',parameter_bytes=parameter_bytes,threads=16,cap=64,
            enable_thinking=False,input_tokens=n,generated_tokens=len(ids),hit_cap=hit_cap,
            prediction=pred,correct=bool(pred and pred==normalize_answer(str(item['gold']))),
            load_seconds=load_seconds,preparation_seconds=inference_start-started,
            inference_seconds=elapsed,first_forward_seconds=progress[0]-inference_start,
            generation_tokens_per_second_including_prefill=len(ids)/elapsed,
            decode_tokens_per_second=(len(ids)-1)/(progress[-1]-progress[0]) if len(progress)>1 else None,
            peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
            total_seconds=time.monotonic()-started,request_receipt=engine.last_receipt,
            cpu_only=all(p.device.type=='cpu' for p in model.parameters()),seed=42,
            parser_sha256=sha(RUN/'author_parser.py'))
        save(RUN/'answer_private.json',dict(answer_text=answer,generated_ids=ids,question_id=item['question_id']))
        save(RUN/'result.json',result)
        save(RUN/'state.json',dict(phase='complete',pid=os.getpid()))
        print(json.dumps({k:v for k,v in result.items() if k!='request_receipt'}),flush=True)
    except BaseException as exc:
        save(RUN/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),unix=time.time()))
        save(RUN/'state.json',dict(phase='failed',pid=os.getpid()))
        raise


if __name__=='__main__':main()
