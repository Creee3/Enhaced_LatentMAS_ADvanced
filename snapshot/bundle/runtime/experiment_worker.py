import argparse, copy, gc, hashlib, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
import torch
from inference_engine import Engine, InvalidGeneration
from kv_packet import roundtrip, sparse_roundtrip, signature
from runtime_common import append, save, seed_all, sha, verify_source
import runpy

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'data'/'questions.jsonl'
PROMPTS=runpy.run_path(str(ROOT/'vendor'/'LatentMAS'/'prompts.py'))
AGENTS=[{'name':'Planner','role':'planner'},{'name':'Critic','role':'critic'},{'name':'Refiner','role':'refiner'},{'name':'Judger','role':'judger'}]
C2_PROMPT='''Question: {question}\n\nYou are a second Critic Agent. The preceding Planner, Critic, and Refiner have produced a current plan for solving the question. Evaluate the latest Refiner plan for remaining factual errors, unsupported steps, or omissions. Focus your feedback on the latest Refiner plan, and provide concrete corrections for the next Refiner.\n\nNow, output your critique below:\n'''
R2_PROMPT='''Question: {question}\n\nYou are a second Refiner Agent. You are provided with the latest Refiner plan and the second Critic\'s feedback in latent-format information. Revise the latest plan into a correct, concise step-by-step plan for solving the question. Do not produce the final answer.\n\nNow, output your revised plan below:\n'''

def load_items():
    # JSON permits U+2028 inside a quoted string.  Python splitlines() treats
    # U+2028 as a line boundary, which corrupts one GPQA record.  JSONL here is
    # newline-delimited by literal LF, so split only on '\\n'.
    return [json.loads(x) for x in DATA.read_text(encoding='utf-8').split('\n') if x.strip()]

def active_len(cache): return 0 if cache is None else cache.get_seq_length()
def legacy(cache): return cache.to_legacy_cache() if hasattr(cache,'to_legacy_cache') else cache

def zero_select(cache, keep_positions):
    if cache is None: return None, []
    keep=set(keep_positions); layers=[]
    for pair in legacy(cache):
        out=[]
        for t in pair:
            q=t.detach().clone()
            if q.shape[-2]:
                mask=torch.zeros(q.shape[-2],dtype=torch.bool,device=q.device)
                idx=[i for i in keep if 0<=i<q.shape[-2]]
                if idx: mask[torch.tensor(idx,device=q.device)]=True
                q[...,~mask,:]=0
            out.append(q)
        layers.append(tuple(out))
    from transformers.cache_utils import DynamicCache
    return DynamicCache.from_legacy_cache(tuple(layers)), [1 if i in keep else 0 for i in range(cache.get_seq_length())]

def stage_prompt(role, question, args):
    return PROMPTS['build_agent_message_sequential_latent_mas'](role=role, question=question, context='', method='latent_mas', args=args)

def text_prompt(role, question, context, args):
    return PROMPTS['build_agent_messages_sequential_text_mas'](role=role, question=question, context=context, method='text_mas', args=args)

def single_prompt(question,args): return PROMPTS['build_agent_messages_single_agent'](question=question,args=args)

def args_for(dataset, model_name, method='latent_mas'):
    task='gpqa' if dataset=='gpqa_diamond' else 'arc_easy'
    return SimpleNamespace(task=task,model_name=model_name,prompt='sequential',think=False,method=method,text_mas_context_length=-1,latent_space_realign=False)

def blocks_from_receipts(receipts):
    blocks={};
    for r in receipts:
        blocks[r['role']]=list(range(r['past_before'],r['past_after']))
    return blocks

def answer_result(item, arm, answer, stages, transfers, started, engine):
    txt=answer.strip(); pred=None
    import re
    boxes=re.findall(r'\\boxed\{([^}]*)\}',txt)
    if boxes: pred=boxes[-1].strip().lower()
    else: pred=''
    gold=str(item.get('gold','')).strip().lower()
    ok=(pred==gold)
    return {'dataset':item['dataset'],'question_id':item['question_id'],'condition':arm,'seed':42,'answer_text':txt,'prediction':pred,'gold':gold,'correct':ok,'status':'ok','stages':stages,'transfers':transfers,'elapsed_seconds':time.monotonic()-started,'network_time_measured':False}

class Runner:
    def __init__(self, out, device):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        self.trace_path=self.out/'trace.jsonl'
        ledger_path=Path(os.environ.get('LATENTMAS_SHARED_LEDGER',str(self.out/'budget.json')))
        self.cfg={'temperature':0.6,'top_p':0.95,'top_k':20,'max_total_context_positions':32768,'request_timeout_seconds':300,'seed':42,'experiment_id':'latentmas_cumulative_history_20261025','budget':{'ledger_path':str(ledger_path),'cumulative_position_limit':115000000}}
        if not ledger_path.exists(): save(ledger_path,{'actual_tokens':0,'reserved_tokens':0,'unknown_cost_upper_bound_tokens':0,'new_calls':0,'native_forward_calls':0,'active_reservations':{}})
        self.engine=Engine(self.cfg,self.trace,self.out/'raw')
        self.device=device
        self.model_name='/home/liuxin/models/Qwen3-8B'
        self.args_cache={}
    def trace(self,event,**fields): append(self.trace_path,dict(unix=time.time(),event=event,**fields))
    def stage_latent(self,item,role,past,steps):
        args=args_for(item['dataset'],self.model_name); msgs=stage_prompt(role,item['question'],args)
        before=active_len(past); new,rec,prompt=self.engine.latent(msgs,steps,dict(dataset=item['dataset'],question_id=item['question_id'],receiver=role),past=past)
        after=active_len(new); rec.update(role=role,past_before=before,past_after=after,prompt=prompt,latent_steps=steps)
        return new,rec
    def judge(self,item,past,past_mask,condition):
        args=args_for(item['dataset'],self.model_name); msgs=stage_prompt('judger',item['question'],args)
        if past is None:
            ans,tr,rec,prompt=self.engine.text(msgs,2048,dict(dataset=item['dataset'],question_id=item['question_id'],receiver='Judger',condition=condition),past=None)
        else:
            # use direct generate with sparse mask while retaining original cache coordinates
            prompt,ids,mask=self.engine.prepare(msgs); before=past.get_seq_length()
            full=torch.cat([torch.tensor(past_mask,dtype=mask.dtype,device=mask.device).view(1,-1),mask],dim=1)
            cache_position=torch.arange(before,before+ids.shape[-1],device=ids.device)
            bound=int(ids.shape[-1])+2048
            val=self.engine.invoke(bound,lambda:self.engine.model.generate(input_ids=ids,attention_mask=full,max_new_tokens=2048,temperature=0.6,top_p=0.95,do_sample=True,pad_token_id=self.engine.tokenizer.pad_token_id,return_dict_in_generate=True,output_scores=False,past_key_values=past,cache_position=cache_position),dict(dataset=item['dataset'],question_id=item['question_id'],receiver='Judger',condition=condition),dict(prompt=prompt,input_ids=ids[0].detach().cpu().tolist(),past_positions=before,max_new_tokens=2048),expected_past=past)
            gen=val.sequences[0,ids.shape[-1]:]; ans=self.engine.tokenizer.decode(gen,skip_special_tokens=True).strip(); tr=False; rec=dict(self.engine.last_receipt); rec.update(prompt=prompt,masked_past_positions=sum(past_mask),past_positions=before)
        return ans,rec
    def build_upstream(self,item,steps=10):
        seed_all(42); past=None; stages=[]; transfers=[]
        for role in ['planner','critic','refiner']:
            past,rec=self.stage_latent(item,role,past,steps); stages.append(rec)
            restored,info=roundtrip(past,dict(dataset=item['dataset'],question_id=item['question_id'],sender=role,receiver='next',condition='upstream'))
            past=restored; transfers.append(info)
        return past,stages,transfers
    def run_latent_from_snapshot(self,item,condition,past,upstream_stages,upstream_transfers,rng_state):
        torch.random.set_rng_state(rng_state[0]); torch.cuda.set_rng_state_all(rng_state[1])
        started=time.monotonic(); stages=[dict(x) for x in upstream_stages]; transfers=[dict(x) for x in upstream_transfers]
        blocks=blocks_from_receipts(stages); allpos=list(range(active_len(past)))
        keepmap={'H0':[],'H1':blocks['refiner'],'H2':blocks['critic']+blocks['refiner'],'H3':allpos,'P':blocks['planner'],'C':blocks['critic'],'PR':blocks['planner']+blocks['refiner'],'PC':blocks['planner']+blocks['critic']}
        chosen=keepmap[condition]
        if chosen:
            selected,mask,sparse_info=sparse_roundtrip(past,chosen,dict(dataset=item['dataset'],question_id=item['question_id'],sender='refiner',receiver='judger',condition=condition))
        else:
            selected,mask,sparse_info=None,[],None
        ans,jrec=self.judge(item,selected,mask,condition); stages.append(jrec)
        if sparse_info is not None:
            transfers.append(sparse_info)
        # The receiver keeps original cache coordinates; dropped rows are zero-filled and masked.
        return answer_result(item,condition,ans,stages,transfers,started,self.engine)

    def run_single(self,item):
        started=time.monotonic(); seed_all(42); args=args_for(item['dataset'],self.model_name,method='baseline'); msgs=single_prompt(item['question'],args); ans,tr,rec,prompt=self.engine.text(msgs,2048,dict(dataset=item['dataset'],question_id=item['question_id'],receiver='Single'),None); rec['prompt']=prompt; return answer_result(item,'baseline',ans,[rec],[],started,self.engine)
    def run_text(self,item):
        started=time.monotonic(); seed_all(42); args=args_for(item['dataset'],self.model_name,method='text_mas'); history=''; stages=[]; ans=''
        for role in ['planner','critic','refiner','judger']:
            msgs=text_prompt(role,item['question'],history,args); ans,tr,rec,prompt=self.engine.text(msgs,2048,dict(dataset=item['dataset'],question_id=item['question_id'],receiver=role),None); rec['prompt']=prompt; rec['role']=role; stages.append(rec)
            if role!='judger': history += f'[{role.title()}]:\n{ans.strip()}\n\n'
        return answer_result(item,'text_mas',ans,stages,[],started,self.engine)
    def run_round3(self,item,condition):
        started=time.monotonic(); seed_all(42); steps=10 if condition=='n6_fixed10' else 6; past=None; stages=[]
        for role in ['planner','critic','refiner','critic2','refiner2']:
            if role=='critic2': msgs=[{'role':'system','content':'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'},{'role':'user','content':C2_PROMPT.format(question=item['question'])}]; actual='critic2'
            elif role=='refiner2': msgs=[{'role':'system','content':'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'},{'role':'user','content':R2_PROMPT.format(question=item['question'])}]; actual='refiner2'
            else: actual=role; msgs=stage_prompt(role,item['question'],args_for(item['dataset'],self.model_name))
            before=active_len(past); past,rec,prompt=self.engine.latent(msgs,steps,dict(dataset=item['dataset'],question_id=item['question_id'],receiver=role,condition=condition),past=past); after=active_len(past); rec.update(role=role,past_before=before,past_after=after,prompt=prompt,latent_steps=steps); stages.append(rec)
        ans,jrec=self.judge(item,past,[1]*active_len(past),condition); stages.append(jrec)
        return answer_result(item,condition,ans,stages,[],started,self.engine)
    def run(self, items, conditions, shard, shards):
        results=self.out/'results.jsonl'; done=set()
        if results.exists():
            for line in results.read_text(encoding='utf-8').split('\\n'):
                if line.strip():
                    x=json.loads(line); done.add((x['question_id'],x['condition']))
        shard_items=[x for i,x in enumerate(items) if i%shards==shard]
        target=sum(sum((q['question_id'],c) not in done for c in conditions) for q in shard_items); count=0
        for item in shard_items:
            needed=[c for c in conditions if (item['question_id'],c) not in done]
            if not needed: continue
            # Controls are independent author baselines; latent conditions share one upstream snapshot for this question.
            if 'baseline' in needed:
                try: r=self.run_single(item)
                except Exception as e: r={'dataset':item['dataset'],'question_id':item['question_id'],'condition':'baseline','status':'invalid_exception','error':repr(e),'seed':42,'gold':item.get('gold',''),'correct':False}
                append(results,r); done.add((item['question_id'],'baseline')); count+=1
            if 'text_mas' in needed:
                try: r=self.run_text(item)
                except Exception as e: r={'dataset':item['dataset'],'question_id':item['question_id'],'condition':'text_mas','status':'invalid_exception','error':repr(e),'seed':42,'gold':item.get('gold',''),'correct':False}
                append(results,r); done.add((item['question_id'],'text_mas')); count+=1
            latent_needed=[c for c in needed if c in ('H0','H1','H2','H3','P','C','PR','PC')]
            if latent_needed:
                try:
                    past,ustages,utrans=self.build_upstream(item,10)
                    rng=(torch.random.get_rng_state(),torch.cuda.get_rng_state_all())
                    for c in latent_needed:
                        try: r=self.run_latent_from_snapshot(item,c,past,ustages,utrans,rng)
                        except Exception as e: r={'dataset':item['dataset'],'question_id':item['question_id'],'condition':c,'status':'invalid_exception','error':repr(e),'seed':42,'gold':item.get('gold',''),'correct':False}
                        append(results,r); done.add((item['question_id'],c)); count+=1
                    del past
                except Exception as e:
                    for c in latent_needed:
                        r={'dataset':item['dataset'],'question_id':item['question_id'],'condition':c,'status':'invalid_exception_upstream','error':repr(e),'seed':42,'gold':item.get('gold',''),'correct':False}
                        append(results,r); done.add((item['question_id'],c)); count+=1
            deep_needed=[c for c in needed if c in ('n6_fixed10','n6_total30')]
            for c in deep_needed:
                try: r=self.run_round3(item,c)
                except Exception as e: r={'dataset':item['dataset'],'question_id':item['question_id'],'condition':c,'status':'invalid_exception','error':repr(e),'seed':42,'gold':item.get('gold',''),'correct':False}
                append(results,r); done.add((item['question_id'],c)); count+=1
            save(self.out/'status.json',{'shard':shard,'shards':shards,'completed_local':count,'target_local':target,'global_done':len(done),'last_key':item['question_id']})
        save(self.out/f'shard_{shard}_complete.json',{'completed_local':count,'target_local':target,'conditions':conditions})

def main():
    verify_source()
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--shard',type=int,required=True); ap.add_argument('--shards',type=int,default=3); ap.add_argument('--round',type=int,required=True); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--max_items',type=int,default=-1); args=ap.parse_args()
    items=load_items()
    if args.max_items>0: items=items[:args.max_items];
    if args.round == 1: cond=['baseline','text_mas','H0','H1','H2','H3']
    elif args.round == 2: cond=['P','C','PR','PC']
    elif args.round==3: cond=['n6_fixed10','n6_total30']
    else: raise SystemExit('bad round')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    assert visible not in (None,'','NoDevFiles'), 'CUDA_VISIBLE_DEVICES must be set by the launcher to the selected physical GPU'
    assert torch.cuda.is_available() and torch.cuda.device_count()==1, 'exactly one launcher-selected CUDA device is required'
    runner=Runner(args.out,torch.device('cuda:0')); runner.run(items,cond,args.shard,args.shards)
if __name__=='__main__': main()




