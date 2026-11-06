"""Offline migration checks; no model construction, generation or settlement."""
import hashlib,importlib.metadata,json,os,subprocess,sys
from pathlib import Path

M=Path('/home/ubuntu/cloud_migration_20261106')
ROOT=Path('/home/ubuntu/ICML2027')
EXPERIMENT=ROOT/'experiments/latentmas_cumulative_history_20261025'
ROLE=EXPERIMENT/'evidence/role_blocks_gpqa_main448_s42_l10_20261102T142146Z'
OLD=EXPERIMENT/'evidence/receiver_gpqa_main448_s42_l10_l40_20261101T133212Z'
G=EXPERIMENT/'evidence/attention_G_gpqa_matched225_s42_l10_v2_20261103T111226Z'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def rows(p):return [json.loads(b) for b in p.read_bytes().split(b'\n') if b]
def key(r):return r['dataset'],r['question_id'],r['latent_steps'],r['method']
def answer_rows(run):
    r=rows(run/'reused_results.jsonl')
    for p in (run/'attempts').glob('*/results.jsonl'):r+=rows(p)
    return r
def main():
    assert load(M/'verification/RESTORE_VERIFIED.json')['files']==92648
    assert sha(ROLE/'run_lock.json')=='70828faf23244d950cc1a562af9253773a2848ef962f718ba6c40e5e92468de9'
    for run in [ROLE,G]:
        lock=load(run/'run_lock.json')
        for entry in ['files','snapshots']:
            for n,e in lock[entry].items():assert sha(run/n)==e['sha256']
        assert (run/'STOP').exists(),'Historical pause marker missing'
    new=answer_rows(ROLE);baseline=rows(ROLE/'baseline_results.jsonl');old=answer_rows(OLD)
    assert len(new)==len(set(map(key,new)))==675
    assert len(baseline)==len(set(map(key,baseline)))==1792
    assert len(old)==len(set(map(key,old)))==3584
    cohort=load(OLD/'cohort_questions.json')
    expected={(q['dataset'],q['question_id'],s,m) for q in cohort for s in [10,40]
              for m in ['full','headobf','jive','receiver_attention_sink_recent']}
    assert set(map(key,old))==expected
    common=set.intersection(*[{(r['dataset'],r['question_id']) for r in new if r['method']==method} for method in ['T','R','R-shuffle']])
    assert len(common)==225
    assert not list((G/'attempts').glob('*/results.jsonl'))
    budget=load(EXPERIMENT/'cumulative_budget.json')
    wanted=dict(actual_tokens=74904762,reserved_tokens=0,unknown_cost_upper_bound_tokens=58106)
    assert {k:budget[k] for k in wanted}==wanted and not budget['active_reservations']
    expected_ledger=load(M/'metadata/current_recovery_manifest.json')['files']['ICML2027/experiments/latentmas_cumulative_history_20261025/cumulative_budget.json']['sha256']
    assert sha(EXPERIMENT/'cumulative_budget.json')==expected_ledger
    # Process ownership comes from real UID/argv, never old stored PID numbers.
    own=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            argv=(p/'cmdline').read_bytes().split(bytes([0]))
            names=[Path(x.decode(errors='replace')).name for x in argv[1:] if x]
            if any(x in {'manager.py','manager_recovery.py','cloud_worker.py','managed_worker.py'} for x in names):own.append(int(p.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    assert not own
    evidence={'passed':True,'restored_files':92648,'old_exact_answers':3584,'frozen_baseline':1792,
        'role_answers':675,'common_questions':225,'G_answers':0,'source_locks_intact':True,
        'budget':wanted,'ledger_sha_unchanged':True,'active_reservations':0,'own_experiment_processes':own,
        'model_forward_calls':0,'experiments_resumed':False}
    (M/'verification/EVIDENCE_VERIFIED.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if '--evidence-only' in sys.argv:
        print(json.dumps(evidence),flush=True);return
    versions={}
    for line in (M/'requirements-core.txt').read_text().splitlines():
        if not line or line.startswith('#'):continue
        name,want=line.split('==');actual=importlib.metadata.version(name)
        assert actual==want,(name,want,actual);versions[name]=actual
    os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='1'
    import torch,transformers,accelerate,numpy,scipy,datasets
    assert torch.version.cuda=='12.4'
    assert torch.cuda.is_available() and torch.cuda.device_count()==4
    assert load(M/'verification/WEIGHT_HEADERS_VERIFIED.json')['tensor_bytes']==16381470720
    from transformers import AutoConfig,AutoTokenizer
    model=Path('/home/ubuntu/models/Qwen3-8B')
    for n,e in load(M/'metadata/frozen_model_lock.json')['files'].items():assert sha(model/n)==e['sha256']
    config=AutoConfig.from_pretrained(str(model),local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
    assert config.model_type=='qwen3' and config.num_hidden_layers==36 and config.num_key_value_heads==8
    assert tokenizer.encode('migration verification')
    result={**evidence,
        'core_versions':versions,'python':sys.version.split()[0],'model_frozen_files':15,
        'offline_tokenizer_config_passed':True,'cuda_compiled':torch.version.cuda,
        'cuda_discovery_available':True,'visible_gpu_count':4,'bf16_tensor_header_bytes':16381470720,
        'model_loaded':False,'model_forward_calls':0,'experiments_resumed':False}
    (M/'verification/MIGRATION_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
