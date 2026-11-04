"""Offline scope/critical-evidence audit, no raw answers on stdout."""
import hashlib,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
PRIVATE=REPO.parent/'advanced_MOC/experiments/cloud_rescue_20261104'


def long(p):
    p=p.resolve();return Path('\\\\?\\'+str(p)) if os.name=='nt' else p


def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_bytes().split(b'\n') if x]
def key(r):return r['dataset'],r['question_id'],r['latent_steps'],r['method']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main(critical_only=False):
    groups={};names=set()
    for group in (['current','source'] if critical_only else ['current','history','source']):
        result=load(PRIVATE/(group+'_verified.json'));assert result['passed'] and not result['source_changed_files']
        dest=long(PRIVATE/'verified'/group)
        manifest=load(PRIVATE/'history_recovery_manifest.json') if result.get('extraction_format')=='sqlite_chunk_store' else load(dest/'recovery_manifest.json')
        assert not names&set(manifest['files']);names.update(manifest['files'])
        groups[group]={k:result[k] for k in ['archive_sha256','compressed_bytes','source_files','source_bytes','windows_encoded_paths']}
    root=long(PRIVATE/'verified/current/ICML2027/experiments/latentmas_cumulative_history_20261025')
    role=root/'evidence/role_blocks_gpqa_main448_s42_l10_20261102T142146Z'
    lock=load(role/'run_lock.json')
    assert sha(role/'run_lock.json')=='70828faf23244d950cc1a562af9253773a2848ef962f718ba6c40e5e92468de9'
    for n,e in lock['files'].items():assert sha(role/n)==e['sha256']
    for n,e in lock['snapshots'].items():assert sha(role/n)==e['sha256']
    baseline=rows(role/'baseline_results.jsonl');new=rows(role/'reused_results.jsonl')
    for p in (role/'attempts').glob('*/results.jsonl'):new+=rows(p)
    assert len(baseline)==len(set(map(key,baseline)))==1792
    assert len(new)==len(set(map(key,new)))==675
    assert {r['method'] for r in new}=={'T','R','R-shuffle'}
    assert all(sum(r['method']==m for r in new)==225 for m in ['T','R','R-shuffle'])
    assert len({(r['dataset'],r['question_id']) for r in new})==225
    older=root/'evidence/receiver_gpqa_main448_s42_l10_l40_20261101T133212Z'
    older_rows=rows(older/'reused_results.jsonl')
    for p in (older/'attempts').glob('*/results.jsonl'):older_rows+=rows(p)
    older_cohort=load(older/'cohort_questions.json')
    expected={(q['dataset'],q['question_id'],s,m) for q in older_cohort for s in [10,40]
              for m in ['full','headobf','jive','receiver_attention_sink_recent']}
    assert len(older_cohort)==448 and len(older_rows)==len(set(map(key,older_rows)))==3584
    assert set(map(key,older_rows))==expected
    cpu_active=load(REPO.parent/'advanced_MOC/experiments/latentmas_cpu_single_question_20261104/active.json')
    cpu=root/'evidence'/Path(cpu_active['run']).name
    cpu_result=load(cpu/'result.json');assert cpu_result['completed'] and cpu_result['cpu_only'] and cpu_result['generated_tokens']==6
    g_active=load(REPO.parent/'advanced_MOC/experiments/latentmas_attention_control225_20261103/cloud_active.json')
    g=root/'evidence'/Path(g_active['run']).name
    assert not list((g/'attempts').glob('*/results.jsonl'))
    for n,e in load(g/'run_lock.json')['snapshots'].items():assert sha(g/n)==e['sha256']
    budget=load(root/'cumulative_budget.json')
    assert sum(budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens'])<=115000000
    assert budget['reserved_tokens']==0 and not budget['active_reservations']
    env=load(PRIVATE/'environment_recovery.json');assert len(env['environments'])==4
    models=PRIVATE/'public_model_metadata'
    assert (models/'Qwen3-8B/tokenizer_config.json').exists()
    assert (models/'Qwen3-8B/generation_config.json').exists()
    result=dict(passed=True,groups=groups,total_files=len(names),total_source_bytes=sum(g['source_bytes'] for g in groups.values()),
        total_compressed_bytes=sum(g['compressed_bytes'] for g in groups.values()),all_groups_disjoint=True,
        critical=dict(role_source_lock_valid=True,completed_older_main448_latent10_40=3584,frozen_old_baseline=1792,new_three_arm_answers=675,common_questions=225,
            duplicate_keys=0,G_answers=0,G_reference_snapshots_valid=True,cpu_single_question_result_present=True,
            budget={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']},
            active_reservations=0,environment_count=len(env['environments']),public_model_metadata_present=True),
        caveat='Known credential files, rebuildable environments/caches and external pretrained model weights excluded; symlinks recorded, not followed.')
    metadata=load(PRIVATE/'metadata_manifest.json')
    for n,e in metadata.items():assert sha(PRIVATE/n)==e['sha256'] and (PRIVATE/n).stat().st_size==e['bytes']
    result.update(all_three_groups_verified=not critical_only,metadata_files_sha_verified=len(metadata))
    (PRIVATE/('CRITICAL_VERIFIED.json' if critical_only else 'BACKUP_VERIFIED.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main('--critical-only' in sys.argv)
