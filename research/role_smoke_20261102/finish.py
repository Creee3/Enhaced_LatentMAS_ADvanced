"""Archive a stopped smoke and verify all members, then summarize engineering checks."""
import datetime,gzip,hashlib,io,json,math,tarfile
from pathlib import Path,PurePosixPath
from remote import ROOT,call
from capture import PRIVATE,HERE

def main():
    active=json.loads((PRIVATE/'active.json').read_text(encoding='utf-8'))
    script='''import io,json,sys,tarfile,hashlib,fcntl
from pathlib import Path
run=Path(RUN_VALUE);out=Path(OUT_VALUE);sys.path.insert(0,str(run))
from cloud_common import identity
assert identity(PID_VALUE) is None, 'worker still live; no final archive'
assert (out/'exit.json').exists()
root=run.parents[1]
with (root/'budget.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX);budget=json.loads((root/'cumulative_budget.json').read_text())
 assert not any(x.get('metadata',{}).get('experiment')==run.name for x in budget['active_reservations'].values()), 'own RID still reserved'
 closing={k:budget[k] for k in ['actual_tokens','reserved_tokens','unknown_cost_upper_bound_tokens']}
 assert sum(closing.values())<=115000000
(run/'management').mkdir(exist_ok=True)
(run/'management/closing_budget_check.json').write_text(json.dumps(dict(own_reserved=0,ledger=closing),indent=2)+'\\n',encoding='utf-8')
base=run.parents[1]/'evidence/gpqa_repair_8096_20261028T161254Z/bundle'
files={}
for p in sorted(run.rglob('*')):
 if p.is_file() and '__pycache__' not in p.parts:
  assert not p.is_symlink();files['run/'+p.relative_to(run).as_posix()]=p
lock=json.loads((base/'source_lock.json').read_text())
for n in list(lock['files'])+['source_lock.json']:
 p=base/n;assert p.is_file() and not p.is_symlink();files['referenced_bundle37/'+n]=p
manifest={n:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for n,p in files.items()}
raw=(json.dumps(manifest,ensure_ascii=False,indent=2)+'\\n').encode()
class HashWriter:
 def __init__(self):self.digest=hashlib.sha256();self.size=0
 def write(self,b):self.digest.update(b);self.size+=len(b);return sys.stdout.buffer.write(b)
 def flush(self):sys.stdout.buffer.flush()
stream=HashWriter()
with tarfile.open(fileobj=stream,mode='w|gz') as tar:
 for n,p in files.items():tar.add(p,arcname=n,recursive=False)
 info=tarfile.TarInfo('archive_manifest.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
stream.flush()
(run.parent/(run.name+'_archive_stream.json')).write_text(json.dumps(dict(sha256=stream.digest.hexdigest(),bytes=stream.size))+'\\n')
'''.replace('RUN_VALUE',repr(active['run'])).replace('OUT_VALUE',repr(active['attempt'])).replace('PID_VALUE',str(active['identity']['pid']))
    raw=call(script,180)
    digest=hashlib.sha256(raw).hexdigest()
    remote_proof=json.loads(call('from pathlib import Path\nprint(Path('+repr(str(Path(active['run']).parent/(Path(active['run']).name+'_archive_stream.json')).replace('\\','/'))+').read_text())'))
    assert digest==remote_proof['sha256'] and len(raw)==remote_proof['bytes']
    archive=PRIVATE/(Path(active['run']).name+'_full_'+digest[:12]+'.tar.gz')
    archive.write_bytes(raw)
    # Reading to EOF explicitly checks gzip CRC, independently of tar's EOF.
    plain=gzip.decompress(raw)
    extract=PRIVATE/('verified_extracted_'+digest[:12]);extract.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(plain),mode='r:') as tar:
        members=tar.getmembers();names=[m.name for m in members]
        assert len(names)==len(set(names))
        manifest=json.load(tar.extractfile('archive_manifest.json'))
        assert set(names)==set(manifest)|{'archive_manifest.json'}
        for m in members:
            name=PurePosixPath(m.name)
            assert m.isfile() and not name.is_absolute() and '..' not in name.parts
            b=tar.extractfile(m).read();target=extract.joinpath(*name.parts)
            assert target.resolve().is_relative_to(extract.resolve())
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
            if m.name in manifest:
                entry=manifest[m.name]
                assert len(b)==entry['bytes'] and hashlib.sha256(b).hexdigest()==entry['sha256']
                assert hashlib.sha256(target.read_bytes()).hexdigest()==entry['sha256']
    proof=dict(archive_sha256=hashlib.sha256(raw).hexdigest(),archive_bytes=len(raw),members=len(names),
        stream_sha=True,gzip_crc=True,tar_exact_set=True,all_member_sha=True,safe_full_extract_sha=True)
    (PRIVATE/'archive_verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8')
    out=extract/'run'/'attempts'/Path(active['attempt']).name
    exit=json.loads((out/'exit.json').read_text());plan=json.loads((extract/'run/plan.json').read_text())
    rows=[json.loads(s) for s in (out/'results.jsonl').read_text(encoding='utf-8').split('\n') if s] if (out/'results.jsonl').exists() else []
    expected={(x['question_id'],r,m) for x in plan['questions'] for r in [1,2,3] for m in plan['arms']}
    keys={(x['question_id'],x['repeat'],x['method']) for x in rows}
    trace=[json.loads(s) for s in (out/'trace.jsonl').read_text(encoding='utf-8').split('\n') if s] if (out/'trace.jsonl').exists() else []
    starts={x['reservation_id']:x for x in trace if x['event']=='request_started'}
    settles=[x for x in trace if x['event']=='request_settled']
    assert len(settles)==len({x['reservation_id'] for x in settles})
    assert set(starts)=={x['reservation_id'] for x in settles}, 'unsettled own RID'
    rawpath=out/'raw/requests_raw.jsonl'
    rawrows=[json.loads(s) for s in rawpath.read_text(encoding='utf-8').split('\n') if s] if rawpath.exists() else []
    inputs=[x for x in rawrows if x['event']=='request_input']
    outputs=[x for x in rawrows if x['event']=='request_output']
    assert len(inputs)==len(outputs)==len(starts)
    assert {x['reservation_id'] for x in inputs}=={x['reservation_id'] for x in outputs}==set(starts)
    assert all(next(y for y in outputs if y['reservation_id']==x['reservation_id'])['actual_positions']==x['actual_positions'] for x in settles)
    repeats=[];timings=[]
    for q in plan['questions']:
        for method in plan['arms']:
            group=[r for r in rows if r['question_id']==q['question_id'] and r['method']==method]
            repeats.append(dict(question_id=q['question_id'],method=method,completed=len(group),
                identical_position_selection=len({r['selected_position_sha'] for r in group})==1 if group else None,
                identical_generated_tokens=len({r['token_sha'] for r in group})==1 if group else None))
            if group:
                ts=[r['timing'] for r in group]
                fields=['required_shared_judger_probe_seconds','selector_and_gather_seconds','judger_seconds','reconstructed_standalone_seconds']
                assert all(math.isfinite(t[f]) and t[f]>=0 for t in ts for f in fields)
                timings.append(dict(question_id=q['question_id'],method=method,**{f:dict(mean=sum(t[f] for t in ts)/len(ts),min=min(t[f] for t in ts),max=max(t[f] for t in ts)) for f in fields},
                    generated_tokens=[r['generated_tokens'] for r in group]))
    complete=bool(exit['completed'] and len(rows)==len(keys)==18 and keys==expected)
    ready=json.loads((out/'model_ready.json').read_text()) if (out/'model_ready.json').exists() else None
    if complete:
        assert len(starts)==54
        assert ready['bf16_cuda'] and ready['identity_norm'] and ready['parameter_bytes']==16381470720
        model_lock=json.loads((extract/'referenced_bundle37/runtime/model_lock.json').read_text())
        assert {x['file'] for x in trace if x['event']=='model_file_verified'}==set(model_lock['files'])
        for row in rows:
            assert row['positions_shape']==[36,8,130] and row['retained_kv_bytes']==19169280
            assert row['source_k_equal_measured'] and row['source_v_equal_measured'] and row['protected_latent_sink']
            assert row['timing']['cuda_sync_both_boundaries'] and row['parser_sha']=='92163f3a5389c4a7e25868ff931fc919703eb1e500ef74dc03b4810ab6f0225d'
    report=dict(complete=complete,answers=len(rows),unique=len(keys),missing=len(expected-keys),archive=proof,
        model_summary={k:ready[k] for k in ['device','verified_weight_files','bf16_cuda','parameter_bytes','identity_norm','target_norm','model_load_hash_prepare_seconds']} if ready else None,
        nonempty_frozen_parser_outputs=sum(bool(r.get('prediction')) for r in rows),
        output_cap_reached=sum(r['generated_tokens']>=8192 for r in rows),
        requests=len(starts),actual_positions=sum(x['actual_positions'] for x in settles),
        new_unknown=sum(x['independent_unknown'] for x in settles),repeats=repeats,timings=timings,
        no_effectiveness_claim=True,reconstructed_times_not_independent_e2e=True,
        timing_scope_notes=['selector_and_gather_seconds includes objective diagnostics and scalar readbacks; it is an outer compressor time, not pure kernel/core time',
            'native model metering and position proof hooks are present',
            'row peak_allocated_bytes is cumulative since start of its repeat, not independent per-arm peak',
            'shared probe is charged once; reconstructed per-arm time includes it once',
            'KV bytes use measured 36x8x130 positions and locked BF16 128-dimension paired K/V geometry, rather than a separately serialized packet size'],exit=exit)
    (PRIVATE/'engineering_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'engineering_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 两题三次工程检查报告','',
        '状态：'+('18个结果键齐全，工程检查完成。' if complete else '尚有缺失或失败，不能报告全部通过。'),'',
        f"- 完成{len(rows)}/18条；真实请求{len(starts)}个，逐RID原始输入/输出与结算记录一致。",
        f"- 实际新增位置费用{report['actual_positions']}；新增unknown={report['new_unknown']}。历史费用不重新结算。",
        f"- 冻结解析器非空输出{report['nonempty_frozen_parser_outputs']}/{len(rows)}；触及8192输出上限{report['output_cap_reached']}条。",
        '- 实测36层×8个KV头，每头130行；按锁定BF16成对K/V维度计算为19,169,280字节，不是网络传输测量。K/V与原位置的gather校验，以及全部latent/sink保护，逐臂执行。',
        '- seed42固定；每题三次重复，三臂轮换顺序。两题来自同一main集合，一题属于Diamond；不把它们当作独立的两套数据集。','']
    if ready:lines += [f"设备：{ready['device']}；15个文件SHA、全BF16 CUDA及identity norm通过。模型加载/哈希/初始化用时{ready['model_load_hash_prepare_seconds']:.2f}秒，独立于下表。",'']
    lines += ['| 题ID | 方法 | 次数 | 位置三次一致 | token序列三次一致 | 选择与gather均值(s) | Judger均值(s) | 生成tokens |',
              '|---|---|---:|---|---|---:|---:|---|']
    for r in repeats:
        t=next((x for x in timings if x['question_id']==r['question_id'] and x['method']==r['method']),None)
        if t:
            lines.append(f"| {r['question_id']} | {r['method']} | {r['completed']} | {r['identical_position_selection']} | {r['identical_generated_tokens']} | {t['selector_and_gather_seconds']['mean']:.3f} | {t['judger_seconds']['mean']:.3f} | {', '.join(map(str,t['generated_tokens']))} |")
    lines += ['', '计时范围：CUDA两端同步；选择与gather是包含目标诊断及标量回读的外层压缩时间，模型审计hook仍在。必须的probe计入成本，并由三臂真实共享一次。每臂重建总时间不是独立端到端实测；不同答案长度不作为方法速度胜负。显存峰值字段按每次重复累计。',
        '', '这轮不作准确率优劣、显著性、创新有效性或加速比结论；未运行Full/OBF/JIVE/旧receiver新答案，未启动全量新实验。',
        '',f"完整归档SHA256：`{proof['archive_sha256']}`；{proof['members']}个成员。云端流SHA、gzip CRC、tar精确集合、逐成员及安全全解包SHA全部通过。私有题文/原答/tokens不进入仓库。"]
    (HERE/'engineering_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(complete=complete,answers=len(rows),missing=report['missing'],requests=len(starts),actual_positions=report['actual_positions'],new_unknown=report['new_unknown'],archive=proof,repeats=repeats),ensure_ascii=False))

if __name__=='__main__':main()
