"""Bounded, restartable offline codec check on 40 existing Full captures."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
CAPTURE=Path('/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025/evidence/prompt_full_sampling120_s42_l10_20261112T053404Z')


def read(path):return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('w') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def prepare():
    assert not (HERE/'run_lock.json').exists()
    collection=read(CAPTURE/'COLLECTION_COMPLETE.json')
    assert collection['passed'] and collection['audit_sha']==sha(CAPTURE/'collection_audit.json')
    assert collection['source_lock']==sha(CAPTURE/'run_lock.json')
    audit=read(CAPTURE/'collection_audit.json');cases=[]
    for row in audit['manifest']:
        result=read(row['result_file'])
        if result['dataset_group'] not in ('AIME24','GPQA-Diamond'):continue
        assert sha(row['result_file'])==row['result_sha']
        files=[f for f in result['files'] if f['label'] in ('serial_planner','serial_critic','serial_refiner')]
        assert [f['label'] for f in files]==['serial_planner','serial_critic','serial_refiner']
        cases.append(dict(id=f'{len(cases):03d}',index=result['index'],dataset_group=result['dataset_group'],
            question_sha256=result['question_sha256'],files=files,reference=result['reference'],
            source_result=row['result_file'],source_result_sha256=row['result_sha']))
    assert len(cases)==40 and all(sum(c['dataset_group']==g for c in cases)==20 for g in ('AIME24','GPQA-Diamond'))
    protocol=dict(version=1,cases=40,groups={'AIME24':20,'GPQA-Diamond':20},
        dataset_policy='Only these two existing groups; no MMLU-Pro work',
        source='Historical Full serial L10 native-template BF16 captures, not new aligned228 protocol',
        selection='All existing 20 questions per selected group, no answer-based selection',
        model_calls=0,new_answers=0,new_model_workers=0,math_workers=1,
        layers=36,kv_heads=8,feature_dimension=128,channels=['K','V'],
        packet_bytes_per_layer_head_channel=33280,bytes_per_handoff=19169280,
        exact_rows='global sink4, newest prompt tail8, all 30 historical latent rows',
        prompt_rows='All other prompt rows encoded and restored in original order; none sampled',
        arms=['Global','Source-Shared','Source-Separate','Shuffle-42','Shuffle-43','Shuffle-44'],
        primary='Question-level sqrt(sum optional-row reconstruction SSE / sum global-centered prompt energy), joint K/V',
        secondary='Separate K and V; raw-energy relative RMS; actual packet bytes and padding; encoding/decoding wall time',
        structure='Explicit group means; common basis of group-centered data; orthogonal private residual directions',
        rank_rule='Global maximal rank fitting bytes; Source-Shared shared:private rank=2:1; Source-Separate shared rank0; all ranks determined only by sizes and budget',
        accounting='BF16 means, bases, coefficients and protected rows; uint8 labels for grouped methods; 64B header; counted zero padding to equal packet length',
        key_handling='Encode stored post-RoPE K directly, restore at same absolute index; no unrotation or rebasing',
        inference_memory='Receiver reconstructs full-length dense cache; no GPU-resident cache reduction or speedup claimed',
        statistics='Question is sampling unit; 2000 paired bootstrap replicates, seed20261117; exploratory development data',
        decision='Only suggest a new-protocol receiver pilot if Source-Shared has lower primary error than Global and mean Shuffle in each dataset, with paired bootstrap CI upper bound below0; Source-Separate remains an ablation',
        limits='Geometry feasibility only. Source grouping also reflects content, order, position and template. Not accuracy evidence.',
        no_parameter_sweep=True,no_automatic_answer_generation=True)
    save(HERE/'manifest.json',cases);save(HERE/'protocol.json',protocol)
    files=['run.py','codec.py','checks.py','README.md']
    save(HERE/'run_lock.json',dict(created=time.time(),source={f:sha(HERE/f) for f in files},
        manifest_sha256=sha(HERE/'manifest.json'),protocol_sha256=sha(HERE/'protocol.json'),
        capture_lock_sha256=sha(CAPTURE/'run_lock.json'),collection_audit_sha256=sha(CAPTURE/'collection_audit.json')))
    print(json.dumps(dict(prepared=True,cases=40,model_calls=0,arms=6)))


def verify():
    lock=read(HERE/'run_lock.json')
    assert all(sha(HERE/k)==v for k,v in lock['source'].items())
    assert sha(HERE/'manifest.json')==lock['manifest_sha256']
    assert sha(HERE/'protocol.json')==lock['protocol_sha256']
    assert sha(CAPTURE/'run_lock.json')==lock['capture_lock_sha256']
    assert sha(CAPTURE/'collection_audit.json')==lock['collection_audit_sha256']
    return lock


def question(entry,device,update):
    import numpy as np
    import torch
    from codec import ARMS,layout,encode,decode,errors,PACKET_BYTES
    start=time.monotonic()
    assert sha(entry['source_result'])==entry['source_result_sha256']
    for f in entry['files']:assert sha(f['path'])==f['sha256']
    ref=entry['reference'];assert sha(ref['data_file'])==ref['data_sha']
    arrays=[np.load(f['path'],mmap_mode='r',allow_pickle=False) for f in entry['files']]
    lengths=[a.shape[-2] for a in arrays];info=layout(lengths)
    assert all(a.shape[:3]==(36,2,8) and a.shape[-1]==128 and a.dtype==np.uint16 for a in arrays)
    with np.load(ref['data_file'],allow_pickle=False) as z:latents=z['serial_latent_bits'].copy()
    assert latents.shape==(3,36,2,8,10,128)
    measurements={a:{c:dict(sse=0.,centered_energy=0.,total_energy=0.,encode_seconds=0.,decode_seconds=0.,packets=0,bytes=0,padding_bytes=0) for c in ('K','V')} for a in ARMS}
    seed=int(entry['question_sha256'][:12],16)
    spec_saved={};packet_checks=0;io_seconds=time.monotonic()-start
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    for layer in range(36):
        update(phase='codec',question=entry['id'],layer=layer)
        for ch,channel in enumerate(('K','V')):
            blocks=[torch.from_numpy(np.array(a[layer,ch],copy=True)).view(torch.bfloat16).to(device) for a in arrays]
            full=torch.cat(blocks,dim=1)
            lat=torch.from_numpy(latents[:,layer,ch].copy()).view(torch.bfloat16).permute(1,0,2,3).reshape(8,30,128).to(device)
            exact=torch.cat([full[:,info['exact_prompt']],lat],dim=1)
            x=full[:,info['optional']]
            original=x.clone();original_exact=exact.clone()
            # Fixed rotation balances cache warmup order without looking at errors.
            shift=(entry['index']+layer+ch)%len(ARMS);order=ARMS[shift:]+ARMS[:shift]
            for arm in order:
                sync();begin=time.monotonic()
                packets,spec=encode(x,exact,lengths,arm,seed=seed)
                sync();encoded=time.monotonic()-begin
                begin=time.monotonic();decoded=[decode(p,device=device) for p in packets]
                restored=torch.stack([r[0] for r in decoded]);protected=torch.stack([r[1] for r in decoded])
                sync();decode_seconds=time.monotonic()-begin
                assert torch.equal(protected,exact) and all(len(p)==PACKET_BYTES for p in packets)
                assert torch.equal(original,x) and torch.equal(original_exact,exact)
                m=measurements[arm][channel]
                for k,v in errors(x,restored).items():m[k]+=v
                m['encode_seconds']+=encoded;m['decode_seconds']+=decode_seconds
                m['packets']+=8;m['bytes']+=sum(map(len,packets));m['padding_bytes']+=8*spec['padding_bytes']
                spec_saved[arm]=spec;packet_checks+=8
                if entry['id']=='000' and layer==0:
                    out=HERE/'engineering_packets';out.mkdir(exist_ok=True)
                    p=out/(arm+'_'+channel+'.bin')
                    if p.exists():assert p.read_bytes()==packets[0]
                    else:p.write_bytes(packets[0])
            del full,blocks,x,exact,lat,restored,protected
    for arm in ARMS:
        for channel in ('K','V'):
            m=measurements[arm][channel]
            m['centered_relative_RMS']=(m['sse']/max(m['centered_energy'],1e-30))**.5
            m['raw_relative_RMS']=(m['sse']/max(m['total_energy'],1e-30))**.5
        s=measurements[arm]
        s['joint_centered_relative_RMS']=(sum(s[c]['sse'] for c in ('K','V'))/max(sum(s[c]['centered_energy'] for c in ('K','V')),1e-30))**.5
        assert sum(s[c]['bytes'] for c in ('K','V'))==19169280
    return dict(id=entry['id'],index=entry['index'],dataset_group=entry['dataset_group'],
        question_sha256=entry['question_sha256'],prompt_lengths=lengths,optional_rows=info['n'],
        measurements=measurements,specifications=spec_saved,model_calls=0,answers_generated=0,
        layers=36,heads=8,packet_checks=packet_checks,exact_rows_verified=True,
        seconds=time.monotonic()-start,input_verification_and_io_seconds=io_seconds,
        source_result_sha256=entry['source_result_sha256'],source_latent_sha256=ref['data_sha'],
        run_lock_sha256=sha(HERE/'run_lock.json'))


def worker(gpu,limit=None):
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    guard=(HERE/'worker.lock').open('a')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),gpu=gpu,started=time.time(),kind='offline_matrix_worker_no_model',
        process_start_ticks=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()[19])
    def update(**kw):state.update(kw,updated=time.time());save(HERE/'worker.json',state)
    try:
        verify();assert read(HERE/'checks.json')['passed']
        import torch
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
        assert torch.cuda.is_available();device=torch.device('cuda:0')
        torch.cuda.reset_peak_memory_stats(device)
        update(status='running',torch=torch.__version__,device_name=torch.cuda.get_device_name(device))
        done=0
        for entry in read(HERE/'manifest.json'):
            if (HERE/'STOP').exists():break
            target=HERE/'results'/(entry['id']+'.json')
            if target.exists():continue
            result=question(entry,device,update)
            result['peak_cuda_bytes']=torch.cuda.max_memory_allocated(device);save(target,result)
            done+=1;update(completed_by_worker=done,phase='committed',last_question=entry['id'])
            print(json.dumps(dict(id=entry['id'],group=entry['dataset_group'],seconds=result['seconds'],
                rms={a:v['joint_centered_relative_RMS'] for a,v in result['measurements'].items()})),flush=True)
            if limit and done>=limit:break
        complete=len(list((HERE/'results').glob('*.json')))==40
        update(status='exited',completed=complete,peak_cuda_bytes=torch.cuda.max_memory_allocated(device))
        if complete:summary()
    except BaseException:
        update(status='failed',error=traceback.format_exc());raise


def summary():
    import numpy as np
    rows=[read(p) for p in sorted((HERE/'results').glob('*.json'))]
    assert len(rows)==40
    source=sha(HERE/'run_lock.json');assert all(r['run_lock_sha256']==source for r in rows)
    result=dict(complete=True,cases=40,model_calls=0,answers_generated=0,source_lock_sha256=source,groups={})
    for group in ('AIME24','GPQA-Diamond'):
        part=[r for r in rows if r['dataset_group']==group];assert len(part)==20
        arms=list(part[0]['measurements']);item={'n':20,'arms':{},'paired':{}}
        for a in arms:
            item['arms'][a]=dict(joint_centered_RMS_mean=float(np.mean([r['measurements'][a]['joint_centered_relative_RMS'] for r in part])),
                K_centered_RMS_mean=float(np.mean([r['measurements'][a]['K']['centered_relative_RMS'] for r in part])),
                V_centered_RMS_mean=float(np.mean([r['measurements'][a]['V']['centered_relative_RMS'] for r in part])),
                encode_seconds_mean=float(np.mean([sum(r['measurements'][a][c]['encode_seconds'] for c in ('K','V')) for r in part])),
                decode_seconds_mean=float(np.mean([sum(r['measurements'][a][c]['decode_seconds'] for c in ('K','V')) for r in part])),
                bytes_per_handoff=19169280,
                padding_bytes_mean=float(np.mean([sum(r['measurements'][a][c]['padding_bytes'] for c in ('K','V')) for r in part])))
        candidate=np.array([r['measurements']['Source-Shared']['joint_centered_relative_RMS'] for r in part])
        for label in ('Global','Shuffle-mean','Source-Separate'):
            refs=[label] if label!='Shuffle-mean' else ['Shuffle-42','Shuffle-43','Shuffle-44']
            baseline=np.array([np.mean([r['measurements'][a]['joint_centered_relative_RMS'] for a in refs]) for r in part])
            delta=candidate-baseline;rng=np.random.default_rng(20261117)
            boot=delta[rng.integers(0,20,size=(2000,20))].mean(-1)
            item['paired'][label]=dict(mean_delta=float(delta.mean()),lower_error_cases=int((delta<0).sum()),
                relative_mean_change=float(candidate.mean()/baseline.mean()-1),
                bootstrap95=list(map(float,np.quantile(boot,[.025,.975]))))
        result['groups'][group]=item
    result['worth_new_protocol_receiver_pilot']=all(result['groups'][g]['paired'][a]['bootstrap95'][1]<0
        for g in result['groups'] for a in ('Global','Shuffle-mean'))
    result['decision_scope']='Exploratory byte-matched geometry feasibility; no automatic launch; not downstream accuracy'
    save(HERE/'summary.json',result)
    save(HERE/'COMPLETE.json',dict(passed=True,cases=40,summary_sha256=sha(HERE/'summary.json'),
        result_hashes={p.name:sha(p) for p in sorted((HERE/'results').glob('*.json'))},
        run_lock_sha256=source,completed=time.time()))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker','summary']);p.add_argument('--gpu',type=int,default=0);p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='summary':summary()
    else:worker(args.gpu,args.limit)
