"""Probe physical GPU indices independently; driver failure on one card is local."""
import subprocess


def collect(gpus, runner=subprocess.run):
    rows=[];errors={}
    for gpu in gpus:
        try:
            result=runner(['nvidia-smi','-i',str(gpu),
                '--query-gpu=index,uuid,utilization.gpu,memory.free,memory.total',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
            if result.returncode:raise ValueError((result.stderr or result.stdout).strip()[:240])
            lines=result.stdout.strip().splitlines()
            if len(lines)!=1:raise ValueError('Expected one GPU record')
            a,b,c,d,e=lines[0].split(',')
            row=dict(gpu=int(a),uuid=b.strip(),utilization=float(c),free_mib=float(d),total_mib=float(e))
            if row['gpu']!=gpu or not row['uuid'].startswith('GPU-') or not 0<=row['free_mib']<=row['total_mib'] or row['total_mib']<=0 or not 0<=row['utilization']<=100:
                raise ValueError('Invalid or mismatched GPU telemetry')
            rows.append(row)
        except (subprocess.TimeoutExpired,OSError,ValueError) as exc:errors[gpu]=str(exc)[:240]
    return rows,errors
