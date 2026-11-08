"""Reuse validated full-run collector, adding private-MPS ownership gates."""
from pathlib import Path
def main():
 here=Path(__file__).resolve().parent
 source=(here.parent/'real_kv_batch_20261108/collect.py').read_text(encoding='utf-8')
 source=source.replace("experiments/real_kv_batch_20261108')","experiments/mps_validation_20261108')")
 needle="assert load(r/'result.json')['passed'] and not list(r.glob('failure_*.json')) and not (r/'controller_failure.json').exists()"
 assert needle in source
 extra="""
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
stop=load(r/'mps_stop.json');assert stop['returncode']==0 and not stop['owned_after']
assert identity(load(r/'launch.json')['pid']) is None
scene=load(r/'mps_scene.json');assert scene['server_list'].strip()
assert any(Path(x['argv'][0]).name=='nvidia-cuda-mps-server' and x['CUDA_VISIBLE_DEVICES']==load(r/'launch.json')['gpu_uuid'] for x in scene['owned_processes'])
parent=Path(load(r/'launch.json')['parent']);assert sha(r/'worker.py')==sha(parent/'worker.py')
save(r/'parent_reference.json',dict(archive=load(parent/'archive_verified.json'),worker_sha=sha(parent/'worker.py'),packets={f'packet{i}.pt':sha(parent/f'packet{i}.pt') for i in range(4)}))
"""
 source=source.replace(needle,needle+'\n'+extra)
 exec(compile(source,'validated_full_collector_with_MPS_gates','exec'),dict(__name__='__main__',__file__=str(here.parent/'real_kv_batch_20261108/collect.py')))
if __name__=='__main__':main()
