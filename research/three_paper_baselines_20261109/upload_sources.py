"""Copy verified reference source snapshots to the existing SSH container; no imports/install/inference."""
from pathlib import Path
import hashlib, json, subprocess, sys, tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REFERENCES = ROOT / '04_参考代码与历史实验/上游参考代码'
PRIVATE = ROOT / 'advanced_MOC/experiments/reference_sources_20261109'
PRIVATE.mkdir(parents=True, exist_ok=True)

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def remote(code):
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', 'DevContainer',
                        '/home/ubuntu/.conda/envs/kv-graph/bin/python -'],
                       input=code.encode('utf-8'), capture_output=True, timeout=90)
    if r.returncode: raise RuntimeError(r.stderr.decode('utf-8', errors='replace')[-2000:])
    return json.loads(r.stdout)

receipts = []
for name in sys.argv[1:]:
    assert name in ['rclc-official', 'rclc-paper-v1']
    source = (REFERENCES / name).resolve()
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    files = []
    for p in sorted(source.rglob('*')):
        rel = p.relative_to(source)
        if '.git' in rel.parts or p.is_symlink() or not p.is_file(): continue
        if p.suffix.lower() in ['.mp4', '.webm', '.mov', '.gif', '.zip', '.gz']: continue
        files.append(dict(path=rel.as_posix(), bytes=p.stat().st_size, sha256=digest(p)))
    assert any(x['path'] == 'README.md' for x in files)
    assert any(x['path'].endswith('.py') for x in files)
    manifest = dict(requested_url='https://github.com/maxr0ssi/rclc',
                    canonical_url='https://github.com/agentcacheback/cacheback', commit=commit,
                    snapshot=name, files=files, scope='Source snapshot; Git database, datasets archives and large media excluded')
    bundle = PRIVATE / (name + '-' + commit[:12] + '.tar.gz')
    with tarfile.open(bundle, 'w:gz') as t:
        for x in files: t.add(source / x['path'], arcname=x['path'], recursive=False)
    sha = digest(bundle)
    target = '/home/ubuntu/ICML2027/reference_repos/' + name + '_' + commit[:12]
    remote_archive = '/home/ubuntu/ICML2027/reference_repos/' + bundle.name
    remote("import os,json\nfrom pathlib import Path\nassert os.getuid()==1000 and str(Path.home())=='/home/ubuntu'\nPath('/home/ubuntu/ICML2027/reference_repos').mkdir(exist_ok=True)\nprint(json.dumps({'uid':os.getuid()}))")
    subprocess.run(['scp', '-o', 'BatchMode=yes', str(bundle), 'DevContainer:' + remote_archive], check=True, timeout=90, capture_output=True)
    code = '''from pathlib import Path, PurePosixPath
import hashlib,json,tarfile,os
archive=Path(ARCHIVE);target=Path(TARGET);manifest=MANIFEST
assert os.getuid()==1000
assert hashlib.sha256(archive.read_bytes()).hexdigest()==SHA
expected={x['path']:x for x in manifest['files']}
target.mkdir(exist_ok=True)
with tarfile.open(archive,'r:gz') as t:
 members=t.getmembers()
 assert len(members)==len(expected) and {m.name for m in members}==set(expected)
 for m in members:
  p=PurePosixPath(m.name)
  assert not p.is_absolute() and '..' not in p.parts and m.isfile()
  dest=target/m.name
  assert dest.resolve().is_relative_to(target.resolve())
  raw=t.extractfile(m).read();assert len(raw)==expected[m.name]['bytes'] and hashlib.sha256(raw).hexdigest()==expected[m.name]['sha256']
  dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():assert hashlib.sha256(dest.read_bytes()).hexdigest()==expected[m.name]['sha256']
  else:dest.write_bytes(raw)
for n,x in expected.items():assert hashlib.sha256((target/n).read_bytes()).hexdigest()==x['sha256']
(target/'SOURCE_SNAPSHOT.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(path=str(target),commit=manifest['commit'],files=len(expected),archive_sha256=SHA,stream_sha_and_gzip_crc=True,exact_members=True,all_files_readback_sha=True,uid=os.getuid())))
'''.replace('ARCHIVE', repr(remote_archive)).replace('TARGET', repr(target)).replace('MANIFEST', repr(manifest)).replace('SHA', repr(sha))
    result = remote(code)
    (PRIVATE / (name + '_download_verified.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    (PRIVATE / (name + '_source_manifest.json')).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    receipts.append(result)
    print(json.dumps(result), flush=True)
all_receipts = [json.loads(p.read_text(encoding='utf-8')) for p in PRIVATE.glob('*_download_verified.json')]
(HERE / 'download_receipt.json').write_text(json.dumps(all_receipts, indent=2), encoding='utf-8')
