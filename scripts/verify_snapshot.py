"""Check publication integrity and excluded content; never load a model."""
import hashlib
import json
from pathlib import Path
import re
from reproduce_results import rebuild,check

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'provenance/published_source_manifest.json').read_text(encoding='utf-8'))
    for name,entry in manifest['source_files'].items():
        p=ROOT/name;b=p.read_bytes()
        assert len(b)==entry['bytes'] and hashlib.sha256(b).hexdigest()==entry['sha256'],'Source digest mismatch: '+name
    cells=rebuild();check(cells)
    bad_names={'cumulative_budget.json','budget.lock','requests_raw.jsonl','reused_results.jsonl','.env'}
    bad_suffix={'.safetensors','.pt','.pth','.bin','.npz','.gz','.zip','.pem','.key','.log','.jsonl'}
    patterns=[re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),re.compile(rb'gh[pousr]_[A-Za-z0-9]{30,}'),re.compile(rb'github_pat_[A-Za-z0-9_]{40,}')]
    scanned=0
    for p in ROOT.rglob('*'):
        if not p.is_file() or any(x in {'.git','__pycache__','.venv'} for x in p.relative_to(ROOT).parts):continue
        rel=p.relative_to(ROOT).as_posix()
        # External OBF sources may be fetched explicitly by an end user.
        if rel in ['snapshot/run/author_HOBF.py','snapshot/run/author_BaseKVCompressor.py']:continue
        assert p.name not in bad_names and p.suffix.lower() not in bad_suffix,'Excluded file: '+rel
        assert p.stat().st_size<2_000_000,'Unexpected large file: '+rel
        b=p.read_bytes();assert not any(x.search(b) for x in patterns),'Credential-like content: '+rel
        if p.suffix=='.json':
            value=json.loads(b)
            def inspect(x):
                if isinstance(x,dict):
                    assert not ({'answer_text','generated_ids','input_ids','gold','question'}&x.keys()),'Raw data field: '+rel
                    for v in x.values():inspect(v)
                elif isinstance(x,list):
                    for v in x:inspect(v)
            inspect(value)
        scanned+=1
    print(json.dumps(dict(passed=True,frozen_files=len(manifest['source_files']),unique=3584,cells=len(cells),publication_files_scanned=scanned,model_calls=0)))

if __name__=='__main__':main()
