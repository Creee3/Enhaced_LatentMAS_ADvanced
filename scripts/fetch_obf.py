"""Optional explicit fetch of external HeadOBF dependencies, with SHA pinning."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'snapshot/run/author_manifest.json').read_text())
    names={'BaseKVCompressor':'author_BaseKVCompressor.py','HOBF':'author_HOBF.py'}
    for key,name in names.items():
        item=manifest['files'][key];target=ROOT/'snapshot/run'/name
        if target.exists():data=target.read_bytes()
        else:
            request=urllib.request.Request(item['url'],headers={'User-Agent':'latentmas-research-dependency-fetcher'})
            with urllib.request.urlopen(request,timeout=30) as response:data=response.read()
        assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],'Upstream hash mismatch: '+name
        target.write_bytes(data)
    print(json.dumps(dict(pinned_revision=manifest['revision'],files_verified=2,model_calls=0)))

if __name__=='__main__':main()
