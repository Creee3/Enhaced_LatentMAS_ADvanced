"""Remote stdout is exclusively a gzip tar stream; progress/receipt use stderr."""
import fcntl,gzip,hashlib,io,json,os,stat,sys,tarfile,time
from pathlib import Path

ROOT=Path('/home/liuxin/ICML2027')
SKIP_DIRS={'.secrets','.cache','.bootstrap-venv','.venv-gptswarm','.pydeps','.venv','node_modules','__pycache__','.tools','.pytest_cache'}
SECRET_NAMES={'.env','credentials','tokens.json','token','id_rsa','id_ed25519'}


def selected(p,group):
    parts=p.relative_to(ROOT).parts
    current=('experiments','latentmas_cumulative_history_20261025')
    if group=='current':return parts[:2]==current
    if group=='history':return parts[0]=='experiments' and parts[:2]!=current
    if group=='source':return parts[0]!='experiments'
    raise ValueError(group)


def secret(p):
    name=p.name
    return name in SECRET_NAMES or (name.endswith('.env') and 'template' not in name.lower()) or (name=='config' and '.git' in p.parts)


class Writer:
    def __init__(self):self.h=hashlib.sha256();self.size=0
    def write(self,b):
        sys.stdout.buffer.write(b);self.h.update(b);self.size+=len(b);return len(b)
    def flush(self):sys.stdout.buffer.flush()


class Reader:
    def __init__(self,f):self.f=f;self.h=hashlib.sha256();self.size=0
    def read(self,n=-1):
        b=self.f.read(n);self.h.update(b);self.size+=len(b);return b


def main(group):
    start=time.time();paths=[];links=[];excluded=[]
    for base,dirs,files in os.walk(ROOT,followlinks=False):
        for d in list(dirs):
            p=Path(base)/d
            if d in SKIP_DIRS:
                if selected(p,group):excluded.append(dict(path=str(p.relative_to(ROOT)),reason='rebuildable_environment_cache_or_credentials'))
                dirs.remove(d)
            elif p.is_symlink():
                if selected(p,group):links.append(dict(path=str(p.relative_to(ROOT)),target=os.readlink(p)))
                dirs.remove(d)
            elif group=='current' and p.relative_to(ROOT).parts[0]!='experiments':dirs.remove(d)
            elif group=='current' and len(p.relative_to(ROOT).parts)==2 and p.name!='latentmas_cumulative_history_20261025':dirs.remove(d)
            elif group=='source' and p.relative_to(ROOT).parts[0]=='experiments':dirs.remove(d)
            elif group=='history' and p.relative_to(ROOT).parts[:2]==('experiments','latentmas_cumulative_history_20261025'):dirs.remove(d)
        for name in files:
            p=Path(base)/name
            if not selected(p,group):continue
            if p.is_symlink():links.append(dict(path=str(p.relative_to(ROOT)),target=os.readlink(p)));continue
            if secret(p):excluded.append(dict(path=str(p.relative_to(ROOT)),reason='credential_or_git_remote_configuration'));continue
            s=p.stat()
            if stat.S_ISREG(s.st_mode):paths.append((p,s))
    paths.sort(key=lambda x:str(x[0]));manifest={};changed=[];writer=Writer()
    ledger=ROOT/'experiments/latentmas_cumulative_history_20261025/cumulative_budget.json'
    ledger_bytes=None
    if group=='current':
        with ledger.with_name('budget.lock').open('r') as lock:
            fcntl.flock(lock,fcntl.LOCK_SH);ledger_bytes=ledger.read_bytes()
    def metadata(b):return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
    with gzip.GzipFile(fileobj=writer,mode='wb',compresslevel=1,mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w|') as tar:
            for i,(p,before) in enumerate(paths,1):
                name='ICML2027/'+p.relative_to(ROOT).as_posix()
                b=ledger_bytes if p==ledger and ledger_bytes is not None else None
                info=tarfile.TarInfo(name);info.size=len(b) if b is not None else before.st_size
                info.mode=stat.S_IMODE(before.st_mode);info.mtime=before.st_mtime;info.uid=before.st_uid;info.gid=before.st_gid
                with (io.BytesIO(b) if b is not None else p.open('rb')) as f:
                    reader=Reader(f);tar.addfile(info,reader)
                assert reader.size==info.size,name
                after=p.stat()
                stable=(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
                if not stable and p!=ledger:changed.append(name)
                manifest[name]=dict(bytes=reader.size,sha256=reader.h.hexdigest(),mode=info.mode,mtime_ns=before.st_mtime_ns,
                    source_stable=stable,ledger_locked_snapshot=p==ledger)
                if i%500==0:print(json.dumps(dict(progress_files=i,total_files=len(paths),compressed_bytes=writer.size)),file=sys.stderr,flush=True)
            recovery=dict(group=group,source_root=str(ROOT),started=start,snapshot_end=time.time(),files=manifest,
                excluded=excluded,symbolic_links_recorded_not_followed=links,changed_source_files=changed,
                pretrained_models_outside_scope='/home/liuxin/models; preserved by per-run model_lock.json',
                secrets_not_included=True,experiments_not_resumed=True)
            b=(json.dumps(recovery,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
            info=tarfile.TarInfo('recovery_manifest.json');info.size=len(b);tar.addfile(info,io.BytesIO(b))
    writer.flush()
    receipt=dict(group=group,stream_sha256=writer.h.hexdigest(),compressed_bytes=writer.size,source_files=len(paths),
        tar_members=len(paths)+1,source_bytes=sum(v['bytes'] for v in manifest.values()),
        changed_source_files=changed,excluded_count=len(excluded),recorded_links=len(links),seconds=time.time()-start)
    print('RECEIPT_JSON '+json.dumps(receipt),file=sys.stderr,flush=True)


if __name__=='__main__':main(GROUP_VALUE)
