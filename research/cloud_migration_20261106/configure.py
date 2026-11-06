"""Configure new-host paths without rewriting frozen historical artifacts."""
import json,os,pwd,shutil,subprocess
from pathlib import Path

MIGRATION=Path('/home/ubuntu/cloud_migration_20261106')
ROOT=Path('/home/ubuntu/ICML2027')
PYTHON=Path('/home/ubuntu/.conda/envs/kv-graph/bin/python')
GIT=PYTHON.with_name('git')
def main():
    assert os.getuid()==1000 and (MIGRATION/'verification/RESTORE_VERIFIED.json').exists()
    assert json.loads((MIGRATION/'verification/RESTORE_VERIFIED.json').read_text())['passed']
    alias=Path('/home/liuxin')
    if alias.exists() or alias.is_symlink():assert alias.is_symlink() and alias.resolve()==Path.home()
    else:
        try:pwd.getpwnam('liuxin')
        except KeyError:pass
        else:raise RuntimeError('A real liuxin account exists; do not alias its home')
        subprocess.run(['sudo','-n','ln','-s','/home/ubuntu','/home/liuxin'],check=True)
    repo=ROOT/'Enhaced_LatentMAS_ADvanced'
    if not repo.exists():
        subprocess.run([str(GIT),'clone','--branch','main',str(MIGRATION/'metadata/local_code_history.bundle'),str(repo)],check=True)
        subprocess.run([str(GIT),'-C',str(repo),'remote','set-url','origin','https://github.com/Creee3/Enhaced_LatentMAS_ADvanced.git'],check=True)
    assert (repo/'.git').is_dir()
    copied=0
    for p in (MIGRATION/'metadata/local_management_state').rglob('*'):
        if p.is_file():
            dest=ROOT/'advanced_MOC/experiments'/p.relative_to(MIGRATION/'metadata/local_management_state')
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert dest.read_bytes()==p.read_bytes()
            else:shutil.copy2(p,dest)
            copied+=1
    links=[]
    for group in ['current','source','history']:
        manifest=json.loads((MIGRATION/'metadata'/(group+'_recovery_manifest.json')).read_text())
        for entry in manifest['symbolic_links_recorded_not_followed']:
            path=ROOT/entry['path'];target=Path(entry['target'])
            if not target.is_absolute():target=path.parent/target
            resolved=target.resolve()
            allowed=resolved.is_relative_to(ROOT) or resolved.is_relative_to(Path.home()/'models')
            if allowed and target.exists():
                path.parent.mkdir(parents=True,exist_ok=True)
                if not path.is_symlink():
                    assert not path.exists();path.symlink_to(entry['target'])
                assert path.readlink()==Path(entry['target'])
                links.append({'path':entry['path'],'restored':True})
            else:links.append({'path':entry['path'],'restored':False,'reason':'target absent or outside restored scope'})
    mapping={'old_home':'/home/liuxin','new_home':'/home/ubuntu','compatibility_symlink':str(alias),
        'research_root':str(ROOT),'repo':str(repo),'python':str(PYTHON),
        'model':'/home/ubuntu/models/Qwen3-8B','shared_budget':str(ROOT/'experiments/latentmas_cumulative_history_20261025/cumulative_budget.json'),
        'experiments_state':'PAUSED_BY_USER','automation_state':'PAUSED',
        'old_pid_metadata_historical_only':True,'model_forward_calls':0,'local_management_files':copied,'symlinks':links}
    (MIGRATION/'paths.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'MIGRATION_PAUSED.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n')
    (MIGRATION/'activate.sh').write_text('''# Source this file; it only activates the offline environment.
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /home/ubuntu/.conda/envs/kv-graph
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export LATENTMAS_ROOT=/home/ubuntu/ICML2027/experiments/latentmas_cumulative_history_20261025
export LATENTMAS_MODEL=/home/ubuntu/models/Qwen3-8B
export LATENTMAS_EXPERIMENT_STATE=PAUSED_BY_USER
cd /home/ubuntu/ICML2027/Enhaced_LatentMAS_ADvanced
''')
    bashrc=Path.home()/'.bashrc';text=bashrc.read_text()
    marker='# BEGIN LatentMAS migration 20261106'
    if marker not in text:
        with bashrc.open('a') as f:f.write('\n'+marker+'\nlatentmas() { source /home/ubuntu/cloud_migration_20261106/activate.sh; }\n# END LatentMAS migration 20261106\n')
    workspace={'folders':[{'name':'LatentMAS code','path':'Enhaced_LatentMAS_ADvanced'}],
        'settings':{'python.defaultInterpreterPath':str(PYTHON),'python.terminal.activateEnvironment':True},
        'extensions':{'recommendations':['ms-python.python','ms-python.vscode-pylance']}}
    (ROOT/'LatentMAS.code-workspace').write_text(json.dumps(workspace,indent=2)+'\n')
    print(json.dumps({'configured':True,'research_root':str(ROOT),'local_management_files':copied,
        'restored_symlinks':sum(x['restored'] for x in links),'unresolved_symlinks':sum(not x['restored'] for x in links),
        'historical_files_modified':0,'experiments_resumed':False}),flush=True)
if __name__=='__main__':main()
