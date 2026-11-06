"""Compact read-only deployment summary; never imports experiment launchers."""
import json
from pathlib import Path

def main():
    migration=Path('/home/ubuntu/cloud_migration_20261106')
    proof=migration/'verification/MIGRATION_VERIFIED.json'
    evidence=migration/'verification/EVIDENCE_VERIFIED.json'
    result={'host':'liuxin-163','research_root':'/home/ubuntu/ICML2027',
            'experiments_state':'PAUSED_BY_USER','automation_state':'PAUSED',
            'migration_verified':proof.exists()}
    if evidence.exists():
        e=json.loads(evidence.read_text())
        for name in ['restored_files','old_exact_answers','frozen_baseline','role_answers','common_questions','G_answers','budget']:
            result[name]=e[name]
    if proof.exists():
        e=json.loads(proof.read_text());result['model_files_verified']=e['model_frozen_files']
        result['runtime_versions']={n:e['core_versions'][n] for n in ['torch','transformers','numpy','scipy']}
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
