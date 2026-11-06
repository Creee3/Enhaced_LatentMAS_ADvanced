"""Record an already-captured healthy snapshot; no extra polling or scoring."""
import argparse,base64,datetime,json
from pathlib import Path
from capture_current import HERE,PRIVATE,call
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--utc',required=True);args=ap.parse_args()
 utc=datetime.datetime.fromisoformat(args.utc.replace('Z','+00:00'));assert utc.utcoffset().total_seconds()==0
 stamp=utc.strftime('%Y%m%dT%H%M%SZ');active=json.loads((PRIVATE/'cloud_active.json').read_text())
 files=sorted(PRIVATE.glob('inspect_*.json'),key=lambda p:p.stat().st_mtime);assert files
 full=json.loads(files[-1].read_bytes());summary=json.loads((PRIVATE/'latest_summary.json').read_text())
 assert full['source']['run_lock_sha256']==active['source_lock_sha256'] and all(full['snapshots'].values())
 assert not full['duplicates'] and not full['unexpected'] and not full['queue']['failed_held']
 assert not full['manager_failure'] and all(not w['failure'] for w in full['workers'])
 assert full['manager_identity'] or full['manager_exit'],'Manager exit requires actual exit evidence'
 live={w['pid'] for w in full['workers'] if w['identity']}
 for w in full['workers']:
  assert w['identity'] or (w['exit'] and w['exit'].get('completed')),'Unexplained worker exit'
 assert all(x['pid'] in live and x.get('metadata',{}).get('experiment')==full['run'] for x in full['own_reservations'].values())
 assert full['budget']['unknown_cost_upper_bound_tokens']==58106 and sum(full['budget'].values())<=115000000
 if len(files)>1:
  previous=json.loads(files[-2].read_bytes());assert full['combined_unique']>=previous['combined_unique']
 record=dict(utc=args.utc,snapshot=summary,source_lock_valid=True,snapshot_shas_valid=True,
  own_inflight_rids=len(full['own_reservations']),all_reservation_owners_verified=True,new_failure=False,
  accuracy_computed=False,manual_settlements=0,full_inspect_file=files[-1].name)
 target=PRIVATE/'management'/('heartbeat_'+stamp+'.json');target.parent.mkdir(exist_ok=True)
 target.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 remote=active['run']+'/management/'+target.name;payload=base64.b64encode(target.read_bytes()).decode()
 call('from pathlib import Path\nimport base64\nPath('+repr(remote)+').write_bytes(base64.b64decode('+repr(payload)+'))')
 q=full['queue'];b=full['budget'];counts='/'.join(str(w['committed']) for w in full['workers'])
 text=(f"\n{utc.strftime('%Y-%m-%dT%H:%MZ')}巡检：一次完整UTF8 inspect私有落地，{full['combined_unique']}/675唯一"
  f"（新{full['new_unique']}+复用4），剩{full['remaining']}；{q['complete']}complete/{q['running']}running/{q['pending']}pending/0held。"
  f"源锁与快照SHA完好，无重复/越界/新故障，manager28169与{len(live)}个真实自有worker正常，各提交{counts}。"
  f"实际RID预留均归属本run存活worker；账本{b['actual_tokens']}/reserved{b['reserved_tokens']}/unknown{b['unknown_cost_upper_bound_tokens']}，"
  "健康在途不手工结算，未算未完成准确率。中文接续两地保存，原科学锁/旧暂停run不变，监控ACTIVE40分钟继续。\n")
 progress=HERE/'progress.md';progress.write_text(progress.read_text(encoding='utf-8')+text,encoding='utf-8')
 print(json.dumps(dict(recorded=True,unique=full['combined_unique'],live_workers=len(live),new_failure=False)))
if __name__=='__main__':main()
