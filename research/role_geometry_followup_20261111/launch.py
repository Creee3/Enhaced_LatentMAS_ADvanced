"""Explicit single controller launch; no implicit generation on import/tests."""
import datetime,json,subprocess,sys
from remote import HERE,PRIVATE
def main():
 assert json.loads((PRIVATE/'source_sealed.json').read_text())['passed'];path=PRIVATE/'controller_process.json';assert not path.exists(),'Inspect existing controller; never duplicate'
 with (PRIVATE/'controller.stdout.log').open('ab') as out,(PRIVATE/'controller.stderr.log').open('ab') as err:
  p=subprocess.Popen([sys.executable,'-u','-X','utf8',str(HERE/'orchestrate.py')],cwd=HERE,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
 details=dict(pid=p.pid,script=str(HERE/'orchestrate.py'),creation_time=datetime.datetime.now().astimezone().isoformat(),max_workers=6,gpus=[0,1],per_gpu=3)
 path.write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8');print(json.dumps(details))
if __name__=='__main__':main()
