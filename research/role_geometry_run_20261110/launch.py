"""Explicit launch entry; launches only after sealed evidence exists."""
import datetime,json,subprocess,sys
from pathlib import Path
from remote import HERE,PRIVATE,active

def main():
 proof=json.loads((PRIVATE/'source_sealed.json').read_text());assert proof['passed']
 pfile=PRIVATE/'controller_process.json'
 if pfile.exists():raise RuntimeError('Controller already launched; inspect exact identity, do not duplicate')
 with (PRIVATE/'controller.stdout.log').open('ab') as out,(PRIVATE/'controller.stderr.log').open('ab') as err:
  p=subprocess.Popen([sys.executable,'-u','-X','utf8',str(HERE/'orchestrate.py')],cwd=HERE,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
 details=dict(pid=p.pid,script=str(HERE/'orchestrate.py'),creation_time=datetime.datetime.now().astimezone().isoformat(),max_workers=6,gpu1_max=4,gpu0_max=2)
 pfile.write_text(json.dumps(details,indent=2)+'\n',encoding='utf-8');print(json.dumps(details))
if __name__=='__main__':main()
