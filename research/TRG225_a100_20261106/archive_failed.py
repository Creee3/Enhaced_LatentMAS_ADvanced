"""Archive only an exited worker using the existing validated archive adapter."""
import contextlib,io,json,sys,types
from pathlib import Path
from capture_current import active,call,PRIVATE,HERE
def main():
 active();module=types.ModuleType('deploy');module.PRIVATE=PRIVATE;module.call=call;sys.modules['deploy']=module
 source=(HERE.parent/'attention_control225_20261103/archive_failed_worker.py').read_text(encoding='utf-8')
 source=source.replace("tracefile=out/'trace.jsonl'","for extra in (run/'management').rglob('*'):\n if extra.is_file():add('run_snapshot/management/'+extra.relative_to(run/'management').as_posix(),extra)\ntracefile=out/'trace.jsonl'")
 with contextlib.redirect_stdout(io.StringIO()) as output:exec(compile(source,'existing_archive_adapter','exec'),{'__name__':'__main__'})
 value=json.loads(output.getvalue());print(json.dumps({k:value[k] for k in ['archive_sha256','bytes','members','stream_sha','gzip_crc','tar_exact_members','all_member_sha','safe_full_extract_sha']}))
 print(json.dumps({k:value['cost'][k] for k in ['pid','committed','request_starts','own_reserved','actual','unknown','manual_settlements']}))
if __name__=='__main__':main()
