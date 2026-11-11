"""Pure resource admission rules for the authorized six-worker pool."""
def admit(*,live_total,live_gpu,all_ready,blocked,util,free,total,pending):
 if not pending or blocked or live_total>=6 or live_gpu>=4 or not all_ready:return False
 if not live_gpu:return util<=30 and free/total>=.70
 return free>=21*1024
def cpu_set(text):
 result=[]
 for block in text.strip().split(','):
  if not block:continue
  if '-' in block:
   lo,hi=map(int,block.split('-'));result.extend(range(lo,hi+1))
  else:result.append(int(block))
 return sorted(set(result))
