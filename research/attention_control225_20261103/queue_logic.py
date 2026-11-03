def choose(tasks):
    complete={t['id'] for t in tasks if t['state']=='complete'}
    ready=[t for t in tasks if t['state']=='pending' and set(t.get('depends',[]))<=complete]
    return min(ready,key=lambda t:(t['priority'],t['id'])) if ready else None
def terminal(tasks):
    return not any(t['state']=='running' for t in tasks) and choose(tasks) is None
