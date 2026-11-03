"""Management entry after archived /proc PermissionError; no science edits."""
import os
from pathlib import Path
from management_identity import guarded_identity

def install():
    import cloud_common
    if getattr(cloud_common.identity,'permission_guard',False):return
    original=cloud_common.identity
    def safe(pid):
        return guarded_identity(original,pid,lambda p:(Path('/proc')/str(p)).stat().st_uid,os.getuid())
    safe.permission_guard=True
    cloud_common.identity=safe

if __name__=='__main__':
    install()
    import manager
    manager.main()
