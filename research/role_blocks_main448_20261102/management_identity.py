"""Conservative management-only /proc access guard; never grants ownership."""
def guarded_identity(reader, pid, uid_reader, expected_uid):
    try:
        if uid_reader(pid) != expected_uid:
            return None
        return reader(pid)
    except (PermissionError, FileNotFoundError, ProcessLookupError):
        return None
