import errno
import os
import re
import uuid
from pathlib import Path

from .domain import UserError


def _acquire(file) -> None:
    if os.name == "nt":
        import msvcrt

        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release(file) -> None:
    if os.name == "nt":
        import msvcrt

        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def locked_instance(database_path: Path) -> str | None:
    """Read an existing holder's identity without creating or writing any files."""
    path = database_path.with_name(database_path.name + ".lock")
    try:
        file = path.open("rb")
    except FileNotFoundError:
        return None
    with file:
        try:
            _acquire(file)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
        else:
            _release(file)
            return None
        # Windows locks byte zero, so readers must not include it in their read.
        file.seek(1)
        identity = file.read(33)
        if re.fullmatch(rb"[0-9a-f]{32}", identity):
            return identity.decode("ascii")
        return None


class InstanceLock:
    """An OS lock survives neither crashes nor process exit; the file may remain."""

    def __init__(self, database_path: Path):
        self.path = database_path.with_name(database_path.name + ".lock")
        self.file = None
        self.instance_id = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = os.fdopen(os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666), "r+b")
        try:
            if os.fstat(self.file.fileno()).st_size == 0:
                self.file.write(b"0")
                self.file.flush()
            _acquire(self.file)
        except OSError:
            self.file.close()
            self.file = None
            raise UserError("База уже используется другим процессом бота или миграцией.") from None
        try:
            self.instance_id = uuid.uuid4().hex
            self.file.seek(1)
            self.file.write(self.instance_id.encode("ascii"))
            self.file.truncate()
            self.file.flush()
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *args):
        if self.file:
            try:
                _release(self.file)
            finally:
                self.file.close()
                self.file = None
