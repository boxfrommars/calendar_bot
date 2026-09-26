"""Opt-in private model transcripts; never enable raw SDK/HTTP logging."""

import asyncio
import json
import logging
import os
import re
import threading
import uuid
from datetime import timedelta
from pathlib import Path

from .domain import utc_now

log = logging.getLogger(__name__)
TRACE_NAME = re.compile(r"model-trace-[0-9a-f]{32}\.jsonl")
MAX_RECORD_BYTES = 256 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
RETENTION = timedelta(days=1)


class ModelTrace:
    def __init__(self, directory: Path, *, clock=utc_now):
        self.directory = directory.resolve()
        self.clock = clock
        self.lock = threading.Lock()

    def new_session(self) -> str:
        return uuid.uuid4().hex

    async def record(self, session: str, event: dict) -> None:
        try:
            async with asyncio.timeout(1.0):
                await asyncio.to_thread(self._write, session, event)
        except (OSError, ValueError, TypeError, TimeoutError) as exc:
            log.warning("model_trace_write_failed type=%s", type(exc).__name__)

    def _write(self, session: str, event: dict) -> None:
        name = f"model-trace-{session}.jsonl"
        if not TRACE_NAME.fullmatch(name):
            raise ValueError("Invalid trace session")
        now = self.clock()
        entry = {"recorded_at": now.isoformat(), **event}
        text = json.dumps(entry, ensure_ascii=False)
        if len(text.encode("utf-8")) > MAX_RECORD_BYTES:
            # Keep a valid JSON record and make truncation explicit.
            text = json.dumps(
                {
                    "recorded_at": now.isoformat(),
                    "kind": event.get("kind"),
                    "truncated": True,
                    "preview": text[: MAX_RECORD_BYTES // 8],
                },
                ensure_ascii=False,
            )
        with self.lock:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            path = self.directory / name
            if path.is_symlink() or path.resolve().parent != self.directory:
                raise ValueError("Unsafe trace path")
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
            with os.fdopen(os.open(path, flags, 0o600), "a", encoding="utf-8") as stream:
                stream.write(text + "\n")
            self._prune(now.timestamp())

    def _prune(self, now: float) -> None:
        files = []
        for path in self.directory.iterdir():
            if (
                TRACE_NAME.fullmatch(path.name)
                and path.is_file()
                and not path.is_symlink()
                and path.resolve().parent == self.directory
            ):
                stat = path.stat()
                files.append((stat.st_mtime, stat.st_size, path))
        total = sum(size for _, size, _ in files)
        for modified, size, path in sorted(files):
            if modified < now - RETENTION.total_seconds() or total > MAX_TOTAL_BYTES:
                path.unlink()
                total -= size
