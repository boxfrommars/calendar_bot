"""Exercise the lightweight entry point in fresh interpreters, without bot SDKs."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calendar_bot.health import check_health, health_path
from calendar_bot.locking import InstanceLock

# An in-process import check would miss dependencies already loaded by other tests.
# Remember attempted imports too: main() can catch an ImportError and return 1.
GUARDED_CLI = """
import os
import runpy
import sys
import time

# Match the synthetic snapshot's clock, independent of uptime and machine load.
time.monotonic = lambda: 1000.0

forbidden = {
    'aiogram', 'openai', 'pydantic', 'pydantic_core', 'aiohttp', 'aiofiles',
    'httpx', 'httpcore', 'anyio', 'aiosqlite', 'sqlite3', 'asyncio',
    'calendar_bot.polling', 'calendar_bot.runtime', 'calendar_bot.parser',
    'calendar_bot.storage', 'calendar_bot.service', 'calendar_bot.telegram',
    'calendar_bot.worker', 'calendar_bot.presentation', 'calendar_bot.systemd',
}
violations = []

def blocked(name):
    return any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden)

class ImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        if blocked(fullname):
            violations.append(fullname)
            raise ImportError('Health imported a runtime dependency')

def audit(event, args):
    if event == 'open':
        _, mode, flags = args
        if (mode and any(c in mode for c in 'wax+')) or flags & (
            os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        ):
            violations.append('file_write')
    elif event in {
        'sqlite3.connect', 'socket.connect', 'socket.getaddrinfo',
        'os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.truncate',
    }:
        violations.append(event)
    if violations:
        raise AssertionError('Health attempted a forbidden operation')

sys.meta_path.insert(0, ImportGuard())
sys.addaudithook(audit)
sys.argv = ['calendar_bot', *sys.argv[1:]]
try:
    runpy.run_module('calendar_bot', run_name='__main__')
finally:
    violations.extend(name for name in sys.modules if blocked(name))
    if violations:
        raise AssertionError('Health isolation failed: ' + ', '.join(violations))
"""


class LightweightHealthTests(unittest.TestCase):
    def setUp(self):
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(temporary)
        self.path = self.root / "synthetic.sqlite3"
        self.now = 1000.0
        self.env_file = self.root / "absent.env"
        self.env = {
            **os.environ,
            "DATABASE_PATH": str(self.path),
            "BOT_TOKEN": "",
            "OPENAI_API_KEY": "",
            "ALLOWED_USER_IDS": "intentionally-invalid",
            "OPENAI_MODEL": "",
        }

    def snapshot(self, instance, **changes):
        now = self.now
        data = {
            "version": 1,
            "instance_id": instance.instance_id,
            "pid": os.getpid(),
            "phase": "polling",
            "updated": now,
            "phase_started": now - 300,
            "last_started": now - 1,
            "last_completed": now - 1,
            "last_success": now - 1,
            "last_error_type": None,
            "failures": 0,
        }
        health_path(self.path).write_text(json.dumps(data | changes), encoding="utf-8")

    def invoke(self, reason, *, error="none", success=True):
        before = {p.name: p.read_bytes() for p in self.root.iterdir() if p.suffix != ".lock"}
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                GUARDED_CLI,
                "health",
                "--env-file",
                str(self.env_file),
            ],
            cwd=Path(__file__).resolve().parents[1],
            env=self.env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
        )
        self.assertEqual(result.returncode, 0 if reason == "ok" else 1, result.stderr)
        self.assertEqual(result.stderr, "")
        status = "healthy" if reason == "ok" else "unhealthy"
        age = r"\d+\.\d{3}" if success else "none"
        self.assertRegex(
            result.stdout,
            rf"\Apolling_health status={status} reason={reason} "
            rf"last_success_age={age} error_type={error}\n\Z",
        )
        self.assertEqual(
            before, {p.name: p.read_bytes() for p in self.root.iterdir() if p.suffix != ".lock"}
        )
        self.assertFalse(self.path.exists())

    def test_success_and_negative_states_without_runtime_imports_or_io_side_effects(self):
        self.invoke("not_running", success=False)
        with InstanceLock(self.path) as instance:
            self.invoke("snapshot_missing", success=False)
            cases = [
                ("ok", {}),
                ("starting", {"phase": "starting", "last_success": None}),
                ("no_success", {"last_success": None}),
                ("stopping", {"phase": "stopping"}),
                ("stopped", {"phase": "stopped"}),
                (
                    "snapshot_stale",
                    dict.fromkeys(
                        ("updated", "last_started", "last_completed", "last_success"),
                        self.now - 30,
                    ),
                ),
                (
                    "last_success_stale",
                    {
                        "last_success": self.now - 150,
                        "last_error_type": "TelegramNetworkError",
                        "failures": 5,
                    },
                ),
                ("instance_changed", {"instance_id": "0" * 32}),
            ]
            for reason, changes in cases:
                with self.subTest(reason=reason):
                    self.snapshot(instance, **changes)
                    self.invoke(
                        reason,
                        error=changes.get("last_error_type", "none"),
                        success=reason != "instance_changed"
                        and changes.get("last_success", 1) is not None,
                    )
            for content in ("broken", "x" * 8193):
                health_path(self.path).write_text(content, encoding="utf-8")
                self.invoke("snapshot_invalid", success=False)
            self.snapshot(instance)
        # A fresh successful snapshot cannot substitute for the current OS lock.
        self.invoke("not_running", success=False)
        with InstanceLock(self.path):
            self.invoke("instance_changed", success=False)

    def test_env_file_and_environment_precedence(self):
        self.env_file.write_text(f"DATABASE_PATH='{self.path.as_posix()}'\n", encoding="utf-8")
        del self.env["DATABASE_PATH"]
        with InstanceLock(self.path) as instance:
            self.snapshot(instance)
            self.invoke("ok")
            self.env["DATABASE_PATH"] = str(self.root / "missing" / "other.sqlite3")
            self.invoke("not_running", success=False)

    def test_lock_owner_is_rechecked_after_snapshot_read(self):
        with InstanceLock(self.path) as instance:
            self.snapshot(instance)
            for replacement in (None, "0" * 32):
                with self.subTest(replacement=replacement):
                    with patch(
                        "calendar_bot.health.locked_instance",
                        side_effect=[instance.instance_id, replacement],
                    ) as read_lock:
                        self.assertEqual(
                            check_health(self.path, clock=lambda: self.now).reason,
                            "instance_changed",
                        )
                        self.assertEqual(read_lock.call_count, 2)

    def test_unreadable_state_has_safe_reason(self):
        with patch("calendar_bot.health.locked_instance", side_effect=PermissionError("private")):
            self.assertEqual(
                check_health(self.path).describe(),
                "polling_health status=unhealthy reason=state_unreadable "
                "last_success_age=none error_type=none",
            )
