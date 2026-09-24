import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from aiogram.methods import GetUpdates

from calendar_bot.locking import InstanceLock, locked_instance
from calendar_bot.polling import PollingMonitor, check_health, health_path
from tests.polling_support import PRIVATE, MonotonicClock


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "calendar.sqlite3"
        self.clock = MonotonicClock()

    def monitor(self, instance):
        return PollingMonitor(self.path, instance.instance_id, clock=self.clock)

    def succeed(self, monitor):
        monitor.set_phase("polling")
        asyncio.run(monitor(AsyncMock(return_value=[]), None, GetUpdates()))
        monitor.publish()

    def test_absent_state_does_not_create_directories_or_files(self):
        self.assertEqual(check_health(self.path / "nested").reason, "not_running")
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [])

    def test_current_instance_startup_success_and_stop(self):
        with InstanceLock(self.path) as instance:
            monitor = self.monitor(instance)
            self.assertEqual(locked_instance(self.path), instance.instance_id)
            self.assertEqual(check_health(self.path, clock=self.clock).reason, "snapshot_missing")
            monitor.publish()
            self.assertEqual(check_health(self.path, clock=self.clock).reason, "starting")
            monitor.set_phase("polling")
            self.assertEqual(check_health(self.path, clock=self.clock).reason, "no_success")
            self.succeed(monitor)
            before = {p.name: p.read_bytes() for p in self.path.parent.glob("*.json")}
            self.assertTrue(check_health(self.path, clock=self.clock).healthy)
            self.assertEqual(
                before, {p.name: p.read_bytes() for p in self.path.parent.glob("*.json")}
            )
            monitor.set_phase("stopping")
            self.assertEqual(check_health(self.path, clock=self.clock).reason, "stopping")
        self.assertIsNone(locked_instance(self.path))
        self.assertEqual(check_health(self.path, clock=self.clock).reason, "not_running")
        self.assertFalse(self.path.exists())

    def test_crash_restart_and_migration_cannot_reuse_previous_success(self):
        with InstanceLock(self.path) as old:
            monitor = self.monitor(old)
            self.succeed(monitor)
        # Emulate an exit without a stopped snapshot, then a migration/new run holding the lock.
        self.assertEqual(check_health(self.path, clock=self.clock).reason, "not_running")
        with InstanceLock(self.path) as new:
            self.assertNotEqual(old.instance_id, new.instance_id)
            self.assertEqual(check_health(self.path, clock=self.clock).reason, "instance_changed")
            self.monitor(new).publish()
            self.assertEqual(check_health(self.path, clock=self.clock).reason, "starting")

    def test_freshness_uses_monotonic_time_not_wall_clock(self):
        with InstanceLock(self.path) as instance:
            monitor = self.monitor(instance)
            self.succeed(monitor)
            for wall_clock in (0, 10**12):
                with patch("time.time", return_value=wall_clock):
                    self.assertTrue(check_health(self.path, clock=self.clock).healthy)
            self.clock.advance(15)
            self.assertTrue(check_health(self.path, clock=self.clock).healthy)
            self.clock.advance(0.1)
            self.assertEqual(check_health(self.path, clock=self.clock).reason, "snapshot_stale")
            self.clock.advance(104.9)
            monitor.publish()
            self.assertTrue(check_health(self.path, clock=self.clock).healthy)
            self.clock.advance(0.1)
            monitor.publish()
            self.assertEqual(check_health(self.path, clock=self.clock).reason, "last_success_stale")

    def test_corrupt_oversized_future_and_unsafe_state_fail_closed(self):
        with InstanceLock(self.path) as instance:
            monitor = self.monitor(instance)
            self.succeed(monitor)
            valid = json.loads(health_path(self.path).read_text())
            payloads = ["broken", "[", "x" * 8193, "null", "[]"]
            payloads.append(json.dumps({k: v for k, v in valid.items() if k != "last_error_type"}))
            for key, value in (
                ("version", True),
                ("updated", float("nan")),
                ("updated", self.clock() + 1),
                ("phase", PRIVATE),
                ("last_error_type", PRIVATE),
                ("last_success", -1),
            ):
                payloads.append(json.dumps({**valid, key: value}))
            for payload in payloads:
                with self.subTest(payload=payload[:30]):
                    health_path(self.path).write_text(payload, encoding="utf-8")
                    result = check_health(self.path, clock=self.clock)
                    self.assertEqual(result.reason, "snapshot_invalid")
                    self.assertNotIn(PRIVATE, result.describe())

    def test_atomic_write_leaves_previous_complete_snapshot_on_failure(self):
        with InstanceLock(self.path) as instance:
            monitor = self.monitor(instance)
            self.succeed(monitor)
            before = health_path(self.path).read_bytes()
            with patch("calendar_bot.polling.os.replace", side_effect=PermissionError(PRIVATE)):
                with self.assertLogs("calendar_bot.polling", level="ERROR") as captured:
                    monitor.publish()
            self.assertEqual(health_path(self.path).read_bytes(), before)
            self.assertNotIn(PRIVATE, "\n".join(captured.output))

    def test_cli_needs_only_database_path_and_checks_lock_across_processes(self):
        env = {
            **os.environ,
            "DATABASE_PATH": str(self.path),
            "BOT_TOKEN": "",
            "OPENAI_API_KEY": "",
            "ALLOWED_USER_IDS": "intentionally-invalid",
            "OPENAI_MODEL": "",
        }
        command = [
            sys.executable,
            "-m",
            "calendar_bot",
            "health",
            "--env-file",
            str(self.path.parent / "absent.env"),
        ]

        def invoke():
            return subprocess.run(
                command, env=env, capture_output=True, text=True, encoding="utf-8", timeout=30
            )

        with InstanceLock(self.path) as instance:
            monitor = PollingMonitor(self.path, instance.instance_id)
            self.succeed(monitor)
            result = invoke()
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("status=healthy", result.stdout)
        result = invoke()
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("reason=not_running", result.stdout)
        self.assertFalse(self.path.exists())
