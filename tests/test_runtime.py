import asyncio
import contextlib
import io
import json
import logging
import os
import signal
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

from aiogram import Dispatcher, Router
from aiogram.types import Chat, Message, Update, User

from calendar_bot.__main__ import main
from calendar_bot.config import Config
from calendar_bot.health import check_health, health_path
from calendar_bot.locking import InstanceLock
from calendar_bot.polling import PollingMonitor
from calendar_bot.runtime import run, serve_polling
from calendar_bot.storage import migrate
from tests.polling_support import PRIVATE, TOKEN, PollingSession, RecordingNotifier


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "calendar.sqlite3"
        migrate(self.path)
        self.config = Config(self.path, TOKEN, "TEST_ONLY_NOT_A_REAL_KEY", frozenset())
        self.session = PollingSession()
        self.notifier = RecordingNotifier()
        self.parser = AsyncMock()
        self.enterContext(patch("calendar_bot.runtime.AiohttpSession", return_value=self.session))
        self.enterContext(patch("calendar_bot.runtime.OpenAIParser", return_value=self.parser))
        self.enterContext(
            patch("calendar_bot.runtime.SystemdNotifier.from_env", return_value=self.notifier)
        )
        self.enterContext(patch.object(logging.getLogger("aiogram"), "level", logging.CRITICAL))
        self.original_tasks = asyncio.all_tasks()

    def assert_finished(self):
        self.assertTrue(self.session.closed)
        self.assertEqual(self.session.active, 0)
        self.parser.close.assert_awaited_once()
        self.assertEqual(self.notifier.stops, 1)
        self.assertEqual(check_health(self.path).reason, "not_running")
        self.assertEqual(json.loads(health_path(self.path).read_text())["phase"], "stopped")
        remaining = asyncio.all_tasks() - self.original_tasks - {asyncio.current_task()}
        self.assertFalse(remaining, [task.get_coro().__qualname__ for task in remaining])

    async def test_sigterm_and_sigint_stop_requests_before_closing_connections(self):
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        task = asyncio.create_task(run(self.config))
        try:
            await asyncio.wait_for(self.session.requested.wait(), timeout=3)
            self.assertEqual(self.session.calls[0][1], 70)
            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.getsignal(signum)(signum, None)
            await asyncio.wait_for(task, timeout=3)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.assert_finished()
        for signum, handler in previous.items():
            self.assertEqual(signal.getsignal(signum), handler)

    async def test_worker_failure_stops_polling_and_propagates_safely(self):
        session = self.session

        class BrokenWorker:
            def __init__(self, *args):
                pass

            async def run(self, stop):
                await session.requested.wait()
                raise ValueError(PRIVATE)

        with patch("calendar_bot.runtime.NotificationWorker", BrokenWorker):
            with self.assertLogs("calendar_bot.runtime", level="ERROR") as captured:
                with self.assertRaises(ValueError):
                    await asyncio.wait_for(run(self.config), timeout=3)
        self.assertIn("type=ValueError", "\n".join(captured.output))
        self.assertNotIn(PRIVATE, "\n".join(captured.output))
        self.assert_finished()

    async def test_unexpected_worker_return_is_failure(self):
        worker = AsyncMock()
        with patch("calendar_bot.runtime.NotificationWorker", return_value=worker):
            with self.assertLogs("calendar_bot.runtime", level="ERROR"):
                with self.assertRaises(RuntimeError):
                    await asyncio.wait_for(run(self.config), timeout=3)
        self.assert_finished()

    async def test_unexpected_polling_return_is_failure(self):
        with patch.object(Dispatcher, "_polling", new=AsyncMock()):
            with self.assertLogs("calendar_bot.runtime", level="ERROR"):
                with self.assertRaises(RuntimeError):
                    await asyncio.wait_for(run(self.config), timeout=3)
        self.assert_finished()

    async def test_cancelling_runtime_does_not_orphan_aiogram_tasks(self):
        task = asyncio.create_task(run(self.config))
        await asyncio.wait_for(self.session.requested.wait(), timeout=3)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assert_finished()

    async def test_shutdown_drains_received_handler_before_session_close(self):
        from aiogram import Bot

        entered, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        router = Router()

        @router.message()
        async def handler(message):
            entered.set()
            await release.wait()
            completed.set()

        dispatcher = Dispatcher()
        dispatcher.include_router(router)
        bot = Bot(TOKEN, session=self.session)
        self.session.replies.put_nowait(
            [
                Update(
                    update_id=1,
                    message=Message(
                        message_id=1,
                        date=datetime(2026, 9, 24, tzinfo=UTC),
                        chat=Chat(id=101, type="private"),
                        from_user=User(id=101, is_bot=False, first_name="Synthetic"),
                        text="test",
                    ),
                )
            ]
        )
        session = self.session

        class UI:
            async def shutdown(self):
                if session.active:
                    raise AssertionError("Handlers drained before polling stopped")
                release.set()
                await completed.wait()

        class Worker:
            async def run(self, stop):
                await stop.wait()

        with InstanceLock(self.path) as lock:
            monitor = PollingMonitor(self.path, lock.instance_id)
            self.session.middleware(monitor)
            task = asyncio.create_task(
                serve_polling(
                    dispatcher, bot, UI(), Worker(), monitor, lambda: monitor.set_phase("stopping")
                )
            )
            try:
                await asyncio.wait_for(entered.wait(), timeout=3)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(completed.is_set())
                await bot.session.close()
                self.assertEqual(self.session.active, 0)
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_startup_failure_releases_lock_and_closes_resources(self):
        with patch("calendar_bot.runtime.Store.open", side_effect=ValueError(PRIVATE)):
            with self.assertRaises(ValueError):
                await run(self.config)
        self.assertTrue(self.session.closed)
        self.assertEqual(check_health(self.path).reason, "not_running")
        self.assertEqual(json.loads(health_path(self.path).read_text())["phase"], "stopped")
        self.assertEqual(self.notifier.stops, 1)


class RuntimeExitTests(unittest.TestCase):
    def test_failure_exits_nonzero_without_printing_provider_body(self):
        with tempfile.TemporaryDirectory() as directory:
            env = {
                "DATABASE_PATH": str(Path(directory) / "synthetic.sqlite3"),
                "BOT_TOKEN": TOKEN,
                "OPENAI_API_KEY": "TEST_ONLY_NOT_A_REAL_KEY",
            }
            with patch.dict(os.environ, env, clear=True):
                with patch(
                    "sys.argv",
                    ["calendar_bot", "run", "--env-file", str(Path(directory) / "absent.env")],
                ):
                    with patch(
                        "calendar_bot.runtime.run",
                        new=AsyncMock(side_effect=RuntimeError(PRIVATE)),
                    ):
                        output = io.StringIO()
                        with contextlib.redirect_stdout(output):
                            with self.assertLogs(
                                "calendar_bot.__main__", level="ERROR"
                            ) as captured:
                                self.assertEqual(main(), 1)
            self.assertNotIn(PRIVATE, output.getvalue() + "\n".join(captured.output))
