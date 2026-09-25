import asyncio
import logging
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramConflictError, TelegramNetworkError, TelegramServerError
from aiogram.methods import GetMe, GetUpdates
from aiogram.utils.backoff import BackoffConfig

from calendar_bot.health import check_health
from calendar_bot.locking import InstanceLock
from calendar_bot.logging_config import configure_logging
from calendar_bot.polling import PollingMonitor
from tests.polling_support import PRIVATE, TOKEN, MonotonicClock, PollingSession, RecordingNotifier


class PollingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "calendar.sqlite3"
        self.lock = self.enterContext(InstanceLock(self.path))
        self.clock = MonotonicClock()
        self.monitor = PollingMonitor(self.path, self.lock.instance_id, clock=self.clock)
        self.monitor.set_phase("polling")

    async def attempt(self, result=None, error=None):
        request = AsyncMock(return_value=[] if result is None else result, side_effect=error)
        return await self.monitor(request, None, GetUpdates())

    async def test_safe_errors_throttling_and_empty_response_recovery(self):
        with self.assertLogs("calendar_bot.polling", level="INFO") as captured:
            for delay, kind in (
                (0, TelegramConflictError),
                (1, TelegramConflictError),
                (59, TelegramConflictError),
                (1, TelegramNetworkError),
                (1, TelegramServerError),
            ):
                self.clock.advance(delay)
                with self.assertRaises(kind):
                    await self.attempt(error=kind(method=GetUpdates(), message=PRIVATE))
            self.assertIsNone(self.monitor.last_success)
            self.assertEqual(self.monitor.failures, 5)
            await self.attempt()
        messages = "\n".join(captured.output)
        self.assertEqual(messages.count("polling_request_failed"), 4)
        for name in ("TelegramConflictError", "TelegramNetworkError", "TelegramServerError"):
            self.assertIn(name, messages)
        self.assertIn("polling_recovered failures=5", messages)
        self.assertNotIn(PRIVATE, messages)
        self.assertNotIn(TOKEN, messages)
        self.assertTrue(all(record.exc_info is None for record in captured.records))
        self.assertEqual(self.monitor.last_success, self.clock())
        self.assertEqual(self.monitor.failures, 0)
        self.assertIsNone(self.monitor.last_error_type)

    async def test_cancellation_and_other_methods_are_not_polling_failures(self):
        await self.monitor(AsyncMock(return_value="me"), None, GetMe())
        self.assertIsNone(self.monitor.last_started)
        with self.assertNoLogs("calendar_bot.polling", level="WARNING"):
            with self.assertRaises(asyncio.CancelledError):
                await self.attempt(error=asyncio.CancelledError())
        self.assertEqual(self.monitor.failures, 0)
        self.assertIsNone(self.monitor.last_completed)
        self.assertIsNone(self.monitor.last_success)

    async def test_long_outage_degrades_health_but_preserves_watchdog_progress(self):
        await self.attempt()
        with self.assertLogs("calendar_bot.polling", level="ERROR"):
            for _ in range(20):
                # A failed request may take 70 s plus aiogram backoff before retrying.
                self.clock.advance(75)
                with self.assertRaises(TelegramNetworkError):
                    await self.attempt(
                        error=TelegramNetworkError(method=GetUpdates(), message=PRIVATE)
                    )
                self.assertTrue(self.monitor.progressing())
                self.monitor.publish()
        result = check_health(self.path, clock=self.clock)
        self.assertEqual(result.reason, "last_success_stale")
        self.assertEqual(result.error_type, "TelegramNetworkError")
        await self.attempt()
        self.monitor.publish()
        self.assertTrue(check_health(self.path, clock=self.clock).healthy)

    async def test_hung_request_stops_pings_and_logs_once(self):
        notifier = RecordingNotifier()
        stop = asyncio.Event()
        request_started = asyncio.Event()

        async def hung(bot, method):
            request_started.set()
            await asyncio.Event().wait()

        request = asyncio.create_task(self.monitor(hung, None, GetUpdates()))
        observer = asyncio.create_task(self.monitor.run(notifier, stop))
        try:
            await request_started.wait()
            await notifier.pinged.wait()
            self.clock.advance(90)
            before = notifier.pings
            with self.assertLogs("calendar_bot.polling", level="ERROR") as captured:
                await asyncio.sleep(0.03)
            self.assertEqual(notifier.pings, before)
            self.assertEqual(len(captured.records), 1)
            self.assertIn("polling_stalled", captured.output[0])
        finally:
            stop.set()
            request.cancel()
            await asyncio.gather(request, observer, return_exceptions=True)

    async def test_no_heartbeat_when_event_loop_is_blocked(self):
        notifier = RecordingNotifier()
        stop = asyncio.Event()
        observer = asyncio.create_task(self.monitor.run(notifier, stop))
        try:
            await notifier.pinged.wait()
            before = notifier.pings
            # An independent watchdog would see this gap; a background pinger must not mask it.
            time.sleep(0.06)
            self.assertEqual(notifier.pings, before)
            notifier.pinged.clear()
            await asyncio.wait_for(notifier.pinged.wait(), timeout=1)
        finally:
            stop.set()
            await observer

    async def test_startup_grace_and_stopping(self):
        self.monitor.set_phase("starting")
        self.clock.advance(89)
        self.assertTrue(self.monitor.progressing())
        self.clock.advance(1)
        with self.assertLogs("calendar_bot.polling", level="ERROR"):
            self.assertFalse(self.monitor.progressing())
        self.monitor.set_phase("polling")
        self.assertTrue(self.monitor.progressing())
        self.monitor.set_phase("stopping")
        self.assertFalse(self.monitor.progressing())

    async def test_configured_logging_suppresses_raw_sdk_bodies(self):
        for name in ("httpx", "httpcore", "openai", "aiogram", "aiohttp"):
            logger = logging.getLogger(name)
            self.enterContext(patch.object(logger, "level", logger.level))
        configure_logging()
        with self.assertLogs(level="INFO") as captured:
            logging.getLogger("aiogram.dispatcher").error("provider_error %s", PRIVATE)
            logging.getLogger("calendar_bot.polling").error(
                "polling_request_failed type=TelegramConflictError"
            )
        self.assertEqual(len(captured.records), 1)
        self.assertNotIn(PRIVATE, "\n".join(captured.output))

    async def test_reporter_keeps_pinging_during_network_outage(self):
        await self.attempt()
        notifier = RecordingNotifier()
        stop = asyncio.Event()
        reporter = asyncio.create_task(self.monitor.run(notifier, stop))
        try:
            await notifier.pinged.wait()
            with self.assertLogs("calendar_bot.polling", level="ERROR"):
                for _ in range(4):
                    self.clock.advance(75)
                    with self.assertRaises(TelegramNetworkError):
                        await self.attempt(
                            error=TelegramNetworkError(method=GetUpdates(), message=PRIVATE)
                        )
                    notifier.pinged.clear()
                    await asyncio.wait_for(notifier.pinged.wait(), timeout=1)
            self.assertGreaterEqual(notifier.pings, 5)
            self.assertEqual(check_health(self.path, clock=self.clock).reason, "last_success_stale")
        finally:
            stop.set()
            await reporter

    async def test_real_aiogram_retry_loop_is_observed_without_leaking_error_bodies(self):
        session = PollingSession()
        bot = Bot(TOKEN, session=session)
        session.middleware(self.monitor)
        session.replies.put_nowait(TelegramConflictError(method=GetUpdates(), message=PRIVATE))
        session.replies.put_nowait([])
        dispatcher = Dispatcher()
        with patch.object(logging.getLogger("aiogram"), "level", logging.CRITICAL):
            with self.assertLogs("calendar_bot.polling", level="INFO") as captured:
                polling = asyncio.create_task(
                    dispatcher.start_polling(
                        bot,
                        handle_signals=False,
                        close_bot_session=False,
                        backoff_config=BackoffConfig(
                            min_delay=0.01, max_delay=0.02, factor=1.1, jitter=0
                        ),
                    )
                )
                try:
                    async with asyncio.timeout(2):
                        while self.monitor.last_success is None:
                            await asyncio.sleep(0.005)
                finally:
                    await dispatcher.stop_polling()
                    await polling
                    await bot.session.close()
        self.assertGreaterEqual(len(session.calls), 2)
        self.assertIn("TelegramConflictError", "\n".join(captured.output))
        self.assertNotIn(PRIVATE, "\n".join(captured.output))
        self.assertIn("polling_recovered", "\n".join(captured.output))
