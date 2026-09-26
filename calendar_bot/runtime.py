"""Own polling tasks and stop them before handlers, SDKs and storage are closed."""

import asyncio
import logging
import os
import signal
from contextlib import contextmanager

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.types import BotCommand

from .config import Config
from .domain import UserError
from .locking import InstanceLock
from .parser import OpenAIParser
from .polling import POLLING_TIMEOUT, REQUEST_TIMEOUT, PollingMonitor
from .service import CalendarService
from .storage import SCHEMA_VERSION, Store
from .systemd import SystemdNotifier
from .telegram import BotUI
from .worker import NotificationWorker

log = logging.getLogger(__name__)
SHUTDOWN_TIMEOUT = 65


@contextmanager
def stop_signals(stop: asyncio.Event):
    loop = asyncio.get_running_loop()
    previous = {}

    def request_stop(signum, frame):
        loop.call_soon_threadsafe(stop.set)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request_stop)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


async def serve_polling(dispatcher, bot, ui, worker, monitor, begin_shutdown) -> None:
    ready = asyncio.Event()
    worker_stop = asyncio.Event()

    async def started(**kwargs):
        ready.set()

    dispatcher.startup.register(started)
    monitor.set_phase("polling")
    # Do not cancel start_polling: aiogram's own child tasks would survive that.
    polling = asyncio.create_task(
        dispatcher.start_polling(
            bot,
            polling_timeout=POLLING_TIMEOUT,
            allowed_updates=["message", "callback_query"],
            handle_signals=False,
            close_bot_session=False,
        ),
        name="calendar-polling",
    )
    delivery = asyncio.create_task(worker.run(worker_stop), name="calendar-delivery")
    try:
        done, _ = await asyncio.wait((polling, delivery), return_when=asyncio.FIRST_COMPLETED)
        for task, name in ((polling, "polling"), (delivery, "worker")):
            if task in done:
                if not task.cancelled() and (error := task.exception()) is not None:
                    log.error("runtime_task_failed task=%s type=%s", name, type(error).__name__)
                    raise error
                log.error("runtime_task_stopped task=%s", name)
                raise RuntimeError("Unexpected runtime task completion")
    finally:
        begin_shutdown()
        worker_stop.set()
        async with asyncio.timeout(SHUTDOWN_TIMEOUT):
            if not polling.done():
                # stop_polling is valid only after aiogram initialized its stop events.
                readiness = asyncio.create_task(ready.wait())
                try:
                    await asyncio.wait((readiness, polling), return_when=asyncio.FIRST_COMPLETED)
                finally:
                    readiness.cancel()
                    await asyncio.gather(readiness, return_exceptions=True)
                if not polling.done():
                    await dispatcher.stop_polling()
            await asyncio.gather(polling, return_exceptions=True)
            try:
                await ui.shutdown()
            finally:
                await asyncio.gather(delivery, return_exceptions=True)


async def _application(config: Config, monitor: PollingMonitor, begin_shutdown) -> None:
    bot = None
    store = None
    parser = None
    try:
        bot = Bot(config.bot_token, session=AiohttpSession(timeout=REQUEST_TIMEOUT))
        bot.session.middleware(monitor)
        store = await Store.open(config.database_path)
        parser = OpenAIParser(config.openai_api_key, config.openai_model)
        service = CalendarService(store, config.allowed_user_ids)
        dispatcher = Dispatcher()
        ui = BotUI(service, parser, bot)
        dispatcher.include_router(ui.router)
        # Refuse to replace an unexpected webhook or discard incoming updates.
        webhook = await bot.get_webhook_info(request_timeout=15)
        if webhook.url:
            raise UserError(
                "У токена настроен webhook. Используйте отдельный токен для этого polling-бота."
            )
        await bot.set_my_commands(
            [
                BotCommand(command="start", description="Начать работу / возобновить уведомления"),
                BotCommand(command="today", description="Сегодня"),
                BotCommand(command="week", description="Ближайшие 7 дней"),
                BotCommand(command="events", description="События и серии"),
                BotCommand(command="settings", description="Настройки"),
                BotCommand(command="help", description="Примеры и помощь"),
                BotCommand(command="cancel", description="Отменить текущий ввод"),
                BotCommand(command="id", description="Мой Telegram ID"),
            ],
            request_timeout=15,
        )
        worker = NotificationWorker(service, bot)
        log.info(
            "bot_started schema=%s allowed_users=%s", SCHEMA_VERSION, len(config.allowed_user_ids)
        )
        await serve_polling(dispatcher, bot, ui, worker, monitor, begin_shutdown)
    finally:
        begin_shutdown()
        # Polling, handlers and the worker have already finished before closing SDKs.
        async with asyncio.timeout(5):
            try:
                if parser is not None:
                    await parser.close()
            finally:
                try:
                    if bot is not None:
                        await bot.session.close()
                finally:
                    if store is not None:
                        await store.close()
        log.info("bot_stopped")


class _StopRequested(Exception):
    pass


async def run(config: Config) -> None:
    notifier = SystemdNotifier.from_env(os.environ)
    with InstanceLock(config.database_path) as instance:
        monitor = PollingMonitor(config.database_path, instance.instance_id)
        stop = asyncio.Event()
        monitor_stop = asyncio.Event()

        def begin_shutdown():
            if monitor.phase not in ("stopping", "stopped"):
                monitor.set_phase("stopping")
                notifier.stopping()

        async def observe():
            await monitor.run(notifier, monitor_stop)
            raise RuntimeError("Polling monitor stopped unexpectedly")

        async def stopping():
            await stop.wait()
            begin_shutdown()
            raise _StopRequested

        try:
            with stop_signals(stop):
                try:
                    async with asyncio.TaskGroup() as group:
                        group.create_task(observe(), name="calendar-monitor")
                        group.create_task(_application(config, monitor, begin_shutdown))
                        group.create_task(stopping())
                except* _StopRequested:
                    pass
        except ExceptionGroup as exc:
            # Preserve actionable, already-safe ConfigError/UserError messages on startup.
            error = exc
            while isinstance(error, ExceptionGroup) and len(error.exceptions) == 1:
                error = error.exceptions[0]
            raise error from None
        finally:
            monitor_stop.set()
            begin_shutdown()
            monitor.set_phase("stopped")
