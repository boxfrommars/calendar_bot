import argparse
import asyncio
import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BotCommand
from dotenv import load_dotenv

from .config import Config, ConfigError, database_path_from_env
from .domain import UserError
from .locking import InstanceLock
from .logging_config import configure_logging
from .parser import OpenAIParser
from .polling import REQUEST_TIMEOUT, PollingMonitor, check_health
from .runtime import serve_polling, stop_signals
from .service import CalendarService
from .storage import Store, check_database, migrate
from .systemd import SystemdNotifier
from .telegram import BotUI
from .worker import NotificationWorker

log = logging.getLogger(__name__)


async def check_telegram(config: Config) -> None:
    try:
        async with Bot(config.bot_token) as bot:
            await bot.get_me(request_timeout=15)
    except TelegramAPIError:
        raise UserError("Проверка Telegram не прошла. Проверьте токен и сетевой доступ.") from None


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
        log.info("bot_started schema=1 allowed_users=%s", len(config.allowed_user_ids))
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


def main() -> int:
    configure_logging()
    arguments = argparse.ArgumentParser(description="Личный календарь в Telegram")
    arguments.add_argument("command", choices=["run", "migrate", "check", "health"])
    arguments.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Локальная конфигурация; переменные окружения имеют приоритет",
    )
    arguments.add_argument(
        "--offline", action="store_true", help="Для check: только БД и конфигурация, без Telegram"
    )
    args = arguments.parse_args()
    if args.offline and args.command != "check":
        arguments.error("--offline применяется только к check")
    load_dotenv(args.env_file, override=False)
    try:
        if args.command == "health":
            result = check_health(database_path_from_env(os.environ))
            print(result.describe())
            return 0 if result.healthy else 1
        config = Config.from_env(os.environ, require_secrets=args.command != "migrate")
        if args.command == "migrate":
            with InstanceLock(config.database_path):
                migrate(config.database_path)
            print("Схема SQLite готова (версия 1).")
        elif args.command == "check":
            check_database(config.database_path)
            if not args.offline:
                asyncio.run(check_telegram(config))
            print(
                "Проверка пройдена."
                + (" Telegram не проверялся (--offline)." if args.offline else "")
            )
        else:
            if args.offline:
                raise ConfigError("--offline применяется только к check.")
            asyncio.run(run(config))
    except (ConfigError, UserError) as exc:
        print(f"Ошибка: {exc}")
        return 1
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        # Do not print provider exception bodies or an ExceptionGroup with request data.
        log.error("fatal_error type=%s", type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
