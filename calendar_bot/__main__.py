import argparse
import asyncio
import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.exceptions import TelegramAPIError
from aiogram.types import BotCommand
from dotenv import load_dotenv

from .config import Config, ConfigError
from .domain import UserError
from .locking import InstanceLock
from .logging_config import configure_logging
from .parser import OpenAIParser
from .service import CalendarService
from .storage import Store, check_database, migrate
from .telegram import BotUI
from .worker import NotificationWorker

log = logging.getLogger(__name__)


async def check_telegram(config: Config) -> None:
    try:
        async with Bot(config.bot_token) as bot:
            await bot.get_me(request_timeout=15)
    except TelegramAPIError:
        raise UserError("Проверка Telegram не прошла. Проверьте токен и сетевой доступ.") from None


async def run(config: Config) -> None:
    with InstanceLock(config.database_path):
        bot = Bot(config.bot_token)
        store = None
        parser = None
        try:
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
                    BotCommand(
                        command="start", description="Начать работу / возобновить уведомления"
                    ),
                    BotCommand(command="today", description="Сегодня"),
                    BotCommand(command="week", description="Ближайшие 7 дней"),
                    BotCommand(command="events", description="События и серии"),
                    BotCommand(command="settings", description="Настройки"),
                    BotCommand(command="help", description="Примеры и помощь"),
                    BotCommand(command="cancel", description="Отменить текущий ввод"),
                    BotCommand(command="id", description="Мой Telegram ID"),
                ]
            )
            stop = asyncio.Event()
            worker = NotificationWorker(service, bot)

            async def poll():
                try:
                    await dispatcher.start_polling(
                        bot, allowed_updates=["message", "callback_query"], close_bot_session=False
                    )
                finally:
                    try:
                        await ui.shutdown()
                    finally:
                        stop.set()

            log.info("bot_started schema=1 allowed_users=%s", len(config.allowed_user_ids))
            # A failed worker must also stop polling, so a supervisor can restart both.
            async with asyncio.TaskGroup() as group:
                group.create_task(worker.run(stop))
                group.create_task(poll())
        finally:
            if parser is not None:
                await parser.close()
            await bot.session.close()
            if store is not None:
                await store.close()
            log.info("bot_stopped")


def main() -> int:
    configure_logging()
    arguments = argparse.ArgumentParser(description="Личный календарь в Telegram")
    arguments.add_argument("command", choices=["run", "migrate", "check"])
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
