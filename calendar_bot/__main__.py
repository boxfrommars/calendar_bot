import argparse
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from .config import Config, ConfigError, database_path_from_env
from .domain import UserError
from .locking import InstanceLock
from .logging_config import configure_logging

log = logging.getLogger(__name__)


async def check_telegram(config: Config) -> None:
    from aiogram import Bot
    from aiogram.exceptions import TelegramAPIError

    try:
        async with Bot(config.bot_token) as bot:
            await bot.get_me(request_timeout=15)
    except TelegramAPIError:
        raise UserError("Проверка Telegram не прошла. Проверьте токен и сетевой доступ.") from None


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
            from .health import check_health

            result = check_health(database_path_from_env(os.environ))
            print(result.describe())
            return 0 if result.healthy else 1
        config = Config.from_env(os.environ, require_secrets=args.command != "migrate")
        if args.command == "migrate":
            from .storage import SCHEMA_VERSION, migrate

            with InstanceLock(config.database_path):
                migrate(config.database_path)
            print(f"Схема SQLite готова (версия {SCHEMA_VERSION}).")
        elif args.command == "check":
            from .storage import check_database

            check_database(config.database_path)
            if not args.offline:
                import asyncio

                asyncio.run(check_telegram(config))
            print(
                "Проверка пройдена."
                + (" Telegram не проверялся (--offline)." if args.offline else "")
            )
        else:
            import asyncio

            from .runtime import run

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
