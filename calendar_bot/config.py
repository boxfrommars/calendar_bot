from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Config:
    database_path: Path
    bot_token: str
    openai_api_key: str
    allowed_user_ids: frozenset[int]
    openai_model: str = "gpt-5.4-mini-2026-03-17"

    @classmethod
    def from_env(cls, env: Mapping[str, str], *, require_secrets: bool = True) -> "Config":
        token = env.get("BOT_TOKEN", "").strip()
        key = env.get("OPENAI_API_KEY", "").strip()
        try:
            allowed = frozenset(
                int(v.strip()) for v in env.get("ALLOWED_USER_IDS", "").split(",") if v.strip()
            )
        except ValueError:
            raise ConfigError("ALLOWED_USER_IDS должен содержать числовые Telegram ID.") from None
        if any(v <= 0 for v in allowed):
            raise ConfigError("Telegram ID должны быть положительными.")
        if require_secrets and (not token or not key):
            raise ConfigError("Задайте BOT_TOKEN и OPENAI_API_KEY.")
        model = env.get("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17").strip()
        if not model:
            raise ConfigError("OPENAI_MODEL не может быть пустым.")
        path = env.get("DATABASE_PATH", "data/calendar.sqlite3").strip()
        if not path or path == ":memory:":
            raise ConfigError("DATABASE_PATH должен указывать постоянный файл SQLite.")
        return cls(Path(path).expanduser().resolve(), token, key, allowed, model)
