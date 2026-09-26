from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEFAULT_OPENAI_MODEL = "gpt-6-luna"


class ConfigError(ValueError):
    pass


def database_path_from_env(env: Mapping[str, str]) -> Path:
    path = env.get("DATABASE_PATH", "data/calendar.sqlite3").strip()
    if not path or path == ":memory:":
        raise ConfigError("DATABASE_PATH должен указывать постоянный файл SQLite.")
    return Path(path).expanduser().resolve()


@dataclass(frozen=True)
class Config:
    database_path: Path
    bot_token: str
    openai_api_key: str
    allowed_user_ids: frozenset[int]
    openai_model: str = DEFAULT_OPENAI_MODEL
    openai_trace_dir: Path | None = None

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
        model = env.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
        if not model:
            raise ConfigError("OPENAI_MODEL не может быть пустым.")
        trace_dir = env.get("OPENAI_TRACE_DIR", "").strip()
        return cls(
            database_path_from_env(env),
            token,
            key,
            allowed,
            model,
            Path(trace_dir).expanduser().resolve() if trace_dir else None,
        )
