import asyncio
import sqlite3
from contextlib import asynccontextmanager, closing
from pathlib import Path

import aiosqlite

from .domain import UserError

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL, timezone TEXT,
    reminders TEXT NOT NULL DEFAULT '[15,5,1]',
    summary_time TEXT NOT NULL DEFAULT '09:00', summary_enabled INTEGER NOT NULL DEFAULT 1,
    summary_empty INTEGER NOT NULL DEFAULT 0, blocked INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE events (
    id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
    spec TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
    active INTEGER NOT NULL DEFAULT 1, created_at REAL NOT NULL
);
CREATE INDEX events_owner ON events(user_id, active);
CREATE TABLE exceptions (
    event_id TEXT NOT NULL REFERENCES events(id), original_day TEXT NOT NULL,
    spec TEXT, PRIMARY KEY(event_id, original_day)
);
CREATE TABLE occurrences (
    event_id TEXT NOT NULL REFERENCES events(id), original_day TEXT NOT NULL,
    start_at REAL NOT NULL, spec TEXT NOT NULL,
    PRIMARY KEY(event_id, original_day)
);
CREATE INDEX occurrence_time ON occurrences(start_at);
CREATE TABLE drafts (
    id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), spec TEXT NOT NULL,
    anchor_at REAL NOT NULL, expires_at REAL NOT NULL, version INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'pending', event_id TEXT REFERENCES events(id),
    original_day TEXT, event_version INTEGER, result_id TEXT
);
CREATE TABLE requests (
    source_key TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
    draft_ids TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE conversations (
    user_id INTEGER PRIMARY KEY REFERENCES users(id), mode TEXT NOT NULL,
    payload TEXT NOT NULL, expires_at REAL NOT NULL
);
CREATE TABLE processed_updates (id TEXT PRIMARY KEY, processed_at REAL NOT NULL);
CREATE TABLE runtime_state (key TEXT PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id),
    event_id TEXT REFERENCES events(id), occurrence_key TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('reminder','summary')), offset_minutes INTEGER NOT NULL,
    due_at REAL NOT NULL, start_at REAL NOT NULL, version INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending','sent','skipped','cancelled','failed')),
    attempts INTEGER NOT NULL DEFAULT 0, next_attempt_at REAL NOT NULL,
    parts TEXT, part_index INTEGER NOT NULL DEFAULT 0, message_id INTEGER,
    UNIQUE(user_id, kind, occurrence_key, offset_minutes)
);
CREATE INDEX notification_due ON notifications(status, next_attempt_at, due_at);
"""


def migrate(path: Path) -> None:
    """Only an explicit CLI invocation may create or migrate persistent state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            return
        if version != 0:
            raise UserError("Неизвестная версия схемы БД. Автоматический откат запрещён.")
        if db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchone():
            raise UserError("База без версии уже содержит таблицы; миграция остановлена.")
        db.executescript(
            "BEGIN EXCLUSIVE;\n" + SCHEMA + f"\nPRAGMA user_version={SCHEMA_VERSION};\nCOMMIT;"
        )
        db.execute("PRAGMA journal_mode=WAL")


def check_database(path: Path) -> None:
    if not path.is_file():
        raise UserError("База не создана. Сначала выполните python -m calendar_bot migrate.")
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as db:
        if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA_VERSION:
            raise UserError("Версия схемы БД не соответствует приложению.")
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise UserError("Проверка целостности SQLite завершилась ошибкой.")
        if db.execute("PRAGMA foreign_key_check").fetchone():
            raise UserError("Нарушена ссылочная целостность SQLite.")


class Transaction:
    def __init__(self, db: aiosqlite.Connection):
        self.db = db

    async def execute(self, sql: str, parameters: tuple = ()) -> int:
        async with self.db.execute(sql, parameters) as cursor:
            return cursor.rowcount

    async def one(self, sql: str, parameters: tuple = ()) -> dict | None:
        async with self.db.execute(sql, parameters) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def all(self, sql: str, parameters: tuple = ()) -> list[dict]:
        async with self.db.execute(sql, parameters) as cursor:
            return [dict(row) for row in await cursor.fetchall()]


class Store:
    def __init__(self, connection: aiosqlite.Connection):
        self.connection = connection
        self.lock = asyncio.Lock()

    @classmethod
    async def open(cls, path: Path) -> "Store":
        check_database(path)
        db = await aiosqlite.connect(path.as_uri() + "?mode=rw", uri=True)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys=ON")
        await db.execute("PRAGMA busy_timeout=5000")
        await db.execute("PRAGMA journal_mode=WAL")
        return cls(db)

    @asynccontextmanager
    async def transaction(self):
        # A single connection must not interleave independent transactions.
        async with self.lock:
            await self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield Transaction(self.connection)
                await self.connection.commit()
            except BaseException:
                await self.connection.rollback()
                raise

    async def close(self) -> None:
        await self.connection.close()
