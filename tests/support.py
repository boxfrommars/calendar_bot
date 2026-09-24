import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from calendar_bot.domain import EventSpec, zone
from calendar_bot.service import CalendarService
from calendar_bot.storage import Store, migrate


def entity_fragments(text, entities):
    """Decode Telegram's UTF-16 slices, rejecting out-of-bounds/split-surrogate entities."""
    encoded = text.encode("utf-16-le")
    result = []
    for entity in entities or []:
        start, end = entity.offset * 2, (entity.offset + entity.length) * 2
        if not 0 <= start < end <= len(encoded):
            raise AssertionError("Invalid entity bounds")
        result.append((entity.type, encoded[start:end].decode("utf-16-le")))
    return result


class Clock:
    def __init__(self):
        self.now = datetime(2026, 9, 24, 8, tzinfo=UTC)

    def __call__(self):
        return self.now

    def advance(self, **delta):
        self.now += timedelta(**delta)


class FakeBot:
    def __init__(self):
        self.sent = []
        self.errors = []

    async def send_message(self, **kwargs):
        if self.errors:
            raise self.errors.pop(0)
        self.sent.append(kwargs)
        return SimpleNamespace(message_id=len(self.sent))


class DatabaseCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "calendar.sqlite3"
        migrate(self.path)
        self.store = await Store.open(self.path)
        self.clock = Clock()
        self.service = CalendarService(self.store, frozenset({101, 202}), self.clock)
        for uid, tz in ((101, "Asia/Yerevan"), (202, "Europe/Moscow")):
            await self.service.ensure_user(uid, uid)
            await self.service.preferences(uid, timezone=tz, summary_enabled=0)
        self.sequence = 0

    async def asyncTearDown(self):
        await self.store.close()
        self.temporary.cleanup()

    def spec(
        self,
        *,
        when=None,
        title="Встреча",
        repeat="once",
        weekdays=(),
        offsets=None,
        timezone="Asia/Yerevan",
    ):
        when = when or self.clock() + timedelta(hours=1)
        local = when.astimezone(zone(timezone))
        return EventSpec(
            title,
            local.date(),
            local.time().replace(tzinfo=None, second=0, microsecond=0),
            timezone,
            repeat,
            weekdays,
            offsets,
        )

    async def create(self, *, uid=101, spec=None, **kwargs):
        self.sequence += 1
        drafts = await self.service.create_drafts(
            uid, f"test:{uid}:{self.sequence}", [spec or self.spec(**kwargs)], self.clock()
        )
        return await self.service.save_draft(uid, drafts[0]["id"], drafts[0]["version"])

    async def edit(self, event_id, spec, *, day=None, uid=101):
        self.sequence += 1
        event = await self.service.event(uid, event_id)
        drafts = await self.service.create_drafts(
            uid,
            f"edit:{self.sequence}",
            [spec],
            self.clock(),
            event_id=event_id,
            event_version=event["version"],
            original_day=day,
        )
        return await self.service.save_draft(uid, drafts[0]["id"], 1)

    async def rows(self, sql, parameters=()):
        async with self.store.transaction() as tx:
            return await tx.all(sql, parameters)

    async def pending(self, event_id):
        return await self.rows(
            "SELECT * FROM notifications WHERE event_id=? AND status='pending' ORDER BY due_at",
            (event_id,),
        )

    async def restart(self):
        await self.store.close()
        self.store = await Store.open(self.path)
        self.service = CalendarService(self.store, frozenset({101, 202}), self.clock)
