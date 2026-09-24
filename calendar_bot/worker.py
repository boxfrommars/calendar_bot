import asyncio
import json
import logging
from datetime import UTC, date, datetime, time, timedelta

from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
)

from . import presentation as view
from .domain import EventSpec, local_instant, zone
from .service import CalendarService

log = logging.getLogger(__name__)


class NotificationWorker:
    def __init__(self, service: CalendarService, bot):
        self.service, self.bot = service, bot
        self.next_chat: dict[int, datetime] = {}

    async def _prepare(self, tx, notice, now):
        user = await tx.one("SELECT * FROM users WHERE id=?", (notice["user_id"],))
        if user["id"] not in self.service.allowed or user["blocked"] or not user["timezone"]:
            await tx.execute(
                "UPDATE notifications SET status='cancelled' WHERE id=?", (notice["id"],)
            )
            return None
        if notice["kind"] == "reminder":
            event = await tx.one("SELECT * FROM events WHERE id=?", (notice["event_id"],))
            original_day = notice["occurrence_key"].split(":", 1)[1]
            occurrence = await tx.one(
                "SELECT * FROM occurrences WHERE event_id=? AND original_day=?",
                (notice["event_id"], original_day),
            )
            if (
                not event
                or not event["active"]
                or event["version"] != notice["version"]
                or not occurrence
                or occurrence["start_at"] != notice["start_at"]
            ):
                await tx.execute(
                    "UPDATE notifications SET status='cancelled' WHERE id=?", (notice["id"],)
                )
                return None
            # Coalesce overdue offsets of this occurrence, including a failed older attempt.
            due = await tx.all(
                "SELECT * FROM notifications WHERE user_id=? AND occurrence_key=? "
                "AND kind='reminder' AND status='pending' AND due_at<=? ORDER BY due_at DESC",
                (user["id"], notice["occurrence_key"], now.timestamp()),
            )
            if not due:
                return None
            notice = due[0]
            if notice["next_attempt_at"] > now.timestamp():
                # Respect Retry-After even if an older offset became eligible first.
                await tx.execute(
                    "UPDATE notifications SET next_attempt_at=? WHERE user_id=? "
                    "AND occurrence_key=? AND status='pending' AND due_at<=?",
                    (
                        notice["next_attempt_at"],
                        user["id"],
                        notice["occurrence_key"],
                        now.timestamp(),
                    ),
                )
                return None
            for obsolete in due[1:]:
                await tx.execute(
                    "UPDATE notifications SET status='skipped' WHERE id=?", (obsolete["id"],)
                )
            # An explicitly requested on-start reminder has one minute of dispatch grace.
            expires = notice["start_at"] + (60 if notice["offset_minutes"] == 0 else 0)
            if now.timestamp() >= expires:
                await tx.execute(
                    "UPDATE notifications SET status='skipped' WHERE id=?", (notice["id"],)
                )
                return None
            spec = EventSpec.from_json(occurrence["spec"])
            start = datetime.fromtimestamp(occurrence["start_at"], UTC)
            parts = [view.reminder_part(spec, start, now, user["timezone"])]
        else:
            day = date.fromisoformat(notice["occurrence_key"])
            if (
                not user["summary_enabled"]
                or notice["version"] != user["version"]
                or now.timestamp() >= notice["start_at"]
                or day != now.astimezone(zone(user["timezone"])).date()
            ):
                await tx.execute(
                    "UPDATE notifications SET status='skipped' WHERE id=?", (notice["id"],)
                )
                return None
            start = local_instant(day, time(), user["timezone"])
            end = local_instant(day + timedelta(days=1), time(), user["timezone"])
            rows = await tx.all(
                "SELECT o.* FROM occurrences o JOIN events e ON e.id=o.event_id "
                "WHERE e.user_id=? AND o.start_at>=? AND o.start_at<? ORDER BY o.start_at,o.event_id",
                (user["id"], start.timestamp(), end.timestamp()),
            )
            if not rows and not user["summary_empty"]:
                await tx.execute(
                    "UPDATE notifications SET status='skipped' WHERE id=?", (notice["id"],)
                )
                return None
            parts = view.summary_parts(rows, day, user["timezone"], now)
        if notice["parts"] is None or notice["kind"] == "reminder":
            await tx.execute(
                "UPDATE notifications SET parts=? WHERE id=?",
                (json.dumps(parts, ensure_ascii=False), notice["id"]),
            )
        else:
            parts = json.loads(notice["parts"])
        return notice, user, parts

    async def dispatch_one(self) -> bool:
        # The gate linearizes sends with edits/cancellations without holding SQLite open
        # across a network request. A crash after Telegram accepts can still cause a retry.
        async with self.service.gate:
            now = self.service.clock()
            async with self.service.store.transaction() as tx:
                pause = await tx.one(
                    "SELECT value FROM runtime_state WHERE key='telegram_pause_until'"
                )
                if pause and pause["value"] > now.timestamp():
                    return False
                # A long outage must not make an expired backlog delay current reminders.
                await tx.execute(
                    "UPDATE notifications SET status='skipped' WHERE status='pending' AND "
                    "((kind='summary' AND start_at<=?) OR (kind='reminder' AND "
                    "start_at + CASE WHEN offset_minutes=0 THEN 60 ELSE 0 END <=?))",
                    (now.timestamp(), now.timestamp()),
                )
                notice = await tx.one(
                    "SELECT * FROM notifications WHERE status='pending' AND due_at<=? "
                    "AND next_attempt_at<=? ORDER BY due_at,id LIMIT 1",
                    (now.timestamp(), now.timestamp()),
                )
                if notice is None:
                    return False
                prepared = await self._prepare(tx, notice, now)
                if prepared is None:
                    return True
                notice, user, parts = prepared
                if self.next_chat.get(user["id"], now) > now:
                    await tx.execute(
                        "UPDATE notifications SET next_attempt_at=? WHERE id=?",
                        (self.next_chat[user["id"]].timestamp(), notice["id"]),
                    )
                    return True
            try:
                message = await self.bot.send_message(
                    chat_id=user["chat_id"],
                    **view.notification_message(
                        notice["kind"], parts[notice["part_index"]]
                    ).as_kwargs(),
                    request_timeout=15,
                )
            except TelegramForbiddenError:
                async with self.service.store.transaction() as tx:
                    await tx.execute("UPDATE users SET blocked=1 WHERE id=?", (user["id"],))
                    await tx.execute(
                        "UPDATE notifications SET status='cancelled' WHERE user_id=? AND status='pending'",
                        (user["id"],),
                    )
                log.warning("telegram_blocked user_id=%s", user["id"])
            except TelegramRetryAfter as exc:
                await self._retry(notice, float(exc.retry_after), global_pause=True)
            except (TelegramNetworkError, TelegramServerError, TimeoutError):
                await self._retry(notice, min(300, 5 * 2 ** min(notice["attempts"], 6)))
            except TelegramBadRequest:
                async with self.service.store.transaction() as tx:
                    await tx.execute(
                        "UPDATE notifications SET status='failed' WHERE id=?", (notice["id"],)
                    )
                log.error("telegram_permanent_error notification_id=%s", notice["id"])
            else:
                next_part = notice["part_index"] + 1
                status = "sent" if next_part >= len(parts) else "pending"
                async with self.service.store.transaction() as tx:
                    await tx.execute(
                        "UPDATE notifications SET status=?,part_index=?,message_id=?,attempts=0 WHERE id=?",
                        (status, next_part, message.message_id, notice["id"]),
                    )
                log.info(
                    "notification_delivered id=%s kind=%s part=%s",
                    notice["id"],
                    notice["kind"],
                    next_part,
                )
            self.next_chat[user["id"]] = self.service.clock() + timedelta(seconds=1.05)
            return True

    async def _retry(self, notice, seconds, *, global_pause=False):
        async with self.service.store.transaction() as tx:
            deadline = (self.service.clock() + timedelta(seconds=max(1, seconds))).timestamp()
            await tx.execute(
                "UPDATE notifications SET attempts=attempts+1,next_attempt_at=? WHERE id=?",
                (deadline, notice["id"]),
            )
            if global_pause:
                await tx.execute(
                    "INSERT INTO runtime_state VALUES('telegram_pause_until',?) ON CONFLICT(key) "
                    "DO UPDATE SET value=MAX(runtime_state.value,excluded.value)",
                    (deadline,),
                )
        log.warning("notification_retry id=%s delay=%s", notice["id"], seconds)

    async def run(self, stop: asyncio.Event) -> None:
        next_refill = self.service.clock()
        while not stop.is_set():
            if self.service.clock() >= next_refill:
                await self.service.refresh()
                next_refill = self.service.clock() + timedelta(hours=1)
            for _ in range(20):
                if stop.is_set() or not await self.dispatch_one():
                    break
            try:
                await asyncio.wait_for(stop.wait(), timeout=5)
            except TimeoutError:
                pass
