import asyncio
import json
import uuid
from datetime import UTC, date, datetime, timedelta

from .domain import (
    AgendaPeriod,
    EventSpec,
    Occurrence,
    TaskSpec,
    UserError,
    expand,
    local_day,
    local_instant,
    parse_time,
    reminders,
    resolve_occurrence,
    utc_now,
    zone,
)
from .storage import Store, Transaction


def identifier() -> str:
    return uuid.uuid4().hex[:12]


class CalendarService:
    def __init__(self, store: Store, allowed: frozenset[int], clock=utc_now):
        self.store, self.allowed, self.clock = store, allowed, clock
        # Mutations and the final send share a gate: a confirmed cancellation cannot
        # race with a send prepared from an older revision. No DB tx spans network IO.
        self.gate = asyncio.Lock()

    def authorize(self, user_id: int) -> None:
        if user_id not in self.allowed:
            raise UserError("Доступ не разрешён.")

    async def ensure_user(self, user_id: int, chat_id: int, *, resume=False) -> dict:
        self.authorize(user_id)
        async with self.gate, self.store.transaction() as tx:
            await tx.execute(
                "INSERT INTO users(id,chat_id) VALUES(?,?) ON CONFLICT(id) "
                "DO UPDATE SET chat_id=excluded.chat_id",
                (user_id, chat_id),
            )
            if resume:
                await tx.execute("UPDATE users SET blocked=0 WHERE id=?", (user_id,))
            user = await tx.one("SELECT * FROM users WHERE id=?", (user_id,))
            if resume and user["timezone"]:
                await self._refresh_user(tx, user, self.clock())
            return user

    async def user(self, user_id: int) -> dict:
        self.authorize(user_id)
        async with self.store.transaction() as tx:
            user = await tx.one("SELECT * FROM users WHERE id=?", (user_id,))
        if not user:
            raise UserError("Сначала нажмите /start.")
        return user

    async def preferences(self, user_id: int, **changes) -> dict:
        self.authorize(user_id)
        if not changes or set(changes) - {
            "timezone",
            "reminders",
            "summary_time",
            "summary_enabled",
            "summary_empty",
        }:
            raise ValueError("Unknown preference")
        if "timezone" in changes:
            zone(changes["timezone"])
        if "reminders" in changes:
            changes["reminders"] = json.dumps(reminders(changes["reminders"]))
        if "summary_time" in changes:
            parse_time(changes["summary_time"])
        async with self.gate, self.store.transaction() as tx:
            columns = ", ".join(f"{key}=?" for key in changes)
            await tx.execute(
                f"UPDATE users SET {columns}, version=version+1 WHERE id=?",
                (*changes.values(), user_id),
            )
            user = await tx.one("SELECT * FROM users WHERE id=?", (user_id,))
            if not user:
                raise UserError("Сначала нажмите /start.")
            await self._refresh_user(tx, user, self.clock())
            return user

    async def conversation(self, user_id: int) -> dict | None:
        self.authorize(user_id)
        async with self.store.transaction() as tx:
            row = await tx.one(
                "SELECT * FROM conversations WHERE user_id=? AND expires_at>?",
                (user_id, self.clock().timestamp()),
            )
        if row:
            row["payload"] = json.loads(row["payload"])
        return row

    async def set_conversation(self, user_id: int, mode: str | None, payload=None) -> None:
        self.authorize(user_id)
        async with self.store.transaction() as tx:
            if mode is None:
                await tx.execute("DELETE FROM conversations WHERE user_id=?", (user_id,))
            else:
                await tx.execute(
                    "INSERT INTO conversations VALUES(?,?,?,?) ON CONFLICT(user_id) "
                    "DO UPDATE SET mode=excluded.mode,payload=excluded.payload,expires_at=excluded.expires_at",
                    (
                        user_id,
                        mode,
                        json.dumps(payload or {}, ensure_ascii=False),
                        (self.clock() + timedelta(days=1)).timestamp(),
                    ),
                )

    async def processed(self, key: str) -> bool:
        async with self.store.transaction() as tx:
            return await tx.one("SELECT id FROM processed_updates WHERE id=?", (key,)) is not None

    async def mark_processed(self, key: str) -> None:
        async with self.store.transaction() as tx:
            await tx.execute(
                "INSERT OR IGNORE INTO processed_updates VALUES(?,?)",
                (key, self.clock().timestamp()),
            )

    async def request_drafts(self, user_id: int, key: str) -> list[dict] | None:
        self.authorize(user_id)
        async with self.store.transaction() as tx:
            row = await tx.one(
                "SELECT draft_ids FROM requests WHERE source_key=? AND user_id=?", (key, user_id)
            )
            if row is None:
                return None
            return [
                await tx.one("SELECT * FROM drafts WHERE id=? AND user_id=?", (d, user_id))
                for d in json.loads(row["draft_ids"])
            ]

    async def create_drafts(
        self,
        user_id: int,
        key: str,
        specs: list[EventSpec | TaskSpec],
        anchor: datetime,
        *,
        event_id=None,
        original_day=None,
        event_version=None,
        task_id=None,
        task_version=None,
        draft_id=None,
        draft_version=None,
        auto_save=False,
    ) -> list[dict]:
        """Persist a parsed request; only fresh additions can opt into atomic auto-save."""
        self.authorize(user_id)
        if auto_save and (event_id or task_id or draft_id or original_day):
            raise ValueError("Only new items can be saved automatically")
        if event_id and task_id:
            raise ValueError("Only one edit target is allowed")
        if not 1 <= len(specs) <= 10 or ((event_id or task_id or draft_id) and len(specs) != 1):
            raise UserError(
                "При редактировании нужна одна запись; при добавлении — не более десяти."
            )
        async with self.gate, self.store.transaction() as tx:
            previous = await tx.one(
                "SELECT * FROM requests WHERE source_key=? AND user_id=?", (key, user_id)
            )
            if previous:
                return [
                    await tx.one("SELECT * FROM drafts WHERE id=? AND user_id=?", (d, user_id))
                    for d in json.loads(previous["draft_ids"])
                ]
            if event_id:
                target = await self._owned_event(tx, user_id, event_id, event_version)
                # A one-off has no series to make an exception to: moving it changes
                # its actual date, so it remains visible/editable after the old date.
                if (
                    EventSpec.from_json(target["spec"]).repeat == "once"
                    and not target["has_exceptions"]
                ):
                    original_day = None
            if task_id:
                if task_version is None:
                    raise UserError("Откройте свежую карточку дела через /week.")
                await self._owned_task(tx, user_id, task_id, task_version)
            now = self.clock()
            ids = []
            for spec in specs:
                kind = "task" if isinstance(spec, TaskSpec) else "event"
                if (event_id and kind != "event") or (task_id and kind != "task"):
                    raise UserError("Превращение дела в событие и наоборот пока не поддерживается.")
                if original_day and (kind != "event" or spec.repeat != "once"):
                    raise UserError(
                        "Отдельная встреча должна оставаться разовой. Для повторов измените серию."
                    )
                if (
                    isinstance(spec, EventSpec)
                    and spec.repeat == "once"
                    and spec.first_after(now) <= now
                ):
                    raise UserError("Это время уже прошло. Укажите будущую дату и время.")
                if draft_id:
                    old = await self._owned_draft(tx, user_id, draft_id, draft_version)
                    if old["status"] != "pending" or old["expires_at"] <= now.timestamp():
                        raise UserError(
                            "Этот черновик уже закрыт или истёк. Добавьте событие заново."
                        )
                    if old["kind"] != kind:
                        raise UserError("Нельзя менять тип записи при исправлении черновика.")
                    await tx.execute(
                        "UPDATE drafts SET spec=?,version=version+1 WHERE id=?",
                        (spec.to_json(), draft_id),
                    )
                    ids.append(draft_id)
                else:
                    new_id = identifier()
                    await tx.execute(
                        "INSERT INTO drafts(id,user_id,spec,anchor_at,expires_at,event_id,original_day,event_version,kind,task_id,task_version) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            new_id,
                            user_id,
                            spec.to_json(),
                            anchor.timestamp(),
                            (now + timedelta(days=1)).timestamp(),
                            event_id,
                            original_day,
                            event_version,
                            kind,
                            task_id,
                            task_version,
                        ),
                    )
                    ids.append(new_id)
            await tx.execute(
                "INSERT INTO requests VALUES(?,?,?,?)",
                (key, user_id, json.dumps(ids), now.timestamp()),
            )
            if auto_save:
                # The entire message, its deduplication record, and reminders commit together.
                for draft_id in ids:
                    draft = await self._owned_draft(tx, user_id, draft_id)
                    await self._save_draft(tx, user_id, draft, now)
            return [await tx.one("SELECT * FROM drafts WHERE id=?", (d,)) for d in ids]

    async def _owned_draft(
        self, tx: Transaction, user_id: int, draft_id: str, version=None
    ) -> dict:
        row = await tx.one("SELECT * FROM drafts WHERE id=? AND user_id=?", (draft_id, user_id))
        if not row:
            raise UserError("Черновик не найден.")
        if version is not None and row["version"] != version:
            raise UserError("Карточка устарела. Используйте её последнюю версию.")
        return row

    async def draft(self, user_id: int, draft_id: str, version=None) -> dict:
        self.authorize(user_id)
        async with self.store.transaction() as tx:
            return await self._owned_draft(tx, user_id, draft_id, version)

    async def _owned_event(
        self, tx: Transaction, user_id: int, event_id: str, version=None
    ) -> dict:
        row = await tx.one(
            "SELECT e.*,EXISTS(SELECT 1 FROM exceptions x WHERE x.event_id=e.id) AS has_exceptions "
            "FROM events e WHERE e.id=? AND e.user_id=? AND e.active=1",
            (event_id, user_id),
        )
        if not row:
            raise UserError("Событие не найдено или отменено.")
        if version is not None and row["version"] != version:
            raise UserError(
                "Расписание изменилось. Откройте свежую карточку через /events или /week."
            )
        return row

    async def event(self, user_id: int, event_id: str, version=None) -> dict:
        self.authorize(user_id)
        async with self.store.transaction() as tx:
            return await self._owned_event(tx, user_id, event_id, version)

    async def occurrence(
        self, user_id: int, event_id: str, original_day: str, version=None
    ) -> dict:
        self.authorize(user_id)
        async with self.store.transaction() as tx:
            await self._owned_event(tx, user_id, event_id, version)
            return await self._future_occurrence(tx, event_id, original_day, self.clock())

    async def _future_occurrence(self, tx, event_id, original_day, now):
        row = await tx.one(
            "SELECT * FROM occurrences WHERE event_id=? AND original_day=?",
            (event_id, original_day),
        )
        if row and row["start_at"] > now.timestamp():
            return row
        if not row:
            event = await tx.one("SELECT * FROM events WHERE id=? AND active=1", (event_id,))
            exception = await tx.one(
                "SELECT spec FROM exceptions WHERE event_id=? AND original_day=?",
                (event_id, original_day),
            )
            if event:
                day = date.fromisoformat(original_day)
                overrides = (
                    {day: EventSpec.from_json(exception["spec"]) if exception["spec"] else None}
                    if exception
                    else {}
                )
                occurrence = resolve_occurrence(
                    event_id, EventSpec.from_json(event["spec"]), overrides, day
                )
                if occurrence and occurrence.start > now:
                    return {
                        "event_id": event_id,
                        "original_day": original_day,
                        "start_at": occurrence.start.timestamp(),
                        "spec": occurrence.spec.to_json(),
                    }
        raise UserError("Эта встреча уже началась или была отменена. Откройте /week.")

    async def save_draft(self, user_id: int, draft_id: str, version: int) -> str:
        self.authorize(user_id)
        async with self.gate, self.store.transaction() as tx:
            draft = await self._owned_draft(tx, user_id, draft_id, version)
            return await self._save_draft(tx, user_id, draft, self.clock())

    async def _save_draft(self, tx: Transaction, user_id: int, draft: dict, now: datetime) -> str:
        if draft["status"] == "saved":
            return draft["result_id"]
        if draft["status"] != "pending" or draft["expires_at"] <= now.timestamp():
            raise UserError("Черновик отменён или истёк. Добавьте событие заново.")
        user = await tx.one("SELECT * FROM users WHERE id=?", (user_id,))
        if not user["timezone"]:
            raise UserError("Сначала выберите часовой пояс через /start.")
        if draft["kind"] == "task":
            return await self._save_task_draft(tx, user_id, draft, now)
        spec = EventSpec.from_json(draft["spec"])
        if spec.repeat == "once" and spec.first_after(now) <= now:
            raise UserError("Время события уже прошло. Нажмите «Исправить» и выберите новое.")
        if draft["event_id"]:
            event = await self._owned_event(tx, user_id, draft["event_id"], draft["event_version"])
            event_id = event["id"]
            if draft["original_day"]:
                await self._future_occurrence(tx, event_id, draft["original_day"], now)
                if spec.repeat != "once":
                    raise UserError("Отдельную встречу нельзя превратить в серию.")
                await tx.execute(
                    "INSERT INTO exceptions VALUES(?,?,?) ON CONFLICT(event_id,original_day) "
                    "DO UPDATE SET spec=excluded.spec",
                    (event_id, draft["original_day"], spec.to_json()),
                )
                await tx.execute("UPDATE events SET version=version+1 WHERE id=?", (event_id,))
            else:
                old = EventSpec.from_json(event["spec"])
                if old.repeat == "once" and old.first_after(now) <= now:
                    raise UserError("Начавшееся событие нельзя изменить. Создайте новое.")
                await tx.execute(
                    "UPDATE events SET spec=?,version=version+1 WHERE id=?",
                    (spec.to_json(), event_id),
                )
        else:
            event_id = identifier()
            await tx.execute(
                "INSERT INTO events(id,user_id,spec,created_at) VALUES(?,?,?,?)",
                (event_id, user_id, spec.to_json(), now.timestamp()),
            )
        await tx.execute(
            "UPDATE drafts SET status='saved',result_id=? WHERE id=?", (event_id, draft["id"])
        )
        event = await tx.one("SELECT * FROM events WHERE id=?", (event_id,))
        await self._rebuild_event(tx, event, user, now)
        return event_id

    async def _save_task_draft(self, tx, user_id, draft, now) -> str:
        spec = TaskSpec.from_json(draft["spec"])
        task_id = draft["task_id"]
        if task_id:
            await self._owned_task(tx, user_id, task_id, draft["task_version"])
            await tx.execute(
                "UPDATE tasks SET title=?,day=?,version=version+1 WHERE id=?",
                (spec.title, spec.day.isoformat(), task_id),
            )
        else:
            task_id = identifier()
            await tx.execute(
                "INSERT INTO tasks(id,user_id,title,day,created_at) VALUES(?,?,?,?,?)",
                (task_id, user_id, spec.title, spec.day.isoformat(), now.timestamp()),
            )
        await tx.execute(
            "UPDATE drafts SET status='saved',result_id=? WHERE id=?", (task_id, draft["id"])
        )
        await self._invalidate_task_summaries(tx, user_id)
        return task_id

    async def _owned_task(
        self, tx, user_id, task_id, version=None, *, include_deleted=False
    ) -> dict:
        row = await tx.one(
            "SELECT * FROM tasks WHERE id=? AND user_id=?"
            + ("" if include_deleted else " AND active=1"),
            (task_id, user_id),
        )
        if not row:
            raise UserError("Дело не найдено или удалено.")
        if version is not None and row["version"] != version:
            raise UserError("Дело изменилось. Используйте актуальную карточку или /week.")
        return row

    async def task(
        self, user_id: int, task_id: str, version=None, *, include_deleted=False
    ) -> dict:
        self.authorize(user_id)
        async with self.store.transaction() as tx:
            return await self._owned_task(
                tx, user_id, task_id, version, include_deleted=include_deleted
            )

    async def set_task_completion(
        self, user_id: int, task_id: str, version: int, completed: bool
    ) -> None:
        self.authorize(user_id)
        if type(version) is not int or version < 1:
            raise UserError("Откройте свежую карточку дела через /week.")
        if type(completed) is not bool:
            raise ValueError("Expected an explicit completion state")
        async with self.gate, self.store.transaction() as tx:
            row = await self._owned_task(tx, user_id, task_id, version)
            if (row["completed_at"] is not None) == completed:
                return
            await tx.execute(
                "UPDATE tasks SET completed_at=?,version=version+1 WHERE id=?",
                (self.clock().timestamp() if completed else None, task_id),
            )
            await self._invalidate_task_summaries(tx, user_id)

    async def cancel_task(self, user_id: int, task_id: str, version: int) -> None:
        self.authorize(user_id)
        if type(version) is not int or version < 1:
            raise UserError("Откройте свежую карточку дела через /week.")
        async with self.gate, self.store.transaction() as tx:
            await self._owned_task(tx, user_id, task_id, version)
            await tx.execute("UPDATE tasks SET active=0,version=version+1 WHERE id=?", (task_id,))
            await self._invalidate_task_summaries(tx, user_id)

    async def _invalidate_task_summaries(self, tx, user_id) -> None:
        # A started multipart summary is an immutable snapshot. Never rewind delivery.
        await tx.execute(
            "UPDATE notifications SET parts=NULL WHERE user_id=? AND kind='summary' "
            "AND status='pending' AND part_index=0",
            (user_id,),
        )

    async def task_counts(self, tx: Transaction, user_id: int, day: date) -> tuple[int, int]:
        self.authorize(user_id)
        row = await tx.one(
            "SELECT SUM(day=?) AS today,SUM(day<?) AS overdue FROM tasks "
            "WHERE user_id=? AND active=1 AND completed_at IS NULL AND day<=?",
            (day.isoformat(), day.isoformat(), user_id, day.isoformat()),
        )
        return row["today"] or 0, row["overdue"] or 0

    async def open_tasks_through(self, tx: Transaction, user_id: int, day: date) -> list[dict]:
        self.authorize(user_id)
        return await tx.all(
            "SELECT * FROM tasks WHERE user_id=? AND active=1 AND completed_at IS NULL "
            "AND day<=? ORDER BY day,created_at,id",
            (user_id, day.isoformat()),
        )

    async def cancel_draft(self, user_id: int, draft_id: str, version: int) -> None:
        self.authorize(user_id)
        async with self.gate, self.store.transaction() as tx:
            await self._owned_draft(tx, user_id, draft_id, version)
            await tx.execute(
                "UPDATE drafts SET status='cancelled' WHERE id=? AND status='pending'", (draft_id,)
            )

    async def cancel_event(
        self, user_id: int, event_id: str, version: int, original_day=None
    ) -> None:
        self.authorize(user_id)
        async with self.gate, self.store.transaction() as tx:
            event = await self._owned_event(tx, user_id, event_id, version)
            now = self.clock()
            if original_day and (
                EventSpec.from_json(event["spec"]).repeat != "once" or event["has_exceptions"]
            ):
                await self._future_occurrence(tx, event_id, original_day, now)
                await tx.execute(
                    "INSERT INTO exceptions VALUES(?,?,NULL) ON CONFLICT(event_id,original_day) "
                    "DO UPDATE SET spec=NULL",
                    (event_id, original_day),
                )
                await tx.execute("UPDATE events SET version=version+1 WHERE id=?", (event_id,))
            else:
                await tx.execute(
                    "UPDATE events SET active=0,version=version+1 WHERE id=?", (event_id,)
                )
            event = await tx.one("SELECT * FROM events WHERE id=?", (event_id,))
            user = await tx.one("SELECT * FROM users WHERE id=?", (user_id,))
            await self._rebuild_event(tx, event, user, now)

    async def _enqueue(
        self,
        tx: Transaction,
        *,
        user_id: int,
        event_id,
        key: str,
        kind: str,
        offset: int,
        due: float,
        start: float,
        version: int,
    ) -> None:
        await tx.execute(
            """INSERT INTO notifications
            (user_id,event_id,occurrence_key,kind,offset_minutes,due_at,start_at,version,next_attempt_at)
            VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id,kind,occurrence_key,offset_minutes) DO UPDATE SET
            status='pending', due_at=excluded.due_at, start_at=excluded.start_at,
            next_attempt_at=CASE WHEN notifications.version != excluded.version
                OR notifications.due_at != excluded.due_at THEN excluded.due_at
                ELSE MAX(notifications.next_attempt_at,excluded.due_at) END,
            parts=CASE WHEN notifications.part_index>0 THEN notifications.parts
                WHEN notifications.version != excluded.version THEN NULL ELSE notifications.parts END,
            part_index=CASE WHEN notifications.part_index>0 THEN notifications.part_index
                WHEN notifications.version != excluded.version THEN 0 ELSE notifications.part_index END,
            version=excluded.version
            WHERE notifications.status != 'sent' AND (notifications.status IN ('pending','cancelled')
                OR (notifications.version != excluded.version AND
                    (notifications.kind='summary' OR notifications.status='failed'
                     OR notifications.due_at != excluded.due_at
                     OR notifications.start_at != excluded.start_at)))""",
            (user_id, event_id, key, kind, offset, due, start, version, due),
        )

    async def _rebuild_event(self, tx: Transaction, event: dict, user: dict, now: datetime) -> None:
        timestamp = now.timestamp()
        await tx.execute(
            "DELETE FROM occurrences WHERE event_id=? AND start_at>?", (event["id"], timestamp)
        )
        await tx.execute(
            "UPDATE notifications SET status='cancelled' WHERE event_id=? AND status='pending' AND start_at>?",
            (event["id"], timestamp),
        )
        if not event["active"] or user["blocked"] or user["id"] not in self.allowed:
            return
        spec = EventSpec.from_json(event["spec"])
        rows = await tx.all("SELECT * FROM exceptions WHERE event_id=?", (event["id"],))
        exceptions = {
            date.fromisoformat(row["original_day"]): EventSpec.from_json(row["spec"])
            if row["spec"]
            else None
            for row in rows
        }
        past = await tx.all("SELECT original_day FROM occurrences WHERE event_id=?", (event["id"],))
        protected = {row["original_day"] for row in past}
        horizon = now + timedelta(days=30)
        desired = expand(event["id"], spec, exceptions, now, horizon)
        # Explicit single dates and exceptions need no rolling expansion. Persist them
        # immediately, so a move beyond the rolling horizon is still editable.
        if spec.repeat == "once" and spec.day not in exceptions:
            instant = spec.first_after(now)
            if instant >= horizon:
                desired.append(Occurrence(event["id"], spec.day, instant, spec))
        for original_day, override in exceptions.items():
            if override is not None:
                instant = override.first_after(now)
                if instant >= horizon:
                    desired.append(Occurrence(event["id"], original_day, instant, override))
        for occurrence in desired:
            day = occurrence.original_day.isoformat()
            if day in protected:
                continue
            await tx.execute(
                "INSERT INTO occurrences VALUES(?,?,?,?)",
                (event["id"], day, occurrence.start.timestamp(), occurrence.spec.to_json()),
            )
            offsets = occurrence.spec.reminder_minutes
            if offsets is None:
                offsets = json.loads(user["reminders"])
            for offset in offsets:
                await self._enqueue(
                    tx,
                    user_id=user["id"],
                    event_id=event["id"],
                    key=occurrence.key,
                    kind="reminder",
                    offset=offset,
                    due=(occurrence.start - timedelta(minutes=offset)).timestamp(),
                    start=occurrence.start.timestamp(),
                    version=event["version"],
                )

    async def _summary(self, tx: Transaction, user: dict, now: datetime) -> None:
        await tx.execute(
            "UPDATE notifications SET status='cancelled' WHERE user_id=? AND kind='summary' AND status='pending'",
            (user["id"],),
        )
        if (
            not user["timezone"]
            or not user["summary_enabled"]
            or user["blocked"]
            or user["id"] not in self.allowed
        ):
            return
        day = now.astimezone(zone(user["timezone"])).date()
        # Tomorrow is prepared as well, so midnight does not depend on the hourly refill.
        for selected in (day, day + timedelta(days=1)):
            due = local_instant(selected, parse_time(user["summary_time"]), user["timezone"])
            await self._enqueue(
                tx,
                user_id=user["id"],
                event_id=None,
                key=selected.isoformat(),
                kind="summary",
                offset=-1,
                due=due.timestamp(),
                start=(due + timedelta(hours=2)).timestamp(),
                version=user["version"],
            )

    async def _refresh_user(self, tx: Transaction, user: dict, now: datetime) -> None:
        for event in await tx.all(
            "SELECT * FROM events WHERE user_id=? AND active=1", (user["id"],)
        ):
            await self._rebuild_event(tx, event, user, now)
        await self._summary(tx, user, now)

    async def refresh(self) -> None:
        async with self.gate, self.store.transaction() as tx:
            now = self.clock()
            for user in await tx.all("SELECT * FROM users"):
                await self._refresh_user(tx, user, now)
            await tx.execute(
                "DELETE FROM processed_updates WHERE processed_at<?",
                ((now - timedelta(days=7)).timestamp(),),
            )
            await tx.execute("DELETE FROM conversations WHERE expires_at<?", (now.timestamp(),))

    async def agenda(
        self, user_id: int, start: datetime, end: datetime, *, limit=100, offset=0
    ) -> list[dict]:
        self.authorize(user_id)
        async with self.store.transaction() as tx:
            return await tx.all(
                "SELECT o.*,e.version,e.active FROM occurrences o JOIN events e ON e.id=o.event_id "
                "WHERE e.user_id=? AND o.start_at>=? AND o.start_at<? ORDER BY o.start_at,o.event_id "
                "LIMIT ? OFFSET ?",
                (user_id, start.timestamp(), end.timestamp(), limit, offset),
            )

    async def _plan_events(self, tx, user_id, start, end, now) -> list[dict]:
        # History is a stored snapshot. Only future occurrences use the current series.
        rows = await tx.all(
            "SELECT o.*,e.version,e.active,'event' AS kind,0 AS overdue_group "
            "FROM occurrences o JOIN events e ON e.id=o.event_id "
            "WHERE e.user_id=? AND o.start_at>=? AND o.start_at<? AND o.start_at<=?",
            (user_id, start.timestamp(), end.timestamp(), now.timestamp()),
        )
        if end <= now:
            return rows
        events = await tx.all("SELECT * FROM events WHERE user_id=? AND active=1", (user_id,))
        exceptions = await tx.all(
            "SELECT x.* FROM exceptions x JOIN events e ON e.id=x.event_id WHERE e.user_id=?",
            (user_id,),
        )
        past = await tx.all(
            "SELECT o.event_id,o.original_day FROM occurrences o JOIN events e ON e.id=o.event_id "
            "WHERE e.user_id=? AND o.start_at<=?",
            (user_id, now.timestamp()),
        )
        protected = {(row["event_id"], row["original_day"]) for row in past}
        overrides = {}
        for row in exceptions:
            overrides.setdefault(row["event_id"], {})[date.fromisoformat(row["original_day"])] = (
                EventSpec.from_json(row["spec"]) if row["spec"] else None
            )
        for event in events:
            for occurrence in expand(
                event["id"],
                EventSpec.from_json(event["spec"]),
                overrides.get(event["id"], {}),
                max(start, now),
                end,
            ):
                original_day = occurrence.original_day.isoformat()
                if occurrence.start <= now or (event["id"], original_day) in protected:
                    continue
                rows.append(
                    {
                        "event_id": event["id"],
                        "original_day": original_day,
                        "start_at": occurrence.start.timestamp(),
                        "spec": occurrence.spec.to_json(),
                        "version": event["version"],
                        "active": event["active"],
                        "kind": "event",
                        "overdue_group": False,
                    }
                )
        return rows

    async def plan_page(
        self, user_id: int, *, start_day: date | None = None, days=7, page=0
    ) -> tuple[list[dict], int, int]:
        self.authorize(user_id)
        if not 0 <= page <= 10000:
            raise UserError("Некорректная страница.")
        async with self.store.transaction() as tx:
            user = await tx.one("SELECT * FROM users WHERE id=?", (user_id,))
            if not user or not user["timezone"]:
                raise UserError("Сначала выберите часовой пояс через /start.")
            now = self.clock()
            today = local_day(now, user["timezone"])
            period = AgendaPeriod(start_day or today, days)
            include_overdue = period.contains(today)
            task_where = (
                "user_id=? AND active=1 AND ((day>=? AND day<?) "
                "OR (? AND day<? AND completed_at IS NULL))"
            )
            parameters = (
                user_id,
                period.start.isoformat(),
                period.end.isoformat(),
                include_overdue,
                today.isoformat(),
            )
            events = await self._plan_events(tx, user_id, *period.window(user["timezone"]), now)
            total = (
                len(events)
                + (await tx.one("SELECT COUNT(*) AS n FROM tasks WHERE " + task_where, parameters))[
                    "n"
                ]
            )
            page = min(page, max(0, (total - 1) // 8))
            rows = await tx.all(
                "SELECT *, 'task' AS kind, (? AND day<? AND completed_at IS NULL) AS overdue_group "
                "FROM tasks WHERE "
                + task_where
                + " ORDER BY overdue_group DESC,day,completed_at IS NOT NULL,created_at,id LIMIT ?",
                (include_overdue, today.isoformat(), *parameters, (page + 1) * 8),
            )
            for event in events:
                event["day"] = local_day(
                    datetime.fromtimestamp(event["start_at"], UTC), user["timezone"]
                ).isoformat()
            rows.extend(events)
            rows.sort(
                key=lambda row: (
                    not row["overdue_group"],
                    row["day"],
                    0 if row["kind"] == "event" else 2 if row["completed_at"] is not None else 1,
                    row.get("start_at", row.get("created_at")),
                    row.get("event_id", row.get("id")),
                )
            )
            return rows[page * 8 : page * 8 + 8], page, total

    async def event_list(self, user_id: int, *, offset=0, limit=6) -> list[dict]:
        self.authorize(user_id)
        async with self.store.transaction() as tx:
            now = self.clock()
            rows = await tx.all(
                "SELECT e.*,EXISTS(SELECT 1 FROM occurrences o WHERE o.event_id=e.id AND o.start_at>?) AS has_future "
                "FROM events e WHERE e.user_id=? AND e.active=1 ORDER BY e.created_at DESC",
                (now.timestamp(), user_id),
            )
            active = [
                r
                for r in rows
                if EventSpec.from_json(r["spec"]).repeat != "once"
                or r["has_future"]
                or EventSpec.from_json(r["spec"]).first_after(now) > now
            ]
            return active[offset : offset + limit]

    async def upcoming(self, user_id: int, event_id: str) -> datetime | None:
        self.authorize(user_id)
        async with self.store.transaction() as tx:
            event = await self._owned_event(tx, user_id, event_id)
            row = await tx.one(
                "SELECT start_at FROM occurrences WHERE event_id=? AND start_at>? ORDER BY start_at LIMIT 1",
                (event_id, self.clock().timestamp()),
            )
            if row:
                return datetime.fromtimestamp(row["start_at"], UTC)
            spec = EventSpec.from_json(event["spec"])
            if spec.repeat == "once" and spec.first_after(self.clock()) > self.clock():
                return spec.first_after(self.clock())
        return None
