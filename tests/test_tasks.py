import json
import sqlite3
import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from calendar_bot.domain import EventSpec, TaskSpec, UserError, local_day
from calendar_bot.parser import ParsedTask, normalize
from calendar_bot.service import CalendarService
from calendar_bot.storage import SCHEMA, SCHEMA_VERSION, Store, check_database, migrate
from tests.support import Clock, DatabaseCase


class TaskDomainTests(unittest.TestCase):
    def test_date_only_default_uses_original_local_day_and_roundtrips(self):
        reference = datetime(2026, 9, 24, 20, 30, tzinfo=UTC)
        task = normalize(
            ParsedTask(kind="task", title="Купить <хлеб> & сыр", date=None),
            reference,
            "Asia/Yerevan",
        )
        self.assertEqual(task.day, date(2026, 9, 25))
        self.assertEqual(TaskSpec.from_json(task.to_json()), task)
        self.assertEqual(set(task.to_dict()), {"title", "day"})
        self.assertEqual(local_day(reference, "Europe/Moscow"), date(2026, 9, 24))

    def test_task_schema_rejects_time_recurrence_and_reminders(self):
        for field, value in (("time", "16:00"), ("repeat", "daily"), ("reminders", [5])):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ParsedTask(kind="task", title="Дело", date=None, **{field: value})
        for title in ("", "\n", "a\rb", "x" * 181):
            with self.subTest(title=title), self.assertRaises(UserError):
                TaskSpec(title, date(2026, 9, 24))
        for invalid_date in ("2026-02-30", ""):
            with self.subTest(date=invalid_date), self.assertRaises(UserError):
                normalize(
                    ParsedTask(kind="task", title="Дело", date=invalid_date),
                    datetime(2026, 9, 24, tzinfo=UTC),
                    "UTC",
                )


class TaskServiceTests(DatabaseCase):
    async def test_mixed_batch_is_atomic_and_idempotent_even_after_deletion_and_restart(self):
        specs = [TaskSpec("Дело", date(2026, 9, 24)), self.spec()]
        with patch.object(self.service, "_rebuild_event", AsyncMock(side_effect=RuntimeError)):
            with self.assertRaises(RuntimeError):
                await self.service.create_drafts(101, "mixed", specs, self.clock(), auto_save=True)
        for table in ("tasks", "events", "drafts", "requests", "notifications"):
            self.assertEqual(await self.rows(f"SELECT * FROM {table}"), [])
        drafts = await self.service.create_drafts(101, "mixed", specs, self.clock(), auto_save=True)
        self.assertEqual([r["kind"] for r in drafts], ["task", "event"])
        self.assertEqual([r["status"] for r in drafts], ["saved", "saved"])
        self.assertEqual(len(await self.pending(drafts[1]["result_id"])), 3)
        task_id = drafts[0]["result_id"]
        await self.service.cancel_task(101, task_id, 1)
        await self.restart()
        repeated = await self.service.create_drafts(
            101, "mixed", specs, self.clock(), auto_save=True
        )
        self.assertEqual([r["result_id"] for r in repeated], [r["result_id"] for r in drafts])
        self.assertEqual(len(await self.rows("SELECT * FROM tasks")), 1)
        self.assertEqual((await self.rows("SELECT active FROM tasks"))[0]["active"], 0)

    async def test_task_dates_survive_timezone_changes_without_reminders(self):
        task_id = await self.create(spec=TaskSpec("Сегодня", date(2026, 9, 24)))
        past_id = await self.create(spec=TaskSpec("Вчера", date(2026, 9, 23)))
        self.assertEqual(await self.rows("SELECT * FROM occurrences"), [])
        self.assertEqual(await self.rows("SELECT * FROM notifications WHERE kind='reminder'"), [])
        self.clock.now = datetime(2026, 9, 24, 20, 30, tzinfo=UTC)
        async with self.store.transaction() as tx:
            self.assertEqual(
                await self.service.task_counts(tx, 101, local_day(self.clock(), "Asia/Yerevan")),
                (0, 2),
            )
        await self.service.preferences(101, timezone="America/New_York")
        async with self.store.transaction() as tx:
            self.assertEqual(
                await self.service.task_counts(
                    tx, 101, local_day(self.clock(), "America/New_York")
                ),
                (1, 1),
            )
        self.assertEqual((await self.service.task(101, task_id))["day"], "2026-09-24")
        self.assertEqual((await self.service.task(101, past_id))["day"], "2026-09-23")

    async def test_completion_owner_versions_edits_and_soft_deletion(self):
        task_id = await self.create(spec=TaskSpec("Купить", date(2026, 9, 24)))
        for operation in (
            self.service.task(202, task_id),
            self.service.set_task_completion(202, task_id, 1, True),
            self.service.cancel_task(202, task_id, 1),
            self.service.set_task_completion(303, task_id, 1, True),
            self.service.set_task_completion(101, task_id, None, True),
            self.service.cancel_task(101, task_id, None),
        ):
            with self.assertRaises(UserError):
                await operation
        draft = (
            await self.service.create_drafts(
                101,
                "edit-before-completion",
                [TaskSpec("Изменено", date(2026, 9, 25))],
                self.clock(),
                task_id=task_id,
                task_version=1,
            )
        )[0]
        await self.service.set_task_completion(101, task_id, 1, True)
        with self.assertRaises(UserError):
            await self.service.set_task_completion(101, task_id, 1, False)
        with self.assertRaises(UserError):
            await self.service.save_draft(101, draft["id"], 1)
        completed_at = (await self.service.task(101, task_id))["completed_at"]
        draft = (
            await self.service.create_drafts(
                101,
                "edit-completed",
                [TaskSpec("Перенесено", date(2026, 9, 25))],
                self.clock(),
                task_id=task_id,
                task_version=2,
            )
        )[0]
        self.assertEqual((await self.service.task(101, task_id))["title"], "Купить")
        await self.service.save_draft(101, draft["id"], 1)
        row = await self.service.task(101, task_id)
        self.assertEqual(row["completed_at"], completed_at)
        self.assertEqual(row["day"], "2026-09-25")
        await self.service.set_task_completion(101, task_id, 3, False)
        self.assertIsNone((await self.service.task(101, task_id))["completed_at"])
        await self.service.cancel_task(101, task_id, 4)
        with self.assertRaises(UserError):
            await self.service.task(101, task_id)
        self.assertEqual((await self.rows("SELECT version FROM tasks"))[0]["version"], 5)

    async def test_edit_cannot_convert_types_or_auto_save_or_skip_draft_expiry(self):
        task_id = await self.create(spec=TaskSpec("Дело", date(2026, 9, 24)))
        with self.assertRaises(UserError):
            await self.service.create_drafts(
                101, "convert", [self.spec()], self.clock(), task_id=task_id, task_version=1
            )
        event_id = await self.create()
        with self.assertRaises(UserError):
            await self.service.create_drafts(
                101,
                "convert-event",
                [TaskSpec("Дело", date(2026, 9, 24))],
                self.clock(),
                event_id=event_id,
                event_version=1,
            )
        with self.assertRaises(ValueError):
            await self.service.create_drafts(
                101,
                "auto-edit",
                [TaskSpec("Другое", date(2026, 9, 24))],
                self.clock(),
                task_id=task_id,
                task_version=1,
                auto_save=True,
            )
        draft = (
            await self.service.create_drafts(
                101, "pending", [TaskSpec("Не сохранять", date(2026, 9, 24))], self.clock()
            )
        )[0]
        self.clock.advance(days=2)
        with self.assertRaises(UserError):
            await self.service.save_draft(101, draft["id"], 1)
        self.assertEqual(len(await self.rows("SELECT * FROM tasks")), 1)

    async def test_mixed_pagination_and_history_have_no_gaps_or_duplicates(self):
        overdue = await self.create(spec=TaskSpec("Просрочено", date(2026, 9, 20)))
        historical = await self.create(spec=TaskSpec("История", date(2026, 9, 21)))
        await self.service.set_task_completion(101, historical, 1, True)
        task_ids = [
            await self.create(spec=TaskSpec(f"Дело {i}", date(2026, 9, 24))) for i in range(12)
        ]
        await self.service.set_task_completion(101, task_ids[0], 1, True)
        events = [await self.create(when=self.clock() + timedelta(hours=i + 1)) for i in range(3)]
        far = await self.create(spec=TaskSpec("Дальнее", date(2027, 1, 1)))
        await self.create(uid=202, spec=TaskSpec("Чужое", date(2026, 9, 24)))
        first, page, total = await self.service.plan_page(101, days=7)
        second, _, _ = await self.service.plan_page(101, days=7, page=1)
        all_rows = first + second
        self.assertEqual(total, 16)
        self.assertEqual(page, 0)
        self.assertEqual(
            [r.get("id", r.get("event_id")) for r in all_rows],
            [overdue, *events, *sorted(task_ids[1:]), task_ids[0]],
        )
        self.assertEqual((await self.service.plan_page(101, page=99))[1:], (1, 16))
        history, _, _ = await self.service.plan_page(101, start_day=date(2026, 9, 17))
        self.assertEqual([r["id"] for r in history], [overdue, historical])
        task_page, _, _ = await self.service.plan_page(101, start_day=date(2027, 1, 1))
        self.assertEqual([r["id"] for r in task_page], [far])

    async def test_only_unstarted_pending_summaries_are_invalidated(self):
        await self.service.preferences(101, summary_enabled=1)
        parts = json.dumps(["Первая часть", "Вторая часть"], ensure_ascii=False)
        async with self.store.transaction() as tx:
            await tx.execute(
                "UPDATE notifications SET parts=? WHERE user_id=101 AND kind='summary'", (parts,)
            )
            await tx.execute(
                "UPDATE notifications SET part_index=1 WHERE occurrence_key='2026-09-24'"
            )
        await self.create(spec=TaskSpec("Новое", date(2026, 9, 24)))
        notices = await self.rows("SELECT * FROM notifications ORDER BY occurrence_key")
        self.assertEqual((notices[0]["parts"], notices[0]["part_index"]), (parts, 1))
        self.assertIsNone(notices[1]["parts"])
        self.assertEqual(notices[1]["part_index"], 0)


class TaskMigrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_v1_data_and_legacy_draft_survive_explicit_migration(self):
        clock = Clock()
        spec = EventSpec.from_dict(
            {
                "title": "Старое событие",
                "day": "2026-09-25",
                "clock": "16:00",
                "timezone": "Asia/Yerevan",
                "repeat": "once",
                "weekdays": [],
                "reminder_minutes": None,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "migration.sqlite3"
            with sqlite3.connect(path) as db:
                db.executescript(SCHEMA + "\nPRAGMA user_version=1;")
                db.execute("INSERT INTO users(id,chat_id,timezone) VALUES(101,101,'Asia/Yerevan')")
                db.execute(
                    "INSERT INTO events(id,user_id,spec,created_at) VALUES('old',101,?,?)",
                    (spec.to_json(), clock().timestamp()),
                )
                db.execute(
                    "INSERT INTO drafts(id,user_id,spec,anchor_at,expires_at) VALUES('draft',101,?,?,?)",
                    (
                        spec.to_json(),
                        clock().timestamp(),
                        (clock() + timedelta(days=1)).timestamp(),
                    ),
                )
                db.execute(
                    "INSERT INTO requests VALUES('old-key',101,'[\"draft\"]',?)",
                    (clock().timestamp(),),
                )
                db.execute(
                    "INSERT INTO notifications(user_id,occurrence_key,kind,offset_minutes,due_at,start_at,version,next_attempt_at,parts,part_index) VALUES(101,'2026-09-24','summary',-1,0,9999999999,1,0,'[\"sent\",\"pending\"]',1)"
                )
                before_event = db.execute("SELECT * FROM events").fetchall()
                before_notice = db.execute("SELECT * FROM notifications").fetchall()
            db.close()
            with self.assertRaises(UserError):
                check_database(path)
            migrate(path)
            migrate(path)
            check_database(path)
            with sqlite3.connect(path) as db:
                self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION)
                self.assertEqual(db.execute("SELECT * FROM events").fetchall(), before_event)
                self.assertEqual(
                    db.execute("SELECT * FROM notifications").fetchall(), before_notice
                )
                self.assertEqual(
                    db.execute("SELECT kind,status FROM drafts").fetchone(), ("event", "pending")
                )
                self.assertEqual(db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 0)
            db.close()
            store = await Store.open(path)
            try:
                service = CalendarService(store, frozenset({101}), clock)
                draft = (await service.request_drafts(101, "old-key"))[0]
                self.assertEqual(draft["status"], "pending")
                await service.save_draft(101, draft["id"], 1)
            finally:
                await store.close()
