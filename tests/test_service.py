import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from calendar_bot.domain import UserError
from calendar_bot.locking import InstanceLock
from calendar_bot.storage import check_database, migrate
from tests.support import DatabaseCase


class ServiceTests(DatabaseCase):
    async def test_auto_save_is_idempotent_after_restart_and_does_not_restore_deleted_events(self):
        specs = [self.spec(title="Первая"), self.spec(title="Вторая", offsets=(30,))]
        rows = await self.service.create_drafts(101, "auto", specs, self.clock(), auto_save=True)
        self.assertTrue(all(row["status"] == "saved" for row in rows))
        self.assertEqual(
            [r["offset_minutes"] for r in await self.pending(rows[0]["result_id"])], [15, 5, 1]
        )
        self.assertEqual(
            [r["offset_minutes"] for r in await self.pending(rows[1]["result_id"])], [30]
        )
        await self.restart()
        again = await self.service.create_drafts(101, "auto", specs, self.clock(), auto_save=True)
        self.assertEqual(rows, again)
        await self.service.cancel_event(101, rows[0]["result_id"], 1)
        await self.service.create_drafts(101, "auto", specs, self.clock(), auto_save=True)
        self.assertEqual(len(await self.rows("SELECT * FROM events")), 2)
        self.assertEqual(len(await self.rows("SELECT * FROM events WHERE active=1")), 1)
        self.assertEqual(await self.pending(rows[0]["result_id"]), [])

    async def test_auto_save_rolls_back_entire_batch_if_scheduling_fails(self):
        original = self.service._rebuild_event
        calls = 0

        async def fail_second(*args):
            nonlocal calls
            await original(*args)
            calls += 1
            if calls == 2:
                raise RuntimeError("Scheduling interrupted")

        specs = [self.spec(title="Первая"), self.spec(title="Вторая")]
        with patch.object(self.service, "_rebuild_event", side_effect=fail_second):
            with self.assertRaises(RuntimeError):
                await self.service.create_drafts(101, "auto", specs, self.clock(), auto_save=True)
        for table in ("drafts", "requests", "events", "occurrences", "notifications"):
            self.assertEqual(await self.rows(f"SELECT * FROM {table}"), [])
        await self.service.create_drafts(101, "auto", specs, self.clock(), auto_save=True)
        self.assertEqual(len(await self.rows("SELECT * FROM events")), 2)

    async def test_auto_save_is_only_for_new_authorized_requests(self):
        event_id = await self.create()
        with self.assertRaises(UserError):
            await self.service.create_drafts(
                999, "unauthorized", [self.spec()], self.clock(), auto_save=True
            )
        with self.assertRaises(ValueError):
            await self.service.create_drafts(
                101,
                "edit",
                [self.spec()],
                self.clock(),
                event_id=event_id,
                event_version=1,
                auto_save=True,
            )
        legacy = await self.service.create_drafts(101, "legacy", [self.spec()], self.clock())
        again = await self.service.create_drafts(
            101, "legacy", [self.spec()], self.clock(), auto_save=True
        )
        self.assertEqual(again, legacy)
        self.assertEqual(again[0]["status"], "pending")
        self.assertEqual(len(await self.rows("SELECT * FROM events")), 1)

    async def test_confirmation_and_duplicate_updates_are_idempotent(self):
        spec = self.spec(title="Штурм // Ориентир 2027")
        first = await self.service.create_drafts(101, "message:one", [spec], self.clock())
        again = await self.service.create_drafts(101, "message:one", [spec], self.clock())
        self.assertEqual(first[0]["id"], again[0]["id"])
        self.assertEqual(await self.rows("SELECT * FROM events"), [])
        event_id = await self.service.save_draft(101, first[0]["id"], 1)
        self.assertEqual(event_id, await self.service.save_draft(101, first[0]["id"], 1))
        self.assertEqual(len(await self.rows("SELECT * FROM events")), 1)
        self.assertEqual(
            [row["offset_minutes"] for row in await self.pending(event_id)], [15, 5, 1]
        )

    async def test_owner_checks_on_drafts_events_and_callbacks(self):
        drafts = await self.service.create_drafts(101, "one", [self.spec()], self.clock())
        with self.assertRaises(UserError):
            await self.service.save_draft(202, drafts[0]["id"], 1)
        event_id = await self.service.save_draft(101, drafts[0]["id"], 1)
        with self.assertRaises(UserError):
            await self.service.event(202, event_id)
        with self.assertRaises(UserError):
            await self.service.cancel_event(202, event_id, 1)
        with self.assertRaises(UserError):
            await self.service.ensure_user(999, 999)

    async def test_draft_survives_restart_and_expires(self):
        drafts = await self.service.create_drafts(
            101, "one", [self.spec(when=self.clock() + timedelta(days=2))], self.clock()
        )
        await self.restart()
        self.assertEqual((await self.service.draft(101, drafts[0]["id"]))["status"], "pending")
        self.clock.advance(hours=25)
        with self.assertRaises(UserError):
            await self.service.save_draft(101, drafts[0]["id"], 1)

    async def test_stale_draft_and_passed_event_cannot_be_confirmed(self):
        draft = (await self.service.create_drafts(101, "first", [self.spec()], self.clock()))[0]
        await self.service.create_drafts(
            101,
            "correction",
            [self.spec(title="Исправлено")],
            self.clock(),
            draft_id=draft["id"],
            draft_version=1,
        )
        with self.assertRaises(UserError):
            await self.service.save_draft(101, draft["id"], 1)
        self.clock.advance(hours=2)
        with self.assertRaises(UserError):
            await self.service.save_draft(101, draft["id"], 2)

    async def test_defaults_change_only_inherited_reminders(self):
        inherited = await self.create()
        custom = await self.create(offsets=(30,))
        await self.service.preferences(101, reminders=(10, 2))
        self.assertEqual([r["offset_minutes"] for r in await self.pending(inherited)], [10, 2])
        self.assertEqual([r["offset_minutes"] for r in await self.pending(custom)], [30])
        await self.service.preferences(101, reminders=())
        self.assertEqual(await self.pending(inherited), [])
        self.assertEqual(len(await self.pending(custom)), 1)

    async def test_move_one_occurrence_and_cancel_series(self):
        spec = self.spec(repeat="daily")
        event_id = await self.create(spec=spec)
        day = spec.day.isoformat()
        moved = self.spec(when=self.clock() + timedelta(days=1, hours=3), title="Перенесено")
        await self.edit(event_id, moved, day=day)
        rows = await self.rows(
            "SELECT * FROM occurrences WHERE event_id=? AND original_day=?", (event_id, day)
        )
        self.assertEqual(rows[0]["start_at"], moved.first_after(self.clock()).timestamp())
        event = await self.service.event(101, event_id)
        await self.service.cancel_event(101, event_id, event["version"])
        self.assertEqual(await self.pending(event_id), [])
        self.assertEqual(
            await self.rows("SELECT * FROM occurrences WHERE event_id=?", (event_id,)), []
        )

    async def test_cancel_one_and_edit_series_keep_exceptions(self):
        spec = self.spec(repeat="daily")
        event_id = await self.create(spec=spec)
        await self.service.cancel_event(101, event_id, 1, spec.day.isoformat())
        moved = self.spec(when=self.clock() + timedelta(days=3, hours=2), title="Отдельная")
        await self.edit(event_id, moved, day=(spec.day + timedelta(days=1)).isoformat())
        await self.edit(
            event_id, replace(spec, title="Новая серия", clock=spec.clock.replace(hour=15))
        )
        occurrences = await self.rows(
            "SELECT * FROM occurrences WHERE event_id=? ORDER BY original_day", (event_id,)
        )
        self.assertNotIn(spec.day.isoformat(), [o["original_day"] for o in occurrences])
        special = next(
            o
            for o in occurrences
            if o["original_day"] == (spec.day + timedelta(days=1)).isoformat()
        )
        self.assertEqual(special["start_at"], moved.first_after(self.clock()).timestamp())
        self.assertEqual(json.loads(special["spec"])["title"], "Отдельная")

    async def test_history_does_not_move_when_series_is_edited(self):
        spec = self.spec(repeat="daily")
        event_id = await self.create(spec=spec)
        old_start = spec.first_after(self.clock()).timestamp()
        self.clock.advance(hours=2)
        await self.edit(
            event_id, replace(spec, title="Изменено", clock=spec.clock.replace(hour=19))
        )
        rows = await self.rows(
            "SELECT * FROM occurrences WHERE event_id=? AND original_day=?",
            (event_id, spec.day.isoformat()),
        )
        self.assertEqual(rows[0]["start_at"], old_start)
        self.assertEqual(json.loads(rows[0]["spec"])["title"], "Встреча")

    async def test_timezone_change_does_not_shift_events(self):
        event_id = await self.create(repeat="daily")
        before = [r["start_at"] for r in await self.pending(event_id)]
        await self.service.preferences(101, timezone="Europe/Moscow")
        self.assertEqual([r["start_at"] for r in await self.pending(event_id)], before)

    async def test_old_event_revision_cannot_overwrite_new_changes(self):
        event_id = await self.create()
        draft = (
            await self.service.create_drafts(
                101,
                "edit",
                [self.spec(title="Новый")],
                self.clock(),
                event_id=event_id,
                event_version=1,
            )
        )[0]
        await self.edit(event_id, self.spec(title="Другой"))
        with self.assertRaises(UserError):
            await self.service.save_draft(101, draft["id"], 1)

    async def test_backup_migration_and_lock(self):
        await self.create()
        migrate(self.path)  # Applying the same version is a no-op.
        backup = self.path.with_name("backup.sqlite3")
        with (
            closing(sqlite3.connect(self.path)) as source,
            closing(sqlite3.connect(backup)) as target,
        ):
            source.backup(target)
        check_database(backup)
        with closing(sqlite3.connect(backup)) as restored:
            self.assertEqual(restored.execute("SELECT count(*) FROM events").fetchone()[0], 1)
        with InstanceLock(self.path), self.assertRaises(UserError):
            with InstanceLock(self.path):
                pass

    async def test_far_future_oneoff_is_visible_and_eventually_materialized(self):
        event_id = await self.create(when=self.clock() + timedelta(days=60))
        self.assertEqual(len(await self.pending(event_id)), 3)
        self.assertIsNotNone(await self.service.upcoming(101, event_id))
        self.clock.advance(days=40)
        await self.service.refresh()
        self.assertEqual(len(await self.pending(event_id)), 3)

    async def test_single_occurrence_can_be_moved_beyond_the_rolling_horizon(self):
        spec = self.spec(repeat="daily")
        event_id = await self.create(spec=spec)
        moved = self.spec(when=self.clock() + timedelta(days=90), title="Дальний перенос")
        await self.edit(event_id, moved, day=spec.day.isoformat())
        row = await self.service.occurrence(101, event_id, spec.day.isoformat())
        self.assertEqual(row["start_at"], moved.first_after(self.clock()).timestamp())
        event = await self.service.event(101, event_id)
        await self.service.cancel_event(101, event_id, event["version"], spec.day.isoformat())
        with self.assertRaises(UserError):
            await self.service.occurrence(101, event_id, spec.day.isoformat())

    async def test_recurring_creation_after_local_midnight(self):
        self.clock.now = datetime(2026, 9, 24, 20, 30, tzinfo=UTC)
        event_id = await self.create(when=self.clock() + timedelta(minutes=30), repeat="daily")
        self.assertEqual(
            (await self.pending(event_id))[0]["occurrence_key"], f"{event_id}:2026-09-25"
        )

    async def test_moving_oneoff_from_agenda_updates_event_instead_of_hidden_exception(self):
        original = self.spec()
        event_id = await self.create(spec=original)
        moved = self.spec(when=self.clock() + timedelta(days=2), title="Перенос разового")
        await self.edit(event_id, moved, day=original.day.isoformat())
        self.clock.advance(days=1)
        listed = await self.service.event_list(101)
        self.assertEqual([row["id"] for row in listed], [event_id])
        self.assertEqual(json.loads(listed[0]["spec"])["day"], moved.day.isoformat())
        self.assertEqual(
            await self.rows("SELECT * FROM exceptions WHERE event_id=?", (event_id,)), []
        )

    async def test_preserved_exceptions_stay_accessible_after_stopping_recurrence(self):
        original = self.spec(repeat="daily")
        event_id = await self.create(spec=original)
        moved = self.spec(when=self.clock() + timedelta(days=7))
        await self.edit(event_id, moved, day=(original.day + timedelta(days=1)).isoformat())
        await self.edit(event_id, self.spec())
        self.clock.advance(days=2)
        self.assertEqual([row["id"] for row in await self.service.event_list(101)], [event_id])
        changed = self.spec(when=self.clock() + timedelta(days=8), title="Исправленное исключение")
        await self.edit(event_id, changed, day=(original.day + timedelta(days=1)).isoformat())
        row = await self.service.occurrence(
            101, event_id, (original.day + timedelta(days=1)).isoformat()
        )
        self.assertEqual(json.loads(row["spec"])["title"], "Исправленное исключение")
