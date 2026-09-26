import unittest
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

from calendar_bot.domain import AgendaPeriod, TaskSpec, UserError
from tests.support import DatabaseCase


class AgendaPeriodTests(unittest.TestCase):
    def test_shifts_calendar_days_and_uses_dst_boundaries(self):
        period = AgendaPeriod(date(2026, 3, 28))
        start, end = period.window("Europe/Berlin")
        self.assertEqual(end - start, timedelta(days=7, hours=-1))
        self.assertEqual(period.shifted(1).start, date(2026, 4, 4))
        self.assertEqual(period.shifted(-1).start, date(2026, 3, 21))
        self.assertTrue(period.contains(date(2026, 4, 3)))
        self.assertFalse(period.contains(date(2026, 4, 4)))
        fall = AgendaPeriod(date(2026, 10, 24))
        start, end = fall.window("Europe/Berlin")
        self.assertEqual(end - start, timedelta(days=7, hours=1))
        with self.assertRaises(UserError):
            AgendaPeriod(date(9999, 12, 31))


class AgendaServiceTests(DatabaseCase):
    async def test_overdue_is_global_only_in_period_containing_today(self):
        ancient = await self.create(spec=TaskSpec("Старое", date(2026, 8, 1)))
        past = await self.create(spec=TaskSpec("Прошлая неделя", date(2026, 9, 20)))
        done = await self.create(spec=TaskSpec("Готово", date(2026, 9, 20)))
        await self.service.set_task_completion(101, done, 1, True)
        future = await self.create(spec=TaskSpec("Дальнее", date(2026, 11, 24)))
        await self.create(uid=202, spec=TaskSpec("Чужое", date(2026, 9, 20)))
        current, _, total = await self.service.plan_page(101)
        self.assertEqual(total, 2)
        self.assertEqual([r["id"] for r in current], [ancient, past])
        historical, _, _ = await self.service.plan_page(101, start_day=date(2026, 9, 17))
        self.assertEqual([r["id"] for r in historical], [past, done])
        self.assertFalse(any(r["overdue_group"] for r in historical))
        overlapping, _, total = await self.service.plan_page(101, start_day=date(2026, 9, 20))
        self.assertEqual([r["id"] for r in overlapping], [ancient, past, done])
        self.assertEqual(total, 3)
        far, _, _ = await self.service.plan_page(101, start_day=date(2026, 11, 20))
        self.assertEqual([r["id"] for r in far], [future])
        with self.assertRaises(UserError):
            await self.service.plan_page(303)

    async def test_future_week_and_card_are_read_only_with_daily_and_weekly_recurrence(self):
        daily = await self.create(repeat="daily")
        weekly = await self.create(repeat="weekly", weekdays=(0, 3))
        await self.create(uid=202, repeat="daily")
        before = await self.rows("SELECT * FROM occurrences ORDER BY event_id,original_day")
        notices = await self.rows("SELECT * FROM notifications ORDER BY id")
        first, _, total = await self.service.plan_page(101, start_day=date(2026, 11, 23))
        second, _, _ = await self.service.plan_page(101, start_day=date(2026, 11, 23), page=1)
        self.assertEqual(total, 9)
        self.assertEqual(len(first + second), 9)
        self.assertEqual({row["event_id"] for row in first + second}, {daily, weekly})
        selected = next(row for row in first if row["event_id"] == daily)
        resolved = await self.service.occurrence(101, daily, selected["original_day"], 1)
        self.assertEqual(resolved["start_at"], selected["start_at"])
        self.assertEqual(
            await self.rows("SELECT * FROM occurrences ORDER BY event_id,original_day"), before
        )
        self.assertEqual(await self.rows("SELECT * FROM notifications ORDER BY id"), notices)
        with self.assertRaises(UserError):
            await self.service.occurrence(202, daily, selected["original_day"], 1)
        with self.assertRaises(UserError):
            await self.service.occurrence(101, daily, selected["original_day"], 2)

    async def test_far_occurrence_can_be_moved_and_cancelled_with_original_identity(self):
        event_id = await self.create(repeat="daily")
        moved = self.spec(when=self.clock() + timedelta(days=62), title="Перенос")
        await self.edit(event_id, moved, day="2026-11-23")
        version = (await self.service.event(101, event_id))["version"]
        await self.service.cancel_event(101, event_id, version, "2026-11-24")
        rows, _, _ = await self.service.plan_page(101, start_day=date(2026, 11, 23))
        self.assertEqual(len(rows), 6)
        self.assertNotIn("2026-11-24", [row["original_day"] for row in rows])
        selected = next(row for row in rows if row["original_day"] == "2026-11-23")
        self.assertEqual(selected["day"], moved.day.isoformat())
        self.assertIn("Перенос", selected["spec"])
        with self.assertRaises(UserError):
            await self.service.occurrence(101, event_id, "2026-11-24", version + 1)
        await self.service.cancel_event(101, event_id, version + 1, "2026-11-23")
        rows, _, _ = await self.service.plan_page(101, start_day=date(2026, 11, 23))
        self.assertEqual(len(rows), 5)

    async def test_far_moves_enter_and_leave_range_without_duplicates(self):
        event_id = await self.create(repeat="daily")
        await self.edit(
            event_id,
            self.spec(when=datetime(2026, 11, 24, 8, tzinfo=UTC), title="Издалека"),
            day="2026-12-24",
        )
        await self.edit(
            event_id,
            self.spec(when=datetime(2026, 12, 24, 8, tzinfo=UTC), title="Из недели"),
            day="2026-11-25",
        )
        rows, _, total = await self.service.plan_page(101, start_day=date(2026, 11, 23))
        self.assertEqual(total, 7)
        self.assertEqual(len({r["original_day"] for r in rows}), 7)
        self.assertIn("2026-12-24", [r["original_day"] for r in rows])
        self.assertNotIn("2026-11-25", [r["original_day"] for r in rows])

    async def test_history_uses_saved_spec_and_never_recreates_past_occurrences(self):
        spec = self.spec(repeat="daily")
        original_start = spec.first_after(self.clock()).timestamp()
        event_id = await self.create(spec=spec)
        self.clock.advance(hours=2)
        await self.edit(
            event_id, replace(spec, title="Новая серия", clock=spec.clock.replace(hour=19))
        )
        rows, _, _ = await self.service.plan_page(101, days=1)
        self.assertEqual(len(rows), 1)
        self.assertIn("Встреча", rows[0]["spec"])
        self.assertEqual(rows[0]["start_at"], original_start)
        with self.assertRaises(UserError):
            await self.service.occurrence(101, event_id, spec.day.isoformat(), 2)
        self.clock.advance(days=7)
        await self.service.cancel_event(101, event_id, 2)
        history, _, _ = await self.service.plan_page(101, start_day=spec.day)
        self.assertIn("Встреча", history[0]["spec"])
        self.assertFalse(history[0]["active"])
        before_creation, _, total = await self.service.plan_page(101, start_day=date(2026, 9, 1))
        self.assertEqual((before_creation, total), ([], 0))

    async def test_week_boundaries_follow_user_timezone_while_task_date_is_fixed(self):
        await self.service.preferences(101, timezone="Europe/Berlin")
        self.clock.now = datetime(2026, 3, 28, 8, tzinfo=UTC)
        event_id = await self.create(repeat="daily", timezone="Europe/Berlin")
        task_id = await self.create(spec=TaskSpec("Дело", date(2026, 3, 31)))
        rows, _, total = await self.service.plan_page(101, start_day=date(2026, 3, 28))
        self.assertEqual(total, 8)
        event_rows = [row for row in rows if row["kind"] == "event"]
        self.assertEqual(len(event_rows), 7)
        self.assertEqual(event_rows[1]["start_at"] - event_rows[0]["start_at"], 23 * 3600)
        await self.service.preferences(101, timezone="America/New_York")
        rows, _, _ = await self.service.plan_page(101, start_day=date(2026, 3, 28))
        self.assertEqual(next(r for r in rows if r.get("id") == task_id)["day"], "2026-03-31")
        self.assertTrue(any(r.get("event_id") == event_id for r in rows))
