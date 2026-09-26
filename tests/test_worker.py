import json
from datetime import UTC, date, datetime, timedelta

from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
)
from aiogram.methods import SendMessage

from calendar_bot.domain import TaskSpec
from calendar_bot.worker import NotificationWorker
from tests.support import DatabaseCase, FakeBot, entity_fragments


class NotificationTests(DatabaseCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.bot = FakeBot()
        self.worker = NotificationWorker(self.service, self.bot)

    async def drain(self, maximum=30):
        for _ in range(maximum):
            if not await self.worker.dispatch_one():
                return
        self.fail("Queue did not settle")

    async def test_reminder_counters_follow_task_mutations_on_every_offset(self):
        first = await self.create(spec=TaskSpec("Первое", date(2026, 9, 24)))
        second = await self.create(spec=TaskSpec("Второе", date(2026, 9, 24)))
        past = await self.create(spec=TaskSpec("Прошлое", date(2026, 9, 23)))
        done = await self.create(spec=TaskSpec("Выполнено", date(2026, 9, 24)))
        deleted = await self.create(spec=TaskSpec("Удалено", date(2026, 9, 24)))
        await self.service.set_task_completion(101, done, 1, True)
        await self.service.cancel_task(101, deleted, 1)
        await self.create(spec=TaskSpec("Будущее", date(2026, 9, 25)))
        await self.create(uid=202, spec=TaskSpec("Чужое", date(2026, 9, 24)))
        await self.create(offsets=(15, 5, 1, 0))
        self.clock.advance(minutes=45)
        await self.drain()
        self.assertIn("На сегодня осталось 2 дела.", self.bot.sent[-1]["text"])
        self.assertIn("Просроченных: 1.", self.bot.sent[-1]["text"])
        await self.service.set_task_completion(101, first, 1, True)
        draft = (
            await self.service.create_drafts(
                101,
                "move-task",
                [TaskSpec("Перенесено", date(2026, 9, 25))],
                self.clock(),
                task_id=second,
                task_version=1,
            )
        )[0]
        await self.service.save_draft(101, draft["id"], 1)
        added = await self.create(spec=TaskSpec("Новое", date(2026, 9, 24)))
        self.clock.advance(minutes=10)
        await self.drain()
        self.assertIn("На сегодня осталось 1 дело.", self.bot.sent[-1]["text"])
        await self.service.set_task_completion(101, first, 2, False)
        await self.service.cancel_task(101, past, 1)
        self.clock.advance(minutes=4)
        await self.drain()
        self.assertIn("На сегодня осталось 2 дела.", self.bot.sent[-1]["text"])
        self.assertNotIn("Просроченных:", self.bot.sent[-1]["text"])
        await self.service.set_task_completion(101, first, 3, True)
        await self.service.set_task_completion(101, added, 1, True)
        self.clock.advance(minutes=1, seconds=4)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 4)
        self.assertIn("Начинается сейчас", self.bot.sent[-1]["text"])
        self.assertIn("На сегодня нет невыполненных дел.", self.bot.sent[-1]["text"])

    async def test_retry_recomputes_task_counters_from_storage_after_restart(self):
        task_id = await self.create(spec=TaskSpec("Первое", date(2026, 9, 24)))
        await self.create(offsets=(15,))
        self.clock.advance(minutes=45)
        self.bot.errors = [
            TelegramNetworkError(method=SendMessage(chat_id=101, text="test"), message="network")
        ]
        await self.drain()
        self.assertEqual(self.bot.sent, [])
        await self.service.set_task_completion(101, task_id, 1, True)
        await self.create(spec=TaskSpec("Вчера", date(2026, 9, 23)))
        await self.restart()
        self.worker = NotificationWorker(self.service, self.bot)
        self.clock.advance(seconds=6)
        await self.drain()
        self.assertIn("На сегодня нет невыполненных дел.", self.bot.sent[0]["text"])
        self.assertIn("Просроченных: 1.", self.bot.sent[0]["text"])

    async def test_counters_use_send_day_and_user_timezone_across_midnight(self):
        self.clock.now = datetime(2026, 9, 24, 19, 55, tzinfo=UTC)
        for day in (23, 24, 25):
            await self.create(spec=TaskSpec(f"Дело {day}", date(2026, 9, day)))
        await self.create(
            when=datetime(2026, 9, 24, 20, 5, tzinfo=UTC),
            timezone="America/New_York",
            offsets=(15, 5),
        )
        await self.drain()
        self.assertIn("На сегодня осталось 1 дело.", self.bot.sent[-1]["text"])
        self.assertIn("Просроченных: 1.", self.bot.sent[-1]["text"])
        self.clock.advance(minutes=5)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 2)
        self.assertIn("На сегодня осталось 1 дело.", self.bot.sent[-1]["text"])
        self.assertIn("Просроченных: 2.", self.bot.sent[-1]["text"])

    async def test_task_only_summary_includes_overdue_and_link_but_not_completed(self):
        self.clock.now = datetime(2026, 9, 24, 3, tzinfo=UTC)
        await self.service.preferences(101, summary_enabled=1)
        await self.create(spec=TaskSpec("Сегодня & <дело>", date(2026, 9, 24)))
        await self.create(spec=TaskSpec("Просроченное", date(2026, 9, 23)))
        done = await self.create(spec=TaskSpec("Готовое", date(2026, 9, 24)))
        await self.service.set_task_completion(101, done, 1, True)
        await self.create(spec=TaskSpec("Завтрашнее", date(2026, 9, 25)))
        await self.create(uid=202, spec=TaskSpec("Чужое", date(2026, 9, 24)))
        self.clock.advance(hours=2)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 1)
        text = self.bot.sent[0]["text"]
        self.assertIn("Сегодня & <дело>", text)
        self.assertIn("Просроченное", text)
        self.assertIn("23 сентября", text)
        for absent in ("Готовое", "Завтрашнее", "Чужое"):
            self.assertNotIn(absent, text)
        self.assertEqual(
            self.bot.sent[0]["reply_markup"].inline_keyboard[0][0].callback_data, "t:show:0"
        )
        self.assertEqual(await self.rows("SELECT * FROM notifications WHERE kind='reminder'"), [])

    async def test_started_task_summary_remains_a_snapshot_after_all_tasks_complete(self):
        self.clock.now = datetime(2026, 9, 24, 3, tzinfo=UTC)
        await self.service.preferences(101, summary_enabled=1)
        ids = [
            await self.create(spec=TaskSpec(f"{n}: " + "😀" * 160, date(2026, 9, 24)))
            for n in range(22)
        ]
        self.clock.advance(hours=2)
        await self.worker.dispatch_one()
        notice = (await self.rows("SELECT * FROM notifications WHERE occurrence_key='2026-09-24'"))[
            0
        ]
        snapshot = notice["parts"]
        self.assertGreater(len(json.loads(snapshot)), 1)
        for task_id in ids:
            await self.service.set_task_completion(101, task_id, 1, True)
        await self.restart()
        self.worker = NotificationWorker(self.service, self.bot)
        for _ in range(6):
            self.clock.advance(seconds=2)
            await self.drain()
        notice = (await self.rows("SELECT * FROM notifications WHERE occurrence_key='2026-09-24'"))[
            0
        ]
        self.assertEqual(notice["parts"], snapshot)
        self.assertEqual(notice["status"], "sent")
        self.assertEqual(notice["part_index"], len(json.loads(snapshot)))
        self.assertEqual([m["text"] for m in self.bot.sent], json.loads(snapshot))

    async def test_three_notifications_and_no_duplicate_after_restart(self):
        event_id = await self.create()
        self.clock.advance(minutes=45)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 1)
        await self.restart()
        self.worker = NotificationWorker(self.service, self.bot)
        await self.service.refresh()
        await self.drain()
        self.assertEqual(len(self.bot.sent), 1)
        self.clock.advance(minutes=10)
        await self.drain()
        self.clock.advance(minutes=4)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 3)
        self.assertTrue(all("Встреча" in m["text"] for m in self.bot.sent))
        for message in self.bot.sent:
            self.assertIsNone(message["parse_mode"])
            self.assertIn(
                ("bold", "Встреча"), entity_fragments(message["text"], message["entities"])
            )
        self.assertEqual(await self.pending(event_id), [])

    async def test_late_creation_coalesces_old_offsets_and_keeps_next(self):
        event_id = await self.create(when=self.clock() + timedelta(minutes=4))
        await self.drain()
        self.assertEqual(len(self.bot.sent), 1)
        self.assertIn("Через 4 мин", self.bot.sent[0]["text"])
        self.assertEqual([r["offset_minutes"] for r in await self.pending(event_id)], [1])
        self.clock.advance(minutes=3)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 2)

    async def test_started_events_are_not_replayed(self):
        event_id = await self.create()
        self.clock.advance(hours=2)
        await self.drain()
        self.assertEqual(self.bot.sent, [])
        self.assertEqual(await self.pending(event_id), [])

    async def test_on_start_reminder_allows_a_five_second_tick(self):
        await self.create(offsets=(0,))
        self.clock.advance(hours=1, seconds=4)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 1)
        self.assertIn("Начинается сейчас", self.bot.sent[0]["text"])

    async def test_network_retry_is_persisted(self):
        await self.create(offsets=(15,))
        self.clock.advance(minutes=45)
        self.bot.errors = [
            TelegramNetworkError(method=SendMessage(chat_id=101, text="test"), message="network")
        ]
        await self.drain()
        self.assertEqual(self.bot.sent, [])
        await self.restart()
        self.worker = NotificationWorker(self.service, self.bot)
        await self.drain()
        self.assertEqual(self.bot.sent, [])
        self.clock.advance(seconds=6)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 1)

    async def test_retry_after_is_honoured(self):
        await self.create(offsets=(15,))
        self.clock.advance(minutes=45)
        self.bot.errors = [
            TelegramRetryAfter(
                method=SendMessage(chat_id=101, text="test"), message="wait", retry_after=60
            )
        ]
        await self.drain()
        self.clock.advance(seconds=59)
        await self.drain()
        self.assertEqual(self.bot.sent, [])
        self.clock.advance(seconds=1)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 1)

    async def test_retry_recomputes_remaining_time(self):
        await self.create(offsets=(15,))
        self.clock.advance(minutes=45)
        self.bot.errors = [
            TelegramNetworkError(method=SendMessage(chat_id=101, text="test"), message="network")
        ]
        await self.drain()
        self.clock.advance(minutes=10)
        await self.drain()
        self.assertIn("Через 5 мин", self.bot.sent[0]["text"])

    async def test_retry_after_applies_to_new_notifications_after_restart(self):
        await self.create(offsets=(15,))
        self.clock.advance(minutes=45)
        self.bot.errors = [
            TelegramRetryAfter(
                method=SendMessage(chat_id=101, text="test"), message="wait", retry_after=120
            )
        ]
        await self.drain()
        await self.create(uid=202, when=self.clock() + timedelta(minutes=10), offsets=(10,))
        await self.restart()
        self.worker = NotificationWorker(self.service, self.bot)
        await self.drain()
        self.assertEqual(self.bot.sent, [])
        self.clock.advance(minutes=2)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 2)

    async def test_renaming_does_not_replay_coalesced_reminders(self):
        original = self.spec()
        event_id = await self.create(spec=original)
        self.clock.advance(minutes=56)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 1)
        from dataclasses import replace

        await self.edit(event_id, replace(original, title="Новое имя"))
        self.clock.advance(seconds=2)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 1)

    async def test_blocked_bot_requires_start_to_resume(self):
        await self.create()
        self.clock.advance(minutes=45)
        self.bot.errors = [
            TelegramForbiddenError(method=SendMessage(chat_id=101, text="test"), message="blocked")
        ]
        await self.drain()
        self.assertEqual((await self.service.user(101))["blocked"], 1)
        self.clock.advance(minutes=1)
        await self.service.refresh()
        await self.drain()
        self.assertEqual(self.bot.sent, [])
        await self.service.ensure_user(101, 101, resume=True)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 1)

    async def test_permanent_error_is_not_retried_every_refill(self):
        event_id = await self.create(offsets=(15,))
        self.clock.advance(minutes=45)
        self.bot.errors = [
            TelegramBadRequest(method=SendMessage(chat_id=101, text="test"), message="invalid")
        ]
        await self.drain()
        await self.service.refresh()
        self.clock.advance(seconds=10)
        await self.drain()
        self.assertEqual(self.bot.sent, [])
        self.assertEqual(
            (await self.rows("SELECT status FROM notifications WHERE event_id=?", (event_id,)))[0][
                "status"
            ],
            "failed",
        )

    async def test_cancelled_event_is_not_sent(self):
        event_id = await self.create()
        await self.service.cancel_event(101, event_id, 1)
        self.clock.advance(minutes=59)
        await self.drain()
        self.assertEqual(self.bot.sent, [])

    async def test_revoked_user_is_not_sent_notifications(self):
        await self.create()
        self.service.allowed = frozenset({202})
        self.clock.advance(minutes=59)
        await self.drain()
        self.assertEqual(self.bot.sent, [])

    async def test_summary_observes_local_day_and_sends_without_ai(self):
        self.clock.now = datetime(2026, 9, 24, 3, tzinfo=UTC)
        await self.service.preferences(101, summary_enabled=1)
        await self.create(when=datetime(2026, 9, 24, 19, 30, tzinfo=UTC), title="Вечер", offsets=())
        await self.create(
            when=datetime(2026, 9, 24, 20, 30, tzinfo=UTC), title="Следующий день", offsets=()
        )
        await self.create(uid=202, title="Чужое", offsets=())
        self.clock.advance(hours=2)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 1)
        self.assertIn("23:30 — Вечер", self.bot.sent[0]["text"])
        self.assertIn(
            ("bold", "23:30"),
            entity_fragments(self.bot.sent[0]["text"], self.bot.sent[0]["entities"]),
        )
        self.assertNotIn("Следующий день", self.bot.sent[0]["text"])
        self.assertNotIn("Чужое", self.bot.sent[0]["text"])
        await self.service.refresh()
        self.clock.advance(seconds=10)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 1)

    async def test_empty_summary_is_optional(self):
        self.clock.now = datetime(2026, 9, 24, 5, tzinfo=UTC)
        await self.service.preferences(101, summary_enabled=1)
        await self.drain()
        self.assertEqual(self.bot.sent, [])
        await self.service.preferences(101, summary_empty=1)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 1)
        self.assertIn("событий нет", self.bot.sent[0]["text"])

    async def test_summary_catchup_has_two_hour_deadline(self):
        self.clock.now = datetime(2026, 9, 24, 7, 1, tzinfo=UTC)
        await self.service.preferences(101, summary_enabled=1, summary_empty=1)
        await self.drain()
        self.assertEqual(self.bot.sent, [])

    async def test_summary_parts_continue_after_restart(self):
        self.clock.now = datetime(2026, 9, 24, 3, tzinfo=UTC)
        await self.service.preferences(101, summary_enabled=1)
        for index in range(25):
            await self.create(
                title=f"{index}: " + "😀" * 160, offsets=(), when=self.clock() + timedelta(hours=5)
            )
        self.clock.advance(hours=2)
        await self.worker.dispatch_one()
        first = self.bot.sent[0]["text"]
        await self.restart()
        self.worker = NotificationWorker(self.service, self.bot)
        for _ in range(10):
            self.clock.advance(seconds=2)
            await self.drain()
        self.assertGreater(len(self.bot.sent), 1)
        self.assertEqual([m["text"] for m in self.bot.sent].count(first), 1)
        for message in self.bot.sent:
            self.assertLessEqual(len(message["text"].encode("utf-16-le")) // 2, 3500)
            self.assertTrue(entity_fragments(message["text"], message["entities"]))
        rows = await self.rows("SELECT * FROM notifications WHERE kind='summary' AND status='sent'")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["part_index"], len(json.loads(rows[0]["parts"])))
        self.assertTrue(all(isinstance(part, str) for part in json.loads(rows[0]["parts"])))
        delivered = "\n".join(m["text"] for m in self.bot.sent)
        for index in range(25):
            self.assertEqual(delivered.count(f" — {index}: " + "😀" * 160), 1)

    async def test_saved_legacy_parts_resume_unchanged_after_restart(self):
        self.clock.now = datetime(2026, 9, 24, 5, tzinfo=UTC)
        await self.service.preferences(101, summary_enabled=1, summary_empty=1)
        legacy = [
            "☀️ События на 24.09.2026 · Asia/Yerevan\n\n10:00 — Уже отправлено",
            "14:00 — 😀 План & <2027> _команда_\n15:00 — Вторая встреча",
        ]
        async with self.store.transaction() as tx:
            await tx.execute(
                "UPDATE notifications SET parts=?,part_index=1 WHERE kind='summary' AND user_id=101 AND occurrence_key='2026-09-24'",
                (json.dumps(legacy, ensure_ascii=False),),
            )
        await self.restart()
        self.worker = NotificationWorker(self.service, self.bot)
        await self.drain()
        self.assertEqual([m["text"] for m in self.bot.sent], [legacy[1]])
        fragments = entity_fragments(self.bot.sent[0]["text"], self.bot.sent[0]["entities"])
        self.assertEqual(fragments, [("bold", "14:00"), ("bold", "15:00")])
        notice = (
            await self.rows("SELECT * FROM notifications WHERE kind='summary' AND status='sent'")
        )[0]
        self.assertEqual(json.loads(notice["parts"]), legacy)
        self.assertEqual(notice["part_index"], 2)
        await self.restart()
        self.worker = NotificationWorker(self.service, self.bot)
        await self.drain()
        self.assertEqual(len(self.bot.sent), 1)
