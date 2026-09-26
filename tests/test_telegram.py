import asyncio
from datetime import UTC, date, datetime, timedelta
from unittest.mock import patch

from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.exceptions import TelegramBadRequest
from aiogram.methods import AnswerCallbackQuery, EditMessageText, SendMessage
from aiogram.types import (
    CallbackQuery,
    Chat,
    InlineKeyboardMarkup,
    Message,
    MessageEntity,
    Update,
    User,
)

from calendar_bot import presentation as view
from calendar_bot.domain import TaskSpec, UserError
from calendar_bot.parser import ParsedEvent, ParsedTask, ParseResult
from calendar_bot.telegram import MENU, BotUI
from tests.support import DatabaseCase, entity_fragments


class RecordingSession(BaseSession):
    def __init__(self, clock):
        super().__init__()
        self.clock = clock
        self.messages = []
        self.edits = []
        self.edit_errors = []

    async def close(self):
        pass

    async def make_request(self, bot, method, timeout=None):
        if isinstance(method, AnswerCallbackQuery):
            return True
        if not isinstance(method, (SendMessage, EditMessageText)):
            raise AssertionError(type(method).__name__)
        editing = isinstance(method, EditMessageText)
        if editing and self.edit_errors:
            raise self.edit_errors.pop(0)
        if method.parse_mode is not None:
            raise AssertionError("Messages must explicitly disable markup parsing")
        entity_fragments(method.text, method.entities)
        if len(method.text.encode("utf-16-le")) // 2 > 4096:
            raise AssertionError("Telegram message is too long")
        if isinstance(method.reply_markup, InlineKeyboardMarkup):
            for row in method.reply_markup.inline_keyboard:
                for button in row:
                    if not 1 <= len(button.callback_data.encode()) <= 64:
                        raise AssertionError("Telegram callback payload is too long")
        message = Message(
            message_id=method.message_id if editing else len(self.messages) + 100,
            chat=Chat(id=method.chat_id, type="private"),
            date=self.clock(),
            text=method.text,
            entities=method.entities,
            reply_markup=method.reply_markup
            if isinstance(method.reply_markup, InlineKeyboardMarkup)
            else None,
        )
        if editing:
            index = next(
                i
                for i, old in enumerate(self.messages)
                if old.message_id == message.message_id and old.chat.id == message.chat.id
            )
            self.messages[index] = message
            self.edits.append(message)
        else:
            self.messages.append(message)
        return message

    async def stream_content(self, url, **kwargs):
        yield b""


class QueueParser:
    def __init__(self):
        self.results = []
        self.calls = []

    async def parse(self, messages, reference, timezone, base=None):
        self.calls.append((list(messages), reference, timezone, base))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def parsed(title="Встреча", **kwargs):
    data = dict(
        kind="event",
        title=title,
        date="2026-09-25",
        time="16:00",
        timezone=None,
        repeat="once",
        weekdays=[],
        reminders=None,
    )
    return ParsedEvent(**(data | kwargs))


class TelegramFlowTests(DatabaseCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.session = RecordingSession(self.clock)
        self.bot = Bot("123456:TEST_TOKEN_NOT_A_REAL_SECRET", session=self.session)
        self.parser = QueueParser()
        self.ui = BotUI(self.service, self.parser, self.bot)
        self.dispatcher = Dispatcher()
        self.dispatcher.include_router(self.ui.router)
        self.update_id = 0

    async def asyncTearDown(self):
        await self.bot.session.close()
        await super().asyncTearDown()

    async def say(self, text, *, uid=101, message_id=None, chat_type="private"):
        self.update_id += 1
        entities = (
            [MessageEntity(type="bot_command", offset=0, length=len(text.split()[0]))]
            if text.startswith("/")
            else []
        )
        message = Message(
            message_id=message_id or self.update_id,
            from_user=User(id=uid, is_bot=False, first_name="Test"),
            chat=Chat(id=uid, type=chat_type),
            date=self.clock(),
            text=text,
            entities=entities,
        )
        await self.dispatcher.feed_update(
            self.bot, Update(update_id=self.update_id, message=message)
        )

    async def press(self, data, *, uid=101, message=None, callback_id=None):
        self.update_id += 1
        message = message or next(
            (m for m in reversed(self.session.messages) if m.chat.id == uid), None
        )
        message = message or Message(
            message_id=100, date=self.clock(), chat=Chat(id=uid, type="private")
        )
        callback = CallbackQuery(
            id=callback_id or f"callback{self.update_id}",
            from_user=User(id=uid, is_bot=False, first_name="Test"),
            chat_instance="test",
            message=message,
            data=data,
        )
        await self.dispatcher.feed_update(
            self.bot, Update(update_id=self.update_id, callback_query=callback)
        )

    async def test_onboarding_requires_timezone_and_keeps_three_defaults(self):
        async with self.store.transaction() as tx:
            await tx.execute("UPDATE users SET timezone=NULL WHERE id=101")
        await self.say("/start")
        self.assertIn("Часовой пояс", self.session.messages[-1].text)
        self.assertEqual(self.parser.calls, [])
        await self.press("z:yerevan")
        await self.say("/settings")
        self.assertIn("За 15, 5 и 1 минуту", self.session.messages[-1].text)

    async def test_tasks_and_events_auto_save_together_with_default_today(self):
        self.parser.results = [
            ParseResult(
                items=[
                    ParsedTask(kind="task", title="Купить <хлеб> & сыр 😀", date=None),
                    parsed("Встреча"),
                    ParsedTask(kind="task", title="Завтра", date="2026-09-25"),
                ],
                question=None,
            )
        ]
        await self.say("купить хлеб; завтра 16:00 встреча; завтра второе дело")
        tasks = await self.rows("SELECT * FROM tasks ORDER BY day")
        self.assertEqual([r["day"] for r in tasks], ["2026-09-24", "2026-09-25"])
        self.assertEqual(len(await self.rows("SELECT * FROM events")), 1)
        self.assertEqual(
            len(await self.rows("SELECT * FROM notifications WHERE kind='reminder'")), 3
        )
        self.assertEqual(
            [r["status"] for r in await self.rows("SELECT status FROM drafts")], ["saved"] * 3
        )
        await self.say("/week")
        self.assertIn("Купить <хлеб> & сыр 😀", self.session.messages[-1].text)
        self.assertIn("Встреча", self.session.messages[-1].text)
        self.assertEqual(len(self.parser.calls), 1)

    async def test_task_clarification_across_midnight_keeps_original_today(self):
        self.clock.now = datetime(2026, 9, 24, 19, 59, tzinfo=UTC)
        self.parser.results = [
            ParseResult(items=[], question="Что купить?"),
            ParseResult(
                items=[ParsedTask(kind="task", title="Купить хлеб", date=None)], question=None
            ),
        ]
        await self.say("купить")
        self.clock.advance(minutes=2)
        await self.say("хлеб")
        self.assertEqual(self.parser.calls[0][1], self.parser.calls[1][1])
        self.assertEqual((await self.rows("SELECT day FROM tasks"))[0]["day"], "2026-09-24")
        self.assertIn("Просрочено", self.session.messages[-1].text)

    async def test_task_buttons_update_only_clicked_message_and_ignore_stale_repeats(self):
        task_id = await self.create(spec=TaskSpec("Проверить 😀", date(2026, 9, 24)))
        await self.say("/today")
        original = self.session.messages[-1]
        data = next(
            b.callback_data
            for row in original.reply_markup.inline_keyboard
            for b in row
            if b.callback_data.startswith("t:done:")
        )
        await self.say("/week")
        other = self.session.messages[-1]
        sent_count = len(self.session.messages)
        await self.press(data, message=original, callback_id="complete-once")
        self.assertEqual(len(self.session.messages), sent_count)
        self.assertEqual(self.session.messages[-1], other)
        self.assertIn("✅ Проверить 😀", self.session.edits[-1].text)
        self.assertEqual((await self.service.task(101, task_id))["version"], 2)
        edits = len(self.session.edits)
        await self.press(data, message=original, callback_id="complete-once")
        self.assertEqual(len(self.session.edits), edits)
        await self.press(data, message=original)
        self.assertEqual((await self.service.task(101, task_id))["version"], 2)
        current = next(m for m in self.session.messages if m.message_id == original.message_id)
        undo = next(
            b.callback_data
            for row in current.reply_markup.inline_keyboard
            for b in row
            if b.callback_data.startswith("t:open:")
        )
        await self.press(undo, message=current)
        self.assertIsNone((await self.service.task(101, task_id))["completed_at"])
        self.assertEqual(self.parser.calls, [])

    async def test_task_foreign_and_malformed_callbacks_never_mutate(self):
        task_id = await self.create(spec=TaskSpec("Личное", date(2026, 9, 24)))
        await self.press(f"t:done:{task_id}:1:list:0", uid=202)
        self.assertIn("не найдено", self.session.messages[-1].text)
        for suffix in ("list:-1", "bad:0", "list:10001"):
            await self.press(f"t:done:{task_id}:1:{suffix}")
        self.assertEqual((await self.service.task(101, task_id))["version"], 1)
        self.assertEqual(self.parser.calls, [])

    async def test_task_edit_and_delete_require_confirmation_and_preserve_completion(self):
        task_id = await self.create(spec=TaskSpec("Старое имя", date(2026, 9, 24)))
        await self.service.set_task_completion(101, task_id, 1, True)
        await self.say("/week")

        def button(prefix):
            return next(
                b.callback_data
                for row in self.session.messages[-1].reply_markup.inline_keyboard
                for b in row
                if b.callback_data.startswith(prefix)
            )

        await self.press(button("t:card:"))
        await self.press(button("t:edit:"))
        self.parser.results = [
            ParseResult(
                items=[ParsedTask(kind="task", title="Новое имя", date="2026-09-26")],
                question=None,
            )
        ]
        await self.say("перенеси на 26 сентября и назови Новое имя")
        self.assertEqual((await self.service.task(101, task_id))["title"], "Старое имя")
        self.assertEqual(self.parser.calls[-1][3]["kind"], "task")
        await self.press(button("d:save:"))
        task = await self.service.task(101, task_id)
        self.assertEqual((task["title"], task["day"]), ("Новое имя", "2026-09-26"))
        self.assertIsNotNone(task["completed_at"])
        await self.press(button("t:delete:"))
        self.assertEqual((await self.service.task(101, task_id))["active"], 1)
        await self.press(button("t:yes:"))
        self.assertEqual((await self.rows("SELECT active FROM tasks"))[0]["active"], 0)
        self.assertNotIn("Новое имя", self.session.messages[-1].text)

    async def test_task_edit_fallback_sends_current_state_without_repeating_mutation(self):
        task_id = await self.create(spec=TaskSpec("Готово", date(2026, 9, 24)))
        await self.say("/week")
        original = self.session.messages[-1]
        data = next(
            b.callback_data
            for row in original.reply_markup.inline_keyboard
            for b in row
            if b.callback_data.startswith("t:done:")
        )
        self.session.edit_errors = [
            TelegramBadRequest(
                method=EditMessageText(chat_id=101, message_id=original.message_id, text="test"),
                message="message can't be edited",
            )
        ]
        count = len(self.session.messages)
        await self.press(data)
        self.assertEqual(len(self.session.messages), count + 1)
        self.assertIn("✅ Готово", self.session.messages[-1].text)
        self.assertEqual((await self.service.task(101, task_id))["version"], 2)

    async def test_task_confirmation_stale_after_completion_shows_current_card(self):
        task_id = await self.create(spec=TaskSpec("Исходное", date(2026, 9, 24)))
        draft = (
            await self.service.create_drafts(
                101,
                "rename-task",
                [TaskSpec("Не применять", date(2026, 9, 25))],
                self.clock(),
                task_id=task_id,
                task_version=1,
            )
        )[0]
        await self.ui.show_draft(101, draft)
        message = self.session.messages[-1]
        await self.service.set_task_completion(101, task_id, 1, True)
        count = len(self.session.messages)
        await self.press(f"d:save:{draft['id']}:1", message=message)
        self.assertEqual(len(self.session.messages), count)
        self.assertIn("Исходное", self.session.edits[-1].text)
        self.assertIn("Выполнено", self.session.edits[-1].text)
        task = await self.service.task(101, task_id)
        self.assertEqual((task["version"], task["day"]), (2, "2026-09-24"))

    async def test_overdue_completion_clamps_page_and_history_keeps_original_date(self):
        for number in range(9):
            await self.create(spec=TaskSpec(f"Прошлое {number}", date(2026, 9, 23)))
        await self.say("/week")
        self.assertEqual(self.session.messages[-1].text.count("Просроченные дела"), 1)
        await self.press("a:a7_20260924:1")
        current = self.session.messages[-1]
        data = next(
            b.callback_data
            for row in current.reply_markup.inline_keyboard
            for b in row
            if b.callback_data.startswith("t:done:")
        )
        await self.press(data)
        self.assertNotIn(
            "Страница →",
            [b.text for row in self.session.messages[-1].reply_markup.inline_keyboard for b in row],
        )
        self.assertIn("1. ☐", self.session.messages[-1].text)
        await self.press("a:a7_20260917:1")
        self.assertIn("23 сентября", self.session.messages[-1].text)
        self.assertIn("✅", self.session.messages[-1].text)

    async def test_summary_link_opens_new_list_without_editing_the_notification(self):
        await self.create(spec=TaskSpec("Сегодня", date(2026, 9, 24)))
        summary = await self.ui.send(101, "☀️ План на сегодня\n\n☐ Сегодня")
        count = len(self.session.messages)
        await self.press("t:show:0", message=summary)
        self.assertEqual(self.session.messages[count - 1], summary)
        self.assertEqual(len(self.session.messages), count + 1)
        self.assertEqual(self.session.edits, [])
        self.assertIn("Сегодня", self.session.messages[-1].text)

    async def test_deleted_task_button_refreshes_list_without_restoring_the_task(self):
        task_id = await self.create(spec=TaskSpec("Удалённое", date(2026, 9, 24)))
        await self.say("/week")
        original = self.session.messages[-1]
        await self.service.cancel_task(101, task_id, 1)
        count = len(self.session.messages)
        await self.press(f"t:done:{task_id}:1:clist:0", message=original)
        self.assertEqual(len(self.session.messages), count)
        self.assertIn("В этом периоде событий и дел нет", self.session.edits[-1].text)
        self.assertEqual((await self.rows("SELECT active FROM tasks"))[0]["active"], 0)

    async def test_not_modified_response_does_not_duplicate_the_interactive_message(self):
        task_id = await self.create(spec=TaskSpec("Готово", date(2026, 9, 24)))
        await self.say("/week")
        original = self.session.messages[-1]
        self.session.edit_errors = [
            TelegramBadRequest(
                method=EditMessageText(chat_id=101, message_id=original.message_id, text="test"),
                message="Bad Request: message is not modified",
            )
        ]
        count = len(self.session.messages)
        await self.press(f"t:done:{task_id}:1:list:0", message=original)
        self.assertEqual(len(self.session.messages), count)
        self.assertEqual((await self.service.task(101, task_id))["version"], 2)

    async def test_task_edit_keeps_page_and_anchors_date_to_input_not_button(self):
        for number in range(9):
            await self.create(spec=TaskSpec(f"Дело {number}", date(2026, 9, 24)))
        await self.say("/week")
        await self.press("a:a7_20260924:1")

        def button(prefix):
            return next(
                b.callback_data
                for row in self.session.messages[-1].reply_markup.inline_keyboard
                for b in row
                if b.callback_data.startswith(prefix)
            )

        await self.press(button("t:card:"))
        self.clock.now = datetime(2026, 9, 24, 19, 59, tzinfo=UTC)
        await self.press(button("t:edit:"))
        self.clock.advance(minutes=2)
        reference = self.clock()
        self.parser.results = [
            ParseResult(
                items=[ParsedTask(kind="task", title="Перенесено", date=None)], question=None
            )
        ]
        await self.say("перенести на сегодня")
        self.assertEqual(self.parser.calls[-1][1], reference)
        self.assertIn("25 сентября", self.session.messages[-1].text)
        count = len(self.session.messages)
        await self.press(button("d:save:"))
        self.assertEqual(len(self.session.messages), count)
        self.assertEqual(button("a:"), "a:a7_20260924:1")
        await self.press(button("a:"))
        self.assertIn("9. ☐ Перенесено", self.session.messages[-1].text)

    async def test_week_navigation_works_in_empty_periods_and_resets_page(self):
        await self.say("/week")
        original = self.session.messages[-1]
        buttons = {
            b.text: b.callback_data for row in original.reply_markup.inline_keyboard for b in row
        }
        self.assertEqual(buttons["← 7 дней"], "a:a7_20260917:0")
        self.assertEqual(buttons["7 дней →"], "a:a7_20261001:0")
        count = len(self.session.messages)
        await self.press(buttons["7 дней →"])
        self.assertEqual(len(self.session.messages), count)
        self.assertIn("1 октября", self.session.messages[-1].text)
        self.assertIn("7 октября", self.session.messages[-1].text)
        self.assertIn("В этом периоде событий и дел нет", self.session.messages[-1].text)
        self.clock.advance(days=1)
        await self.press("a:7:0")
        self.assertIn("25 сентября", self.session.messages[-1].text)
        self.assertEqual(self.parser.calls, [])

    async def test_task_page_and_deletion_keep_selected_dates_after_midnight_and_timezone_change(
        self,
    ):
        self.clock.now = datetime(2026, 9, 24, 19, 59, tzinfo=UTC)
        for number in range(9):
            await self.create(spec=TaskSpec(f"Будущее {number}", date(2026, 10, 3)))
        await self.say("/week")
        await self.press("a:a7_20261001:1")
        source = self.session.messages[-1]
        data = source.reply_markup.inline_keyboard[0][1].callback_data
        self.clock.advance(minutes=2)
        await self.service.preferences(101, timezone="America/New_York")
        await self.press(data)
        card = self.session.messages[-1]
        back = card.reply_markup.inline_keyboard[-1][0].callback_data
        self.assertEqual(back, "a:a7_20261001:1")
        delete = card.reply_markup.inline_keyboard[1][1].callback_data
        await self.press(delete)
        await self.press(self.session.messages[-1].reply_markup.inline_keyboard[0][0].callback_data)
        result = self.session.messages[-1]
        self.assertEqual(result.message_id, source.message_id)
        self.assertIn("1 октября", result.text)
        self.assertIn("7 октября", result.text)
        self.assertIn("1. ☐", result.text)
        self.assertFalse(
            any(b.text == "Страница →" for row in result.reply_markup.inline_keyboard for b in row)
        )

    async def test_far_event_card_edit_and_delete_keep_selected_period(self):
        event_id = await self.create(repeat="daily")
        await self.say("/week")
        before_notices = await self.rows("SELECT * FROM notifications ORDER BY id")
        await self.press("a:a7_20261123:0")

        def button(prefix):
            return next(
                b.callback_data
                for row in self.session.messages[-1].reply_markup.inline_keyboard
                for b in row
                if b.callback_data.startswith(prefix)
            )

        await self.press(button("o:"))
        self.assertEqual(button("a:"), "a:a7_20261123:0")
        self.assertEqual(await self.rows("SELECT * FROM notifications ORDER BY id"), before_notices)
        await self.press(button("x:"))
        self.parser.results = [
            ParseResult(
                items=[parsed("Дальняя встреча", date="2026-12-01", time="13:00")], question=None
            )
        ]
        await self.say("перенеси на 1 декабря в 13:00")
        self.assertEqual(await self.rows("SELECT * FROM exceptions"), [])
        await self.press(button("d:save:"))
        self.assertIn("1 декабря", self.session.messages[-1].text)
        self.assertEqual(button("a:"), "a:a7_20261123:0")
        await self.press(button("a:"))
        self.assertNotIn("Дальняя встреча", self.session.messages[-1].text)
        await self.press(button("o:"))
        await self.press(button("c:"))
        await self.press(button("yes:"))
        self.assertIn("23 ноября", self.session.messages[-1].text)
        rows, _, total = await self.service.plan_page(101, start_day=date(2026, 11, 23))
        self.assertEqual(total, 5)
        self.assertEqual({r["event_id"] for r in rows}, {event_id})

    async def test_legacy_task_navigation_redirects_without_ai_and_new_menu_has_four_buttons(self):
        self.assertEqual(
            [b.text for row in MENU.keyboard for b in row],
            ["Сегодня", "7 дней", "События", "Настройки"],
        )
        self.assertNotIn("/tasks", view.HELP.render()[0])
        for text in ("/tasks", "Дела"):
            await self.say(text)
            self.assertTrue(self.session.messages[-1].text.startswith("📅 7 дней"))
        for data in ("t:list:2", "t:history:1", "a:7:0"):
            await self.press(data)
            self.assertTrue(self.session.messages[-1].text.startswith("📅 7 дней"))
        previous = self.session.messages[-1]
        await self.press("t:show:0")
        self.assertEqual(self.session.messages[-2], previous)
        self.assertTrue(self.session.messages[-1].text.startswith("📅 Сегодня"))
        self.assertEqual(self.parser.calls, [])

    async def test_invalid_period_callbacks_do_not_change_tasks(self):
        task_id = await self.create(spec=TaskSpec("Не менять", date(2026, 9, 24)))
        await self.say("/week")
        for origin in ("a7_20260230", "a2_20260924", "a7_99991231", "a7_nodate"):
            await self.press(f"t:done:{task_id}:1:{origin}:0")
            self.assertEqual((await self.service.task(101, task_id))["version"], 1)
        self.assertEqual(self.parser.calls, [])

    async def test_four_examples_are_saved_automatically_without_duplicate_updates(self):
        self.parser.results = [
            ParseResult(
                question=None,
                items=[
                    parsed(
                        "Content Retro & Planning",
                        date=None,
                        repeat="weekly",
                        weekdays=[0],
                        time="14:00",
                    ),
                    parsed("Оля 1-1", date=None, repeat="weekly", weekdays=[1], time="14:00"),
                    parsed(
                        "Креаторская рассылки",
                        date=None,
                        repeat="weekly",
                        weekdays=[2],
                        time="15:00",
                    ),
                    parsed("Штурм // Ориентир 2027"),
                ],
            )
        ]
        text = "каждый понедельник 14:00 Content Retro & Planning\nкаждый вторник 14:00 Оля 1-1\nкаждую среду 15:00 Креаторская рассылки\nзавтра 16:00 Штурм // Ориентир 2027"
        await self.say(text, message_id=40)
        drafts = await self.rows("SELECT * FROM drafts")
        self.assertEqual(len(drafts), 4)
        self.assertTrue(all(row["status"] == "saved" for row in drafts))
        self.assertEqual(len(await self.rows("SELECT * FROM events")), 4)
        cards = self.session.messages[-4:]
        self.assertTrue(all(card.text.startswith("✅ ") for card in cards))
        for card in cards:
            buttons = [b for row in card.reply_markup.inline_keyboard for b in row]
            self.assertFalse(any(b.text == "Сохранить" for b in buttons))
            self.assertTrue(any(b.text.startswith("Удалить") for b in buttons))
        for draft in drafts:
            self.assertTrue(await self.pending(draft["result_id"]))
        await self.say(text, message_id=40)
        self.assertEqual(len(self.parser.calls), 1)
        self.assertEqual(len(await self.rows("SELECT * FROM events")), 4)
        self.assertFalse(any("Не удалось" in m.text for m in self.session.messages))

    async def test_unknown_user_and_group_never_reach_parser(self):
        await self.say("/start", uid=999)
        self.assertIn("999", self.session.messages[-1].text)
        count = len(self.session.messages)
        await self.say("завтра 16:00 встреча", chat_type="group")
        self.assertEqual(len(self.session.messages), count)
        self.assertEqual(self.parser.calls, [])
        self.assertEqual(len(await self.rows("SELECT * FROM users")), 2)

    async def test_callback_cannot_access_another_users_draft(self):
        draft = (await self.service.create_drafts(101, "draft", [self.spec()], self.clock()))[0]
        await self.press(f"d:save:{draft['id']}:1", uid=202)
        self.assertEqual(await self.rows("SELECT * FROM events"), [])
        self.assertIn("не найден", self.session.messages[-1].text)

    async def test_clarification_keeps_original_date_across_midnight(self):
        self.clock.now = datetime(2026, 9, 24, 19, 59, tzinfo=UTC)
        self.parser.results = [
            ParseResult(items=[], question="Во сколько?"),
            ParseResult(items=[parsed()], question=None),
        ]
        await self.say("завтра встреча")
        self.assertEqual(await self.rows("SELECT * FROM events"), [])
        self.clock.advance(minutes=2)
        await self.say("16:00")
        self.assertEqual(self.parser.calls[0][1], self.parser.calls[1][1])
        self.assertEqual(len(await self.rows("SELECT * FROM drafts")), 1)
        self.assertEqual(len(await self.rows("SELECT * FROM events")), 1)
        self.assertIn("пт, 25 сентября · 16:00", self.session.messages[-1].text)

    async def test_api_outage_keeps_input_and_retry_creates_one_draft(self):
        self.parser.results = [
            UserError("Сервис временно недоступен."),
            ParseResult(items=[parsed()], question=None),
        ]
        await self.say("завтра 16:00 встреча")
        self.assertEqual((await self.service.conversation(101))["mode"], "parse")
        self.assertEqual(await self.rows("SELECT * FROM drafts"), [])
        await self.press("p:retry")
        self.assertEqual(len(await self.rows("SELECT * FROM drafts")), 1)
        self.assertEqual(len(await self.rows("SELECT * FROM events")), 1)

    async def test_edit_and_single_occurrence_cancellation_via_buttons(self):
        event_id = await self.create(repeat="daily")
        await self.say("/week")
        buttons = [b for row in self.session.messages[-1].reply_markup.inline_keyboard for b in row]
        await self.press(buttons[0].callback_data)
        self.assertIn("Эта встреча", self.session.messages[-1].text)
        original = self.spec().day.strftime("%Y%m%d")
        await self.press(f"x:{event_id}:1:{original}")
        self.parser.results = [ParseResult(items=[parsed("Перенос", time="18:00")], question=None)]
        await self.say("перенеси на завтра 18:00")
        self.assertIn("Изменение одной встречи", self.session.messages[-1].text)
        draft = (
            await self.rows(
                "SELECT * FROM drafts WHERE event_id=? ORDER BY rowid DESC", (event_id,)
            )
        )[0]
        await self.press(f"d:save:{draft['id']}:1")
        event = await self.service.event(101, event_id)
        await self.press(f"c:{event_id}:{event['version']}:{original}")
        self.assertIn("только эту встречу", self.session.messages[-1].text)
        await self.press(f"yes:{event_id}:{event['version']}:{original}")
        self.assertEqual(
            (await self.rows("SELECT spec FROM exceptions WHERE event_id=?", (event_id,)))[0][
                "spec"
            ],
            None,
        )
        self.assertEqual((await self.service.event(101, event_id))["active"], 1)

    async def test_settings_and_custom_event_reminders_are_confirmed(self):
        event_id = await self.create()
        await self.press("s:reminders")
        await self.say("30, 10")
        self.assertIn("За 30 и 10 минут", self.session.messages[-1].text)
        await self.press(f"r:{event_id}:1:all")
        await self.say("нет")
        self.assertIn("🔔 Выключены", self.session.messages[-1].text)
        self.assertIn("Напоминания для этого события.", self.session.messages[-1].text)
        draft = (await self.rows("SELECT * FROM drafts WHERE event_id=?", (event_id,)))[0]
        self.assertTrue(await self.pending(event_id))
        await self.press(f"d:save:{draft['id']}:1")
        self.assertEqual(await self.pending(event_id), [])

    async def test_literal_title_survives_auto_save_and_edits_still_need_confirmation(self):
        title = "🧑‍💻 Retro & <b>Planning</b> _2027_ > 🎯"
        self.bot.default.parse_mode = "HTML"
        self.parser.results = [ParseResult(items=[parsed(title)], question=None)]
        await self.say("завтра 16:00 " + title)
        card = self.session.messages[-1]
        fragments = entity_fragments(card.text, card.entities)
        self.assertIn(("bold", title), fragments)
        self.assertIn(("bold", "16:00"), fragments)
        self.assertIn(("italic", "Ереван · UTC+4"), fragments)
        self.assertNotIn("один раз", card.text)
        self.assertIn("Напоминания из ваших настроек.", card.text)
        buttons = [b for row in card.reply_markup.inline_keyboard for b in row]
        self.assertEqual([b.text for b in buttons], ["Изменить", "Напоминания", "Удалить событие"])
        self.assertTrue(card.text.startswith("✅ Событие сохранено"))
        self.assertNotIn("Черновик", card.text)
        event = (await self.rows("SELECT * FROM events"))[0]
        original = event["spec"]
        await self.press(buttons[0].callback_data)
        self.parser.results = [ParseResult(items=[parsed(title, time="17:00")], question=None)]
        await self.say("перенеси на 17:00")
        self.assertEqual((await self.service.event(101, event["id"]))["spec"], original)
        change = self.session.messages[-1]
        save = change.reply_markup.inline_keyboard[0][0]
        self.assertEqual(save.text, "Сохранить")
        await self.press(save.callback_data)
        saved = self.session.messages[-1]
        self.assertIn(("bold", title), entity_fragments(saved.text, saved.entities))
        self.assertIn(("bold", "17:00"), entity_fragments(saved.text, saved.entities))
        self.assertEqual(self.session.messages[-2].text, "✅ Сохранено.")
        self.assertEqual(len(await self.rows("SELECT * FROM events")), 1)
        await self.ui.send(101, title)
        self.assertEqual(self.session.messages[-1].text, title)
        self.assertFalse(self.session.messages[-1].entities)

    async def test_auto_saved_event_can_be_deleted_with_its_reminders(self):
        self.parser.results = [ParseResult(items=[parsed()], question=None)]
        await self.say("завтра 16:00 встреча")
        event = (await self.rows("SELECT * FROM events"))[0]
        card = self.session.messages[-1]
        delete = card.reply_markup.inline_keyboard[-1][0]
        self.assertEqual(delete.text, "Удалить событие")
        await self.press(delete.callback_data, uid=202)
        self.assertIn("не найдено", self.session.messages[-1].text)
        await self.press(delete.callback_data)
        self.assertTrue(await self.pending(event["id"]))
        confirm = self.session.messages[-1].reply_markup.inline_keyboard[0][0]
        self.assertEqual(confirm.text, "Да, удалить")
        await self.press(confirm.callback_data)
        self.assertEqual(await self.pending(event["id"]), [])
        self.assertEqual(await self.rows("SELECT * FROM events WHERE active=1"), [])
        self.assertEqual(
            self.session.messages[-1].text, "✅ Событие и его будущие напоминания удалены."
        )

    async def test_auto_save_retry_after_restart_does_not_parse_or_save_twice(self):
        self.parser.results = [
            ParseResult(items=[parsed("Первая"), parsed("Вторая")], question=None)
        ]
        with patch.object(
            self.ui, "show_results", side_effect=RuntimeError("Interrupted after commit")
        ):
            await self.say("завтра 16:00 две встречи", message_id=40)
        events = await self.rows("SELECT * FROM events ORDER BY id")
        notices = await self.rows("SELECT * FROM notifications ORDER BY id")
        self.assertEqual(len(events), 2)
        await self.restart()
        self.ui = BotUI(self.service, self.parser, self.bot)
        self.dispatcher = Dispatcher()
        self.dispatcher.include_router(self.ui.router)
        await self.press("p:retry")
        self.assertTrue(self.session.messages[-1].text.startswith("✅ Событие сохранено"))
        await self.say("завтра 16:00 две встречи", message_id=40)
        self.assertEqual(len(self.parser.calls), 1)
        self.assertEqual(await self.rows("SELECT * FROM events ORDER BY id"), events)
        self.assertEqual(await self.rows("SELECT * FROM notifications ORDER BY id"), notices)
        self.assertIsNone(await self.service.conversation(101))

    async def test_invalid_event_in_batch_requires_clarification_before_saving_anything(self):
        self.parser.results = [
            ParseResult(
                items=[parsed("Будущее"), parsed("Прошлое", date="2026-09-23")], question=None
            )
        ]
        await self.say("две встречи")
        self.assertEqual(await self.rows("SELECT * FROM events"), [])
        self.assertEqual(await self.rows("SELECT * FROM drafts"), [])
        self.assertIn("Уточните запись", self.session.messages[-1].text)

    async def test_clarification_stays_literal_in_ui_and_model_history(self):
        question = "Когда обсудим <план> & _бюджет_? 😀"
        self.parser.results = [ParseResult(items=[], question=question)]
        await self.say("обсудить план")
        message = self.session.messages[-1]
        self.assertIn(question, message.text)
        self.assertNotIn(
            question, [value for _, value in entity_fragments(message.text, message.entities)]
        )
        state = await self.service.conversation(101)
        self.assertEqual(
            state["payload"]["messages"][-1], {"role": "assistant", "content": question}
        )

    async def test_week_groups_dates_and_preserves_pagination_and_event_numbers(self):
        await self.create(title="Уже началось")
        for index in range(9):
            await self.create(
                title=f"Встреча {index}", when=self.clock() + timedelta(days=1, minutes=index)
            )
        self.clock.advance(hours=2)
        await self.say("/week")
        first = self.session.messages[-1]
        fragments = entity_fragments(first.text, first.entities)
        self.assertIn(("bold", "чт, 24 сентября"), fragments)
        self.assertIn(("bold", "пт, 25 сентября"), fragments)
        self.assertIn(("italic", "уже началось"), fragments)
        self.assertEqual(first.text.count("пт, 25 сентября"), 1)
        self.assertEqual(first.text.count("Ереван · UTC+4"), 1)
        buttons = [b for row in first.reply_markup.inline_keyboard for b in row]
        self.assertFalse(any(b.text.startswith("1.") for b in buttons))
        await self.press(next(b.callback_data for b in buttons if b.text == "Страница →"))
        second = self.session.messages[-1]
        self.assertIn("9. 12:07 — Встреча 7", second.text)
        self.assertIn("10. 12:08 — Встреча 8", second.text)
        self.assertIn(
            "a:a7_20260924:0",
            [b.callback_data for row in second.reply_markup.inline_keyboard for b in row],
        )

    async def test_series_card_list_and_edit_have_clear_scope(self):
        event_id = await self.create(repeat="weekly", weekdays=(3,), timezone="Europe/Moscow")
        await self.say("/events")
        listing = self.session.messages[-1]
        self.assertIn(("bold", "Встреча"), entity_fragments(listing.text, listing.entities))
        self.assertIn("По четвергам · 12:00", listing.text)
        self.assertIn("Расписание: Москва · UTC+3", listing.text)
        self.assertIn("Ваш пояс: Ереван · UTC+4", listing.text)
        await self.press(f"e:{event_id}:1")
        self.assertIn("Все будущие встречи", self.session.messages[-1].text)
        self.assertIn(
            "Расписание: чт, 24 сентября · 12:00 · Москва · UTC+3", self.session.messages[-1].text
        )
        await self.press(f"r:{event_id}:1:all")
        await self.say("30, 0")
        self.assertIn("Изменение всех будущих встреч", self.session.messages[-1].text)
        self.assertIn("За 30 минут; в момент начала", self.session.messages[-1].text)
        await self.press(f"x:{event_id}:1:all")
        self.parser.results = [ParseResult(items=[parsed(repeat="once")], question=None)]
        await self.say("сделай разовой завтра в 16:00")
        self.assertIn("Изменение всех будущих встреч", self.session.messages[-1].text)

    async def test_help_has_code_examples_and_settings_have_sections(self):
        await self.say("/help")
        message = self.session.messages[-1]
        self.assertIn(
            ("code", "каждый понедельник 14:00 Content Retro & Planning"),
            entity_fragments(message.text, message.entities),
        )
        await self.say("/settings")
        message = self.session.messages[-1]
        self.assertIn(("bold", "Утренняя сводка"), entity_fragments(message.text, message.entities))

    async def test_shutdown_cancels_stalled_parser_before_storage_is_closed(self):
        entered = asyncio.Event()

        async def stalled(*args, **kwargs):
            entered.set()
            await asyncio.Event().wait()

        self.parser.parse = stalled
        update = asyncio.create_task(self.say("завтра 16:00 встреча"))
        await asyncio.wait_for(entered.wait(), timeout=5)
        await self.ui.shutdown(timeout=0)
        self.assertTrue(update.cancelled())
        self.assertEqual(self.ui.middleware.pending, set())
        self.assertEqual(await self.rows("SELECT * FROM events"), [])
