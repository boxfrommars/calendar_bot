import asyncio
import json
import logging
import re
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

from aiogram import BaseMiddleware, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from aiogram.utils.formatting import Bold, Code, Italic, Text

from . import presentation as view
from .domain import (
    AgendaPeriod,
    EventSpec,
    TaskSpec,
    UserError,
    local_day,
    parse_reminders,
    parse_time,
    read_spec,
)
from .parser import Parser, normalize
from .service import CalendarService

log = logging.getLogger(__name__)
MENU = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Сегодня"), KeyboardButton(text="7 дней")],
        [KeyboardButton(text="События"), KeyboardButton(text="Настройки")],
    ],
    resize_keyboard=True,
)
ZONES = {"moscow": "Europe/Moscow", "yerevan": "Asia/Yerevan", "utc": "UTC"}


def period_key(period: AgendaPeriod) -> str:
    return f"a{period.days}_{period.start.strftime('%Y%m%d')}"


def read_period(key: str | None, today: date) -> AgendaPeriod:
    if key in {None, "list", "history", "a7", "7"}:
        return AgendaPeriod(today, 7)
    if key in {"a1", "1"}:
        return AgendaPeriod(today, 1)
    match = re.fullmatch(r"a([17])_(\d{8})", key)
    if not match:
        raise UserError("Некорректный период. Откройте /week.")
    try:
        return AgendaPeriod(date.fromisoformat(match[2]), int(match[1]))
    except ValueError, OverflowError:
        raise UserError("Некорректный период. Откройте /week.") from None


def keyboard(*rows: list[tuple[str, str]]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=text, callback_data=data) for text, data in row]
            for row in rows
        ]
    )


def timezone_keyboard() -> InlineKeyboardMarkup:
    return keyboard(
        [("Москва", "z:moscow"), ("Ереван", "z:yerevan")],
        [("UTC", "z:utc"), ("Другой часовой пояс", "z:custom")],
    )


class AccessMiddleware(BaseMiddleware):
    def __init__(self, service: CalendarService, bot):
        self.service, self.bot = service, bot
        self.locks = {user: asyncio.Lock() for user in service.allowed}
        self.pending: set[asyncio.Task] = set()

    async def __call__(self, handler, event, data):
        task = asyncio.current_task()
        self.pending.add(task)
        try:
            return await self.handle(handler, event, data)
        finally:
            self.pending.discard(task)

    async def handle(self, handler, event, data):
        person = event.from_user
        message = event.message if isinstance(event, CallbackQuery) else event
        if not person or not message or message.chat.type != "private":
            if isinstance(event, CallbackQuery):
                await event.answer("Бот работает только в личном чате.")
            return
        uid = person.id
        if uid not in self.service.allowed:
            if isinstance(event, CallbackQuery):
                await event.answer("Доступ не разрешён.")
            else:
                await self.bot.send_message(
                    chat_id=message.chat.id,
                    **Text(
                        "⚠️ Доступ по списку пользователей.\n\n",
                        Bold("Ваш Telegram ID: "),
                        Code(str(uid)),
                        "\nПередайте его владельцу бота.",
                    ).as_kwargs(),
                )
            return
        if isinstance(event, CallbackQuery):
            try:
                await event.answer()
            except TelegramBadRequest:
                pass
            key = "callback:" + event.id
        else:
            key = f"message:{uid}:{event.message_id}"
        async with self.locks[uid]:
            await self.service.ensure_user(uid, message.chat.id)
            if await self.service.processed(key):
                return
            try:
                result = await handler(event, data)
            except UserError as exc:
                await self.bot.send_message(
                    chat_id=message.chat.id, **view.status(str(exc), error=True).as_kwargs()
                )
                result = None
            except Exception as exc:
                # Provider exception strings can contain request bodies or URLs with tokens.
                log.error("update_failed type=%s user_id=%s", type(exc).__name__, uid)
                await self.bot.send_message(
                    chat_id=message.chat.id,
                    **view.status(
                        "Не удалось обработать запрос. Попробуйте ещё раз или откройте /events.",
                        error=True,
                    ).as_kwargs(),
                )
                return None
            await self.service.mark_processed(key)
            return result


class BotUI:
    def __init__(self, service: CalendarService, parser: Parser, bot):
        self.service, self.parser, self.bot = service, parser, bot
        self.router = Router(name="calendar")
        self.parse_times: dict[int, list[datetime]] = {}
        self.middleware = AccessMiddleware(service, bot)
        self.router.message.outer_middleware(self.middleware)
        self.router.callback_query.outer_middleware(self.middleware)
        self.router.message.register(self.start, CommandStart())
        self.router.message.register(
            self.command,
            Command("today", "week", "events", "tasks", "settings", "help", "cancel", "id"),
        )
        self.router.callback_query.register(self.callback)
        self.router.message.register(self.message)

    async def shutdown(self, timeout=45):
        # Polling has stopped; finish acknowledged updates before closing the DB/SDK.
        await asyncio.sleep(0)
        tasks = self.middleware.pending - {asyncio.current_task()}
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=timeout)
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send(self, uid: int, text: str | Text, markup=None):
        user = await self.service.user(uid)
        content = text if isinstance(text, Text) else Text(text)
        return await self.bot.send_message(
            chat_id=user["chat_id"], **content.as_kwargs(), reply_markup=markup
        )

    async def present(self, uid: int, text: Text, markup, message=None):
        """Only the clicked interactive message is updated; other history is untouched."""
        if message is not None:
            user = await self.service.user(uid)
            kwargs = text.as_kwargs()
            if (
                getattr(message, "text", None) == kwargs["text"]
                and getattr(message, "entities", None) == kwargs["entities"]
                and getattr(message, "reply_markup", None) == markup
            ):
                return
            try:
                return await self.bot.edit_message_text(
                    chat_id=user["chat_id"],
                    message_id=message.message_id,
                    **kwargs,
                    reply_markup=markup,
                )
            except TelegramBadRequest as exc:
                if "message is not modified" in exc.message:
                    return
        return await self.send(uid, text, markup)

    async def ready(self, uid: int) -> dict:
        user = await self.service.user(uid)
        if not user["timezone"]:
            raise UserError("Сначала выберите часовой пояс через /start.")
        return user

    async def start(self, message: Message):
        uid = message.from_user.id
        user = await self.service.ensure_user(uid, message.chat.id, resume=True)
        await self.service.set_conversation(uid, None)
        await self.send(
            uid,
            view.WELCOME,
            MENU,
        )
        if not user["timezone"]:
            await self.service.set_conversation(uid, "timezone")
            await self.send(
                uid,
                view.prompt("Часовой пояс", "Выберите пояс, в котором указываете время встреч."),
                timezone_keyboard(),
            )
        else:
            await self.send(
                uid,
                Text(
                    Bold("Ваш часовой пояс"),
                    "\n",
                    view.timezone_label(user["timezone"], self.service.clock()),
                    "\n\nНапишите событие или дело либо откройте /help.",
                ),
            )

    async def command(self, message: Message):
        uid = message.from_user.id
        command = message.text.split()[0].split("@")[0][1:]
        await self.service.set_conversation(uid, None)
        if command in {"today", "week"}:
            await self.show_agenda(uid, 1 if command == "today" else 7)
        elif command == "events":
            await self.show_events(uid)
        elif command == "tasks":
            await self.send(uid, view.status("Расписание доступно в «Сегодня» и «7 дней»."), MENU)
            await self.show_agenda(uid, 7)
        elif command == "settings":
            await self.show_settings(uid)
        elif command == "id":
            await self.send(uid, Text(Bold("Ваш Telegram ID: "), Code(str(uid))))
        elif command == "cancel":
            await self.send(
                uid,
                view.status("Ввод отменён. Сохранённые записи доступны в /today и /week."),
                MENU,
            )
        else:
            await self.send(uid, view.HELP, MENU)

    async def show_settings(self, uid):
        user = await self.service.user(uid)
        await self.send(
            uid,
            view.settings(user, self.service.clock()),
            keyboard(
                [("Часовой пояс", "s:timezone")],
                [("Напоминания", "s:reminders")],
                [("Время сводки", "s:summary_time")],
                [
                    (
                        "Выключить сводку" if user["summary_enabled"] else "Включить сводку",
                        f"s:enabled:{0 if user['summary_enabled'] else 1}",
                    )
                ],
                [
                    (
                        "Пропускать пустые дни"
                        if user["summary_empty"]
                        else "Сообщать о пустом дне",
                        f"s:empty:{0 if user['summary_empty'] else 1}",
                    )
                ],
            ),
        )

    async def period(self, uid, origin=None):
        user = await self.ready(uid)
        return read_period(origin, local_day(self.service.clock(), user["timezone"]))

    async def show_agenda(self, uid, days, page=0, *, start_day=None, message=None):
        user = await self.ready(uid)
        now = self.service.clock()
        period = AgendaPeriod(start_day or local_day(now, user["timezone"]), days)
        rows, page, total = await self.service.plan_page(
            uid, days=days, start_day=period.start, page=page
        )
        scope = period_key(period)
        buttons = []
        for number, row in enumerate(rows, page * 8 + 1):
            if row["kind"] == "task":
                completed = row["completed_at"] is not None
                suffix = f"{row['id']}:{row['version']}:{scope}:{page}"
                buttons.append(
                    [
                        (
                            f"{'✅' if completed else '☐'} {number}. {row['title'][:38]}",
                            f"t:{'open' if completed else 'done'}:{suffix}",
                        ),
                        ("Карточка", f"t:card:{suffix}"),
                    ]
                )
                continue
            spec = EventSpec.from_json(row["spec"])
            instant = datetime.fromtimestamp(row["start_at"], UTC)
            if instant > now and row["active"]:
                day_key = row["original_day"].replace("-", "")
                buttons.append(
                    [
                        (
                            f"{number}. {spec.title[:38]}",
                            f"o:{row['event_id']}:{row['version']}:{day_key}:{scope}:{page}",
                        )
                    ]
                )
        navigation = []

        def destination(selected):
            return f"a:{scope}:{selected}"

        if page:
            navigation.append(("← Страница", destination(page - 1)))
        if (page + 1) * 8 < total:
            navigation.append(("Страница →", destination(page + 1)))
        if navigation:
            buttons.append(navigation)
        if days == 7:
            periods = []
            for direction, label in ((-1, "← 7 дней"), (1, "7 дней →")):
                try:
                    adjacent = period.shifted(direction)
                except UserError:
                    continue
                periods.append((label, f"a:{period_key(adjacent)}:0"))
            if periods:
                buttons.append(periods)
            buttons.append([("К ближайшим 7 дням", "a:7:0")])
        await self.present(
            uid,
            view.agenda(rows, user["timezone"], now, days, page, start_day=period.start),
            keyboard(*buttons) if buttons else None,
            message,
        )

    async def show_events(self, uid, page=0):
        user = await self.ready(uid)
        if not 0 <= page <= 10000:
            raise UserError("Некорректная страница.")
        rows = await self.service.event_list(uid, offset=page * 6, limit=7)
        buttons = []
        for number, row in enumerate(rows[:6], page * 6 + 1):
            spec = EventSpec.from_json(row["spec"])
            buttons.append([(f"{number}. {spec.title[:38]}", f"e:{row['id']}:{row['version']}")])
        navigation = []
        if page:
            navigation.append(("← Назад", f"l:{page - 1}"))
        if len(rows) > 6:
            navigation.append(("Далее →", f"l:{page + 1}"))
        if navigation:
            buttons.append(navigation)
        await self.send(
            uid,
            view.event_list(rows[:6], user["timezone"], self.service.clock(), page),
            keyboard(*buttons) if buttons else None,
        )

    async def show_event(
        self, uid, event_id, version, day=None, *, saved=False, origin=None, page=0, message=None
    ):
        user = await self.ready(uid)
        if origin is not None:
            origin = period_key(await self.period(uid, origin))
        context = f":{origin}:{page}" if origin is not None else ""
        event = await self.service.event(uid, event_id, version)
        series = EventSpec.from_json(event["spec"])
        is_series = series.repeat != "once" or event["has_exceptions"]
        if day:
            occurrence = await self.service.occurrence(uid, event_id, day, version)
            spec = EventSpec.from_json(occurrence["spec"])
            instant = datetime.fromtimestamp(occurrence["start_at"], UTC)
        else:
            spec = series
            instant = await self.service.upcoming(uid, event_id)
        offsets = (
            spec.reminder_minutes
            if spec.reminder_minutes is not None
            else json.loads(user["reminders"])
        )
        suffix = day.replace("-", "") if day else "all"
        label = (
            "Эта встреча"
            if day and is_series
            else "Событие"
            if not is_series
            else "Все будущие встречи"
        )
        if saved:
            label = "Серия сохранена" if is_series and not day else "Событие сохранено"
        buttons = [
            [("Изменить", f"x:{event_id}:{version}:{suffix}{context}")],
            [("Напоминания", f"r:{event_id}:{version}:{suffix}{context}")],
            [
                (
                    "Удалить встречу"
                    if day
                    else "Удалить всю серию"
                    if is_series
                    else "Удалить событие",
                    f"c:{event_id}:{version}:{suffix}{context}",
                )
            ],
        ]
        if day and is_series:
            buttons.append([("Вся серия →", f"e:{event_id}:{version}:all{context}")])
        if origin is not None:
            buttons.append([("К списку", f"a:{origin}:{page}")])
        await self.present(
            uid,
            view.event_card(
                label,
                spec,
                instant,
                user["timezone"],
                offsets,
                self.service.clock(),
                recurrence=series,
                single=bool(day and is_series),
                saved=saved,
            ),
            keyboard(*buttons),
            message,
        )

    async def show_task(
        self, uid, task_id, version=None, *, saved=False, message=None, origin=None, page=0
    ):
        user = await self.ready(uid)
        origin = period_key(read_period(origin, local_day(self.service.clock(), user["timezone"])))
        row = await self.service.task(uid, task_id, version)
        suffix = f"{task_id}:{row['version']}:c{origin}:{page}"
        completed = row["completed_at"] is not None
        back = f"a:{origin}:{page}"
        await self.present(
            uid,
            view.task_card(
                TaskSpec.from_dict(row),
                local_day(self.service.clock(), user["timezone"]),
                completed=completed,
                saved=saved,
            ),
            keyboard(
                [
                    (
                        "Вернуть в работу" if completed else "Выполнить",
                        f"t:{'open' if completed else 'done'}:{suffix}",
                    )
                ],
                [("Изменить", f"t:edit:{suffix}"), ("Удалить", f"t:delete:{suffix}")],
                [("К списку", back)],
            ),
            message,
        )

    async def task_source(self, uid, scope, page, task_id, message):
        if scope and scope.startswith("c"):
            await self.show_task(uid, task_id, message=message, origin=scope[1:], page=page)
        else:
            period = await self.period(uid, scope)
            await self.show_agenda(uid, period.days, page, start_day=period.start, message=message)

    async def task_callback(self, uid, values, message):
        action = values[1]
        if action in {"list", "history", "show"} and len(values) == 3:
            if not 0 <= int(values[2]) <= 10000:
                raise UserError("Некорректная страница.")
            await self.show_agenda(
                uid,
                1 if action == "show" else 7,
                message=None if action == "show" else message,
            )
            return
        if len(values) != 6 or action not in {"card", "done", "open", "edit", "delete", "yes"}:
            raise UserError("Кнопка устарела. Откройте /week.")
        task_id, version, scope, page = values[2], int(values[3]), values[4], int(values[5])
        if version < 1 or not 0 <= page <= 10000:
            raise UserError("Некорректная кнопка дела. Откройте /week.")
        await self.period(uid, scope.removeprefix("c"))
        row = await self.service.task(uid, task_id, include_deleted=True)
        origin = scope.removeprefix("c")
        if not row["active"]:
            await self.task_source(uid, origin, page, task_id, message)
            return
        if row["version"] != version:
            await self.task_source(uid, scope, page, task_id, message)
            return
        if action == "card":
            await self.show_task(uid, task_id, version, message=message, origin=origin, page=page)
        elif action in {"done", "open"}:
            await self.service.set_task_completion(uid, task_id, version, action == "done")
            await self.task_source(uid, scope, page, task_id, message)
        elif action == "delete":
            await self.present(
                uid,
                Text("⚠️ Удалить дело?\n\n", Bold(row["title"])),
                keyboard(
                    [("Да, удалить", f"t:yes:{task_id}:{version}:{scope}:{page}")],
                    [("Вернуться", f"t:card:{task_id}:{version}:{origin}:{page}")],
                ),
                message,
            )
        elif action == "yes":
            await self.service.cancel_task(uid, task_id, version)
            await self.task_source(uid, origin, page, task_id, message)
        elif action == "edit":
            await self.service.set_conversation(
                uid,
                "edit",
                {
                    "base": {"kind": "task", **TaskSpec.from_dict(row).to_dict()},
                    "task_id": task_id,
                    "task_version": version,
                    "origin": origin,
                    "page": page,
                },
            )
            await self.send(
                uid,
                view.prompt(
                    "Что изменить в деле?",
                    "Напишите новое название или дату.",
                    "После разбора я покажу карточку для подтверждения.",
                    "",
                    Italic("/cancel — выйти."),
                ),
            )

    async def show_results(self, uid, rows, *, origin=None, page=0):
        await self.service.set_conversation(uid, None)
        for row in rows:
            if row["status"] == "saved":
                if row["kind"] == "task":
                    try:
                        await self.show_task(
                            uid, row["result_id"], saved=True, origin=origin, page=page
                        )
                    except UserError:
                        await self.send(
                            uid,
                            view.status(
                                "Дело уже было сохранено и затем удалено. Текущие дела: /week."
                            ),
                        )
                    continue
                event = await self.service.event(uid, row["result_id"])
                await self.show_event(
                    uid,
                    event["id"],
                    event["version"],
                    row["original_day"],
                    saved=True,
                    origin=origin,
                    page=page,
                )
            else:
                await self.show_draft(uid, row, origin=origin, page=page)

    async def show_draft(self, uid, row, *, origin=None, page=0, message=None):
        if row["status"] != "pending":
            await self.send(
                uid,
                view.status(
                    "Эта карточка уже сохранена или отменена. Записи доступны в /today и /week."
                ),
            )
            return
        user = await self.ready(uid)
        if row["kind"] == "task":
            origin = period_key(
                read_period(origin, local_day(self.service.clock(), user["timezone"]))
            )
            suffix = f"{row['id']}:{row['version']}:{origin}:{page}"
            completed = False
            if row["task_id"]:
                task = await self.service.task(uid, row["task_id"])
                completed = task["completed_at"] is not None
            await self.present(
                uid,
                view.task_card(
                    TaskSpec.from_json(row["spec"]),
                    local_day(self.service.clock(), user["timezone"]),
                    completed=completed,
                    draft=True,
                ),
                keyboard(
                    [("Сохранить", f"d:save:{suffix}")],
                    [
                        ("Исправить", f"d:edit:{suffix}"),
                        ("Отмена", f"d:cancel:{suffix}"),
                    ],
                ),
                message,
            )
            return
        spec = EventSpec.from_json(row["spec"])
        instant = spec.first_after(self.service.clock())
        offsets = (
            spec.reminder_minutes
            if spec.reminder_minutes is not None
            else json.loads(user["reminders"])
        )
        is_series = spec.repeat != "once"
        if row["event_id"] and not row["original_day"]:
            event = await self.service.event(uid, row["event_id"])
            is_series = (
                is_series
                or EventSpec.from_json(event["spec"]).repeat != "once"
                or bool(event["has_exceptions"])
            )
        heading = (
            "Изменение одной встречи"
            if row["original_day"]
            else "Изменение всех будущих встреч"
            if row["event_id"] and is_series
            else "Изменение события"
            if row["event_id"]
            else "Новое событие"
        )
        suffix = f"{row['id']}:{row['version']}"
        if origin is not None:
            suffix += f":{period_key(await self.period(uid, origin))}:{page}"
        await self.present(
            uid,
            view.event_card(
                heading,
                spec,
                instant,
                user["timezone"],
                offsets,
                self.service.clock(),
                draft=True,
                single=bool(row["original_day"]),
            ),
            keyboard(
                [("Сохранить", f"d:save:{suffix}")],
                [
                    ("Исправить", f"d:edit:{suffix}"),
                    ("Отмена", f"d:cancel:{suffix}"),
                ],
            ),
            message,
        )

    async def callback(self, query: CallbackQuery):
        uid = query.from_user.id
        values = (query.data or "").split(":")
        try:
            await self._callback(uid, values, query)
        except (IndexError, ValueError, OverflowError) as exc:
            if isinstance(exc, UserError):
                raise
            raise UserError("Кнопка устарела. Откройте /events или /settings.") from None

    async def _callback(self, uid, values, query):
        action = values[0]
        if action == "z":
            if values[1] == "custom":
                await self.service.set_conversation(uid, "timezone")
                await self.send(
                    uid,
                    view.prompt(
                        "Другой часовой пояс",
                        Text(
                            "Введите IANA-пояс, например ",
                            Code("Europe/Berlin"),
                            ", ",
                            Code("Asia/Yerevan"),
                            " или ",
                            Code("Europe/Moscow"),
                            ".",
                        ),
                    ),
                )
            elif values[1] in ZONES:
                await self.service.preferences(uid, timezone=ZONES[values[1]])
                await self.service.set_conversation(uid, None)
                await self.send(
                    uid,
                    view.status(
                        "Часовой пояс сохранён. Существующие события сохраняют свой пояс.\n\n"
                        "Теперь можно написать событие. Примеры: /help."
                    ),
                    MENU,
                )
            return
        if action == "s":
            setting = values[1]
            if setting == "timezone":
                await self.service.set_conversation(uid, "timezone")
                await self.send(
                    uid,
                    view.prompt(
                        "Часовой пояс", "Выберите пояс для новых событий и утренней сводки."
                    ),
                    timezone_keyboard(),
                )
            elif setting in {"enabled", "empty"}:
                enabled = int(values[2])
                if enabled not in {0, 1}:
                    raise UserError("Некорректная настройка.")
                await self.service.preferences(
                    uid, **{"summary_enabled" if setting == "enabled" else "summary_empty": enabled}
                )
                await self.show_settings(uid)
            elif setting in {"reminders", "summary_time"}:
                await self.service.set_conversation(uid, setting)
                await self.send(
                    uid,
                    view.prompt(
                        "Напоминания по умолчанию",
                        Text("Введите минуты через запятую: ", Code("15, 5, 1"), "."),
                        Text(Code("0"), " — в момент начала; ", Code("нет"), " — отключить."),
                        "",
                        Italic("Применится к будущим уведомлениям с общими настройками."),
                    )
                    if setting == "reminders"
                    else view.prompt(
                        "Утренняя сводка", Text("Введите время, например ", Code("09:00"), ".")
                    ),
                )
            return
        await self.ready(uid)
        if action == "t":
            await self.task_callback(uid, values, query.message)
        elif action == "a":
            period = await self.period(uid, values[1])
            await self.show_agenda(
                uid, period.days, int(values[2]), start_day=period.start, message=query.message
            )
        elif action == "l":
            await self.show_events(uid, int(values[1]))
        elif action == "p" and values[1] == "retry":
            state = await self.service.conversation(uid)
            if not state or state["mode"] != "parse":
                raise UserError("Нет сообщения для повторного разбора. Напишите событие.")
            if state["payload"].get("awaiting_answer"):
                raise UserError("Ответьте на последний вопрос бота или нажмите /cancel.")
            await self.run_parse(uid, state["payload"])
        elif action == "d":
            command, draft_id, version = values[1], values[2], int(values[3])
            row = await self.service.draft(uid, draft_id)
            origin, page = None, 0
            if len(values) != 4:
                if len(values) != 6:
                    raise UserError("Некорректная кнопка дела. Откройте /week.")
                origin, page = values[4], int(values[5])
                if not 0 <= page <= 10000:
                    raise UserError("Некорректная кнопка дела. Откройте /week.")
                origin = period_key(await self.period(uid, origin))
            if command not in {"save", "cancel", "edit"} or version < 1:
                raise UserError("Некорректная кнопка. Откройте /today или /week.")
            if row["kind"] == "task":
                target_id = row["result_id"] if row["status"] == "saved" else row["task_id"]
                if target_id:
                    target = await self.service.task(uid, target_id, include_deleted=True)
                    if not target["active"]:
                        await self.task_source(uid, origin, page, target_id, query.message)
                        return
                    if row["status"] == "saved" or target["version"] != row["task_version"]:
                        await self.show_task(
                            uid, target_id, message=query.message, origin=origin, page=page
                        )
                        return
                if row["version"] != version:
                    await self.show_draft(uid, row, origin=origin, page=page, message=query.message)
                    return
            elif row["version"] != version:
                raise UserError("Карточка устарела. Используйте её последнюю версию.")
            if command == "save":
                if row["status"] == "saved":
                    await self.send(
                        uid,
                        view.status(
                            "Эта карточка уже сохранена. Записи доступны в /today и /week."
                        ),
                    )
                    return
                event_id = await self.service.save_draft(uid, draft_id, version)
                if row["kind"] == "task":
                    await self.show_task(
                        uid, event_id, saved=True, message=query.message, origin=origin, page=page
                    )
                    return
                if origin is None:
                    await self.send(uid, view.status("Сохранено."))
                event = await self.service.event(uid, event_id)
                await self.show_event(
                    uid,
                    event_id,
                    event["version"],
                    row["original_day"],
                    origin=origin,
                    page=page,
                    message=query.message if origin is not None else None,
                )
            elif command == "cancel":
                if row["status"] == "saved":
                    raise UserError("Запись уже сохранена. Для удаления откройте /today или /week.")
                await self.service.cancel_draft(uid, draft_id, version)
                if row["kind"] == "task":
                    await self.present(uid, view.status("Черновик закрыт."), None, query.message)
                else:
                    await self.send(uid, view.status("Черновик закрыт."))
            elif command == "edit":
                if (
                    row["status"] != "pending"
                    or row["expires_at"] <= self.service.clock().timestamp()
                ):
                    raise UserError("Черновик закрыт или истёк. Добавьте событие заново.")
                await self.service.set_conversation(
                    uid,
                    "edit",
                    {
                        "base": {
                            "kind": row["kind"],
                            **read_spec(row["spec"], row["kind"]).to_dict(),
                        },
                        "reference": row["anchor_at"],
                        "draft_id": draft_id,
                        "draft_version": version,
                        "origin": origin,
                        "page": page,
                        "event_id": row["event_id"],
                        "event_version": row["event_version"],
                        "task_id": row["task_id"],
                        "task_version": row["task_version"],
                        "original_day": row["original_day"],
                    },
                )
                await self.send(
                    uid,
                    view.prompt(
                        "Что исправить?",
                        Text(
                            "Например: ",
                            Code("на завтра" if row["kind"] == "task" else "в 16:30"),
                            " или ",
                            Code(
                                "название: Купить продукты"
                                if row["kind"] == "task"
                                else "напомни за 30 и 5 минут"
                            ),
                            ".",
                        ),
                        "",
                        Italic("/cancel — выйти."),
                    ),
                )
        elif action in {"e", "o", "x", "r", "c", "yes"}:
            if len(values) not in {3, 4, 6}:
                raise UserError("Некорректная кнопка. Откройте /week.")
            event_id, version = values[1], int(values[2])
            day = (
                date.fromisoformat(values[3]).isoformat()
                if len(values) > 3 and values[3] != "all"
                else None
            )
            origin, page = None, 0
            if len(values) == 6:
                origin = period_key(await self.period(uid, values[4]))
                page = int(values[5])
                if not 0 <= page <= 10000:
                    raise UserError("Некорректная страница.")
            context = f":{origin}:{page}" if origin is not None else ""
            message = query.message if origin is not None else None
            if action in {"e", "o"}:
                await self.show_event(
                    uid, event_id, version, day, origin=origin, page=page, message=message
                )
                return
            event = await self.service.event(uid, event_id, version)
            spec = EventSpec.from_json(event["spec"])
            if day:
                occurrence = await self.service.occurrence(uid, event_id, day, version)
                spec = EventSpec.from_json(occurrence["spec"])
            target = day.replace("-", "") if day else "all"
            if action == "c":
                scope = (
                    "только эту встречу"
                    if day
                    else "все будущие встречи серии"
                    if spec.repeat != "once" or event["has_exceptions"]
                    else "это событие"
                )
                await self.present(
                    uid,
                    Text("⚠️ Удалить ", scope, "?\n\n", Bold(spec.title)),
                    keyboard(
                        [("Да, удалить", f"yes:{event_id}:{version}:{target}{context}")],
                        [
                            (
                                "Вернуться",
                                f"o:{event_id}:{version}:{target}{context}",
                            )
                        ],
                    ),
                    message,
                )
            elif action == "yes":
                await self.service.cancel_event(uid, event_id, version, day)
                if origin is not None:
                    period = await self.period(uid, origin)
                    await self.show_agenda(
                        uid, period.days, page, start_day=period.start, message=message
                    )
                    return
                await self.send(
                    uid,
                    view.status(
                        "Встреча удалена." if day else "Событие и его будущие напоминания удалены."
                    ),
                )
            else:
                await self.service.set_conversation(
                    uid,
                    "event_reminders" if action == "r" else "edit",
                    {
                        "base": spec.to_dict(),
                        "event_id": event_id,
                        "event_version": version,
                        "original_day": day,
                        "reference": self.service.clock().timestamp(),
                        "origin": origin,
                        "page": page,
                    },
                )
                await self.send(
                    uid,
                    view.prompt(
                        "Напоминания для встречи" if day else "Напоминания для события",
                        Text("Введите минуты через запятую: ", Code("15, 5, 1"), "."),
                        Text(Code("0"), " — в момент начала; ", Code("нет"), " — отключить;"),
                        Text(Code("общие"), " — использовать ваши настройки."),
                    )
                    if action == "r"
                    else view.prompt(
                        "Что изменить?",
                        "Напишите новое название, дату, время или повтор.",
                        "После разбора я покажу карточку для подтверждения.",
                        "",
                        Italic("/cancel — выйти."),
                    ),
                )
        else:
            raise UserError("Кнопка устарела. Откройте /help.")

    async def message(self, message: Message):
        uid = message.from_user.id
        if not message.text:
            await self.send(
                uid,
                view.status(
                    "Пока я понимаю только текстовые сообщения. Примеры: /help.", error=True
                ),
            )
            return
        text = message.text.strip()
        menus = {
            "Сегодня": "today",
            "7 дней": "week",
            "События": "events",
            "Дела": "tasks",
            "Настройки": "settings",
        }
        if text in menus:
            await self.service.set_conversation(uid, None)
            if text in {"Сегодня", "7 дней"}:
                await self.show_agenda(uid, 1 if text == "Сегодня" else 7)
            elif text == "События":
                await self.show_events(uid)
            elif text == "Дела":
                await self.send(
                    uid, view.status("Расписание доступно в «Сегодня» и «7 дней»."), MENU
                )
                await self.show_agenda(uid, 7)
            else:
                await self.show_settings(uid)
            return
        if text.startswith("/"):
            await self.send(
                uid, view.status("Неизвестная команда. Список команд: /help.", error=True)
            )
            return
        state = await self.service.conversation(uid)
        if state and state["mode"] == "timezone":
            aliases = {"москва": "Europe/Moscow", "ереван": "Asia/Yerevan", "utc": "UTC"}
            await self.service.preferences(uid, timezone=aliases.get(text.lower(), text))
            await self.service.set_conversation(uid, None)
            await self.show_settings(uid)
            return
        await self.ready(uid)
        if state and state["mode"] in {"reminders", "summary_time"}:
            if state["mode"] == "reminders":
                await self.service.preferences(uid, reminders=parse_reminders(text))
            else:
                await self.service.preferences(uid, summary_time=parse_time(text).strftime("%H:%M"))
            await self.service.set_conversation(uid, None)
            await self.show_settings(uid)
            return
        key = f"message:{uid}:{message.message_id}"
        existing = await self.service.request_drafts(uid, key)
        if existing is not None:
            await self.show_results(uid, existing)
            return
        if state and state["mode"] == "event_reminders":
            payload = state["payload"]
            spec = replace(
                EventSpec.from_dict(payload["base"]),
                reminder_minutes=parse_reminders(text, inherit=True),
            )
            rows = await self.service.create_drafts(
                uid,
                key,
                [spec],
                datetime.fromtimestamp(payload["reference"], UTC),
                event_id=payload["event_id"],
                event_version=payload["event_version"],
                original_day=payload["original_day"],
            )
            await self.service.set_conversation(uid, None)
            await self.show_draft(
                uid, rows[0], origin=payload.get("origin"), page=payload.get("page", 0)
            )
            return
        if len(text) > 4000:
            raise UserError("Сообщение слишком длинное. Разделите его на части до 4000 символов.")
        payload = dict(state["payload"]) if state and state["mode"] in {"edit", "parse"} else {}
        payload.setdefault("reference", message.date.timestamp())
        payload.setdefault("messages", [])
        if (
            len(payload["messages"]) >= 12
            or sum(len(m["content"]) for m in payload["messages"]) + len(text) > 12000
        ):
            raise UserError(
                "Слишком много уточнений. Нажмите /cancel и напишите полное событие заново."
            )
        payload["messages"].append({"role": "user", "content": text})
        payload["key"] = key
        payload["awaiting_answer"] = False
        await self.service.set_conversation(uid, "parse", payload)
        await self.run_parse(uid, payload)

    async def run_parse(self, uid, payload):
        user = await self.ready(uid)
        existing = await self.service.request_drafts(uid, payload["key"])
        if existing is not None:
            await self.show_results(
                uid, existing, origin=payload.get("origin"), page=payload.get("page", 0)
            )
            return
        now = self.service.clock()
        recent = [t for t in self.parse_times.get(uid, []) if t > now - timedelta(minutes=1)]
        if len(recent) >= 10:
            await self.send(
                uid,
                view.status(
                    "Не более десяти разборов в минуту. Подождите немного и нажмите «Повторить».",
                    error=True,
                ),
                keyboard([("Повторить", "p:retry")]),
            )
            return
        self.parse_times[uid] = [*recent, now]
        reference = datetime.fromtimestamp(payload["reference"], UTC)
        await self.send(uid, Text("✏️ ", Italic("Разбираю сообщение…")))
        try:
            result = await self.parser.parse(
                payload["messages"], reference, user["timezone"], payload.get("base")
            )
        except UserError as exc:
            await self.send(
                uid, view.status(str(exc), error=True), keyboard([("Повторить", "p:retry")])
            )
            return
        if result.question:
            await self.ask_clarification(uid, payload, result.question[:1000])
            return
        try:
            specs = [normalize(item, reference, user["timezone"]) for item in result.items]
            rows = await self.service.create_drafts(
                uid,
                payload["key"],
                specs,
                reference,
                event_id=payload.get("event_id"),
                event_version=payload.get("event_version"),
                task_id=payload.get("task_id"),
                task_version=payload.get("task_version"),
                original_day=payload.get("original_day"),
                draft_id=payload.get("draft_id"),
                draft_version=payload.get("draft_version"),
                auto_save=not (
                    payload.get("event_id") or payload.get("task_id") or payload.get("draft_id")
                ),
            )
        except UserError as exc:
            await self.ask_clarification(uid, payload, str(exc))
            return
        await self.show_results(
            uid, rows, origin=payload.get("origin"), page=payload.get("page", 0)
        )

    async def ask_clarification(self, uid, payload, question):
        payload["messages"].append({"role": "assistant", "content": question})
        payload["awaiting_answer"] = True
        await self.service.set_conversation(uid, "parse", payload)
        await self.send(
            uid, view.prompt("Уточните запись", question, "", Italic("/cancel — отменить ввод."))
        )
