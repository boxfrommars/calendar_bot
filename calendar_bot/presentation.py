"""Telegram views built from literal text and entities, never parsed user markup."""

import json
import re
from collections.abc import Iterable, Sequence
from datetime import UTC, date, datetime, time

from aiogram.utils.formatting import Bold, Code, Italic, Text

from .domain import WEEKDAYS, EventSpec, local_instant, split_lines, zone

MONTHS = (
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
)
WEEKLY = (
    "понедельникам",
    "вторникам",
    "средам",
    "четвергам",
    "пятницам",
    "субботам",
    "воскресеньям",
)
ZONE_NAMES = {"Europe/Moscow": "Москва", "Asia/Yerevan": "Ереван", "UTC": "UTC"}
TIME_ROW = re.compile(r"^\d{2}:\d{2} — ")
SCHEDULE_PREFIX = "  По расписанию: "


def lines(items: Iterable[str | Text]) -> Text:
    nodes = []
    for item in items:
        if nodes:
            nodes.append("\n")
        nodes.append(item)
    return Text(*nodes)


def date_label(day: date, today: date) -> str:
    label = f"{WEEKDAYS[day.weekday()]}, {day.day} {MONTHS[day.month - 1]}"
    return label + (f" {day.year}" if day.year != today.year else "")


def utc_offset(timezone: str, instant: datetime) -> str:
    minutes = int(instant.astimezone(zone(timezone)).utcoffset().total_seconds() / 60)
    if not minutes:
        return "UTC"
    hours, remainder = divmod(abs(minutes), 60)
    return f"UTC{'+' if minutes > 0 else '-'}{hours}" + (f":{remainder:02}" if remainder else "")


def timezone_label(timezone: str, instant: datetime) -> str:
    name = ZONE_NAMES.get(timezone, timezone)
    offset = utc_offset(timezone, instant)
    return name if name == offset else f"{name} · {offset}"


def timezone_range(timezone: str, instants: Sequence[datetime]) -> str:
    """A week can straddle a DST change; use the offsets of its actual events."""
    offsets = list(dict.fromkeys(utc_offset(timezone, instant) for instant in instants))
    name = ZONE_NAMES.get(timezone, timezone)
    return name if offsets == [name] else f"{name} · {' → '.join(offsets)}"


def repeat_label(spec: EventSpec) -> str:
    if spec.repeat == "once":
        return ""
    if spec.repeat == "daily":
        return "Каждый день"
    if spec.weekdays == (0, 1, 2, 3, 4):
        return "По будням"
    if len(spec.weekdays) == 1:
        return "По " + WEEKLY[spec.weekdays[0]]
    return "Каждую неделю: " + ", ".join(WEEKDAYS[day] for day in spec.weekdays)


def minute_word(value: int, *, accusative=False) -> str:
    if 11 <= value % 100 <= 14:
        return "минут"
    if value % 10 == 1:
        return "минуту" if accusative else "минута"
    return "минуты" if value % 10 in {2, 3, 4} else "минут"


def reminder_label(values: Sequence[int]) -> str:
    before = [value for value in values if value > 0]
    parts = []
    if before:
        numbers = [str(value) for value in before]
        joined = ", ".join(numbers[:-1]) + " и " + numbers[-1] if len(numbers) > 1 else numbers[0]
        parts.append(f"За {joined} {minute_word(before[-1], accusative=True)}")
    if 0 in values:
        parts.append("в момент начала" if parts else "В момент начала")
    return "; ".join(parts) or "Выключены"


def reminder_source(spec: EventSpec, *, single=False) -> str:
    if spec.reminder_minutes is None:
        return "Напоминания из ваших настроек."
    return "Напоминания для этой встречи." if single else "Напоминания для этого события."


def status(message: str, *, error=False) -> Text:
    return Text("⚠️ " if error else "✅ ", message)


def prompt(heading: str, *body: str | Text) -> Text:
    return Text("✏️ ", Bold(heading), "\n\n", lines(body))


WELCOME = Text(
    "📅 ",
    Bold("Ваше расписание"),
    "\n\n",
    "Напомню о встречах и пришлю события на день.\n",
    "Новые события сохраняются автоматически — их можно изменить или удалить кнопками в карточке.\n\n",
    Italic("Для понимания новых записей и исправлений их текст передаётся OpenAI."),
)

HELP = lines(
    [
        Text("📅 ", Bold("Как добавить событие")),
        "",
        "Напишите название, дату и время:",
        Code("каждый понедельник 14:00 Content Retro & Planning"),
        Code("завтра 16:00 Штурм // Ориентир 2027"),
        "",
        "Можно добавить несколько событий одним сообщением. Они сохранятся автоматически, и я покажу карточки с датой и временем.",
        "Если данных недостаточно, сначала задам уточняющий вопрос.",
        "Поддерживаются разовые встречи, ежедневные повторы, будни и выбранные дни недели.",
        "",
        Bold("Напоминания"),
        "По умолчанию — за 15, 5 и 1 минуту. Для другого набора добавьте:",
        Code("напомни за 30 и 5 минут"),
        "",
        Bold("Расписание и настройки"),
        Text(Code("/today"), " — события сегодня"),
        Text(Code("/week"), " — ближайшие 7 дней"),
        Text(Code("/events"), " — события и серии"),
        Text(Code("/settings"), " — часовой пояс, напоминания и сводка"),
        Text(Code("/cancel"), " — выйти из текущего ввода"),
        Text(Code("/id"), " — ваш Telegram ID"),
        "",
        Bold("Изменить или удалить"),
        "Используйте кнопки в карточке. Изменения и удаление требуют подтверждения.",
        "Для одной встречи из серии откройте /week.",
        "",
        Italic(
            "У каждого события есть время начала. Задачи без времени и ежемесячные повторы пока не поддерживаются."
        ),
    ]
)


def event_card(
    heading: str,
    spec: EventSpec,
    instant: datetime | None,
    user_timezone: str,
    reminders: Sequence[int],
    now: datetime,
    *,
    draft=False,
    recurrence: EventSpec | None = None,
    single=False,
    saved=False,
) -> Text:
    # Drafts show the entered schedule time; saved cards show the user's local time.
    display_timezone = spec.timezone if draft else user_timezone
    today = now.astimezone(zone(display_timezone)).date()
    body = [
        Text("📝 " if draft else "✅ " if saved else "📅 ", Bold(heading)),
        "",
        Bold(spec.title),
    ]
    if instant is not None:
        local = instant.astimezone(zone(display_timezone))
        body.append(Text("📅 ", date_label(local.date(), today), " · ", Bold(f"{local:%H:%M}")))
    else:
        body.extend(
            [
                "Ближайших встреч в течение 30 дней нет.",
                Text("Время по расписанию: ", Bold(f"{spec.clock:%H:%M}")),
            ]
        )
    repeat = repeat_label(recurrence or spec)
    if repeat:
        body.append(Text("🔁 ", repeat))
    body.extend([Text("🔔 ", reminder_label(reminders)), ""])
    reference = instant or local_instant(spec.day, spec.clock, spec.timezone)
    if spec.timezone == user_timezone:
        body.append(Italic(timezone_label(display_timezone, reference)))
    else:
        # Explicit labels keep both the entered time and the converted time unambiguous.
        for label, tz in (("Расписание", spec.timezone), ("Ваше время", user_timezone)):
            local = reference.astimezone(zone(tz))
            moment = f"{date_label(local.date(), today)} · {local:%H:%M} · " if instant else ""
            body.append(Italic(f"{label}: {moment}{timezone_label(tz, reference)}"))
    note = reminder_source(spec, single=single)
    if draft:
        note += " Черновик действует 24 часа."
    body.append(Italic(note))
    return lines(body)


def schedule_note(spec: EventSpec, instant: datetime, user_timezone: str) -> str | None:
    if spec.timezone == user_timezone:
        return None
    local = instant.astimezone(zone(spec.timezone))
    moment = f"{local:%H:%M}"
    user_day = instant.astimezone(zone(user_timezone)).date()
    if local.date() != user_day:
        moment = f"{date_label(local.date(), user_day)} · {moment}"
    return f"{SCHEDULE_PREFIX}{moment} · {timezone_label(spec.timezone, instant)}"


def agenda(rows: list[dict], timezone: str, now: datetime, days: int, page: int) -> Text:
    today = now.astimezone(zone(timezone)).date()
    title = f"Сегодня · {date_label(today, today)}" if days == 1 else "Ближайшие 7 дней"
    body = [Text("📅 ", Bold(title))]
    previous_day = None
    instants = []
    for number, row in enumerate(rows, page * 8 + 1):
        spec = EventSpec.from_json(row["spec"])
        instant = datetime.fromtimestamp(row["start_at"], UTC)
        instants.append(instant)
        local = instant.astimezone(zone(timezone))
        if days > 1 and local.date() != previous_day:
            body.extend(["", Bold(date_label(local.date(), today))])
        body.extend(["", Text(f"{number}. ", Bold(f"{local:%H:%M}"), " — ", spec.title)])
        if instant <= now:
            body[-1] += Text(" · ", Italic("уже началось"))
        if note := schedule_note(spec, instant, timezone):
            body.append(Italic(note))
        previous_day = local.date()
    if not rows:
        body.extend(
            ["", "На сегодня событий нет." if days == 1 else "На ближайшие 7 дней событий нет."]
        )
    body.extend(["", Italic("Время: " + timezone_range(timezone, instants or [now]))])
    return lines(body)


def event_list(rows: list[dict], timezone: str, now: datetime, page: int) -> Text:
    body = [Text("📅 ", Bold("События и серии"))]
    today = now.astimezone(zone(timezone)).date()
    for number, row in enumerate(rows, page * 6 + 1):
        spec = EventSpec.from_json(row["spec"])
        instant = spec.first_after(now)
        repeat = repeat_label(spec) or date_label(spec.day, today)
        body.extend(
            [
                "",
                Text(f"{number}. ", Bold(spec.title)),
                Text(repeat, " · ", Bold(f"{spec.clock:%H:%M}")),
                Italic("Расписание: " + timezone_label(spec.timezone, instant)),
            ]
        )
        if timezone != spec.timezone:
            body.append(Italic("Ваш пояс: " + timezone_label(timezone, instant)))
    if not rows:
        body.extend(["", "Пока нет будущих событий. Напишите название, дату и время."])
    return lines(body)


def settings(user: dict, now: datetime) -> Text:
    timezone = timezone_label(user["timezone"], now) if user["timezone"] else "не выбран"
    summary = user["summary_time"] if user["summary_enabled"] else "Выключена"
    return lines(
        [
            Text("⚙️ ", Bold("Настройки")),
            "",
            Bold("Часовой пояс"),
            timezone,
            "",
            Bold("Напоминания по умолчанию"),
            reminder_label(json.loads(user["reminders"])),
            "",
            Bold("Утренняя сводка"),
            summary,
            Text("Пустые дни: ", "сообщать" if user["summary_empty"] else "пропускать"),
        ]
    )


def reminder_part(spec: EventSpec, start: datetime, now: datetime, timezone: str) -> str:
    seconds = (start - now).total_seconds()
    minutes = max(1, round(seconds / 60))
    label = (
        f"Через {minutes} {minute_word(minutes, accusative=True)}"
        if seconds > 0
        else "Начинается сейчас"
    )
    local = start.astimezone(zone(timezone))
    body = [f"⏰ {label}", "", spec.title, f"Начало в {local:%H:%M}"]
    if note := schedule_note(spec, start, timezone):
        body.append(note)
    body.extend(["", timezone_label(timezone, start)])
    return "\n".join(body)


def summary_parts(rows: list[dict], day: date, timezone: str, now: datetime) -> list[str]:
    today = now.astimezone(zone(timezone)).date()
    body = [f"☀️ События на {date_label(day, today)}", ""]
    instants = []
    for row in rows:
        spec = EventSpec.from_json(row["spec"])
        instant = datetime.fromtimestamp(row["start_at"], UTC)
        instants.append(instant)
        local = instant.astimezone(zone(timezone))
        body.append(f"{local:%H:%M} — {spec.title}")
        if note := schedule_note(spec, instant, timezone):
            body.append(note)
        body.append("")
    if not rows:
        body.extend(["На сегодня событий нет.", ""])
    reference = local_instant(day, time(), timezone)
    body.append("Время: " + timezone_range(timezone, instants or [reference]))
    return split_lines(body)


def notification_message(kind: str, part: str) -> Text:
    """Decorate known line positions/prefixes without changing persisted text.

    Summary parts from older releases have the same HH:MM prefix and ☀️ header.
    No text from titles is interpreted as HTML, Markdown, or formatting commands.
    """
    source = part.split("\n")
    body = []
    modern_reminder = kind == "reminder" and len(source) >= 6 and source[1] == ""
    for index, line in enumerate(source):
        value = line
        if kind == "summary":
            if index == 0 and line.startswith("☀️ События на "):
                value = Text("☀️ ", Bold(line[len("☀️ ") :]))
            elif TIME_ROW.match(line):
                value = Text(Bold(line[:5]), line[5:])
            elif line.startswith(("Время: ", SCHEDULE_PREFIX)):
                value = Italic(line)
        elif modern_reminder:
            if index == 2:
                value = Bold(line)
            elif index == 3 and line.startswith("Начало в "):
                value = Text("Начало в ", Bold(line[len("Начало в ") :]))
            elif index == len(source) - 1 or line.startswith(SCHEDULE_PREFIX):
                value = Italic(line)
        body.append(value)
    return lines(body)
