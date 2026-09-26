import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_REMINDERS = (15, 5, 1)
WEEKDAYS = ("пн", "вт", "ср", "чт", "пт", "сб", "вс")


class UserError(ValueError):
    """Safe, user-facing validation error. Never wrap provider error bodies."""


def utc_now() -> datetime:
    return datetime.now(UTC)


def zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError, ValueError, TypeError:
        raise UserError(
            "Неизвестный часовой пояс. Пример: Europe/Moscow или Asia/Yerevan."
        ) from None


def parse_time(value: str) -> time:
    try:
        if len(value) != 5 or value[2] != ":":
            raise ValueError
        hour, minute = map(int, value.split(":"))
        return time(hour, minute)
    except ValueError, TypeError:
        raise UserError("Укажите время в формате ЧЧ:ММ, например 09:00.") from None


def reminders(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    if any(type(v) is not int or not 0 <= v <= 10080 for v in values):
        raise UserError("Интервалы — целые минуты от 0 до 10080 (семи дней).")
    result = tuple(sorted(set(values), reverse=True))
    if len(result) > 5:
        raise UserError("Можно задать не более пяти интервалов.")
    return result


def parse_reminders(text: str, *, inherit: bool = False) -> tuple[int, ...] | None:
    if inherit and text.strip().lower() in {"общие", "по умолчанию", "default"}:
        return None
    if text.strip().lower() in {"нет", "выкл", "off", "-"}:
        return ()
    try:
        return reminders([int(v.strip()) for v in text.split(",")])
    except ValueError as exc:
        if isinstance(exc, UserError):
            raise
        raise UserError("Введите минуты через запятую: 15, 5, 1. Для отключения — «нет».") from None


def local_instant(day: date, clock: time, timezone: str) -> datetime:
    """Resolve folds once (first instant); move gaps forward by the DST jump."""
    naive = datetime.combine(day, clock)
    tz = zone(timezone)
    first = naive.replace(tzinfo=tz, fold=0)
    roundtrip = first.astimezone(UTC).astimezone(tz)
    if roundtrip.replace(tzinfo=None) != naive:
        # For an imaginary local time, fold=0 maps forward across the gap.
        return roundtrip.astimezone(UTC)
    return first.astimezone(UTC)


def local_day(instant: datetime, timezone: str) -> date:
    return instant.astimezone(zone(timezone)).date()


def day_window(day: date, days: int, timezone: str) -> tuple[datetime, datetime]:
    return (
        local_instant(day, time(), timezone),
        local_instant(shift_day(day, days), time(), timezone),
    )


def shift_day(day: date, days: int) -> date:
    try:
        return day + timedelta(days=days)
    except OverflowError:
        raise UserError("Дата за пределами поддерживаемого календаря.") from None


@dataclass(frozen=True)
class AgendaPeriod:
    start: date
    days: int = 7

    def __post_init__(self):
        if self.days not in {1, 7}:
            raise UserError("Некорректный период расписания.")
        if self.start < date(1, 1, 3) or shift_day(self.start, self.days) > date(9999, 12, 29):
            raise UserError("Дата за пределами поддерживаемого календаря.")

    @property
    def end(self) -> date:
        return shift_day(self.start, self.days)

    def contains(self, day: date) -> bool:
        return self.start <= day < self.end

    def shifted(self, direction: int) -> "AgendaPeriod":
        return AgendaPeriod(shift_day(self.start, self.days * direction), self.days)

    def window(self, timezone: str) -> tuple[datetime, datetime]:
        return day_window(self.start, self.days, timezone)


def validate_title(title: str) -> None:
    if not title.strip() or len(title) > 180 or "\n" in title or "\r" in title:
        raise UserError("Название должно занимать одну строку длиной от 1 до 180 символов.")


@dataclass(frozen=True)
class TaskSpec:
    title: str
    day: date

    def __post_init__(self) -> None:
        validate_title(self.title)

    def to_dict(self) -> dict:
        return {"title": self.title, "day": self.day.isoformat()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, value: dict) -> "TaskSpec":
        return cls(value["title"], date.fromisoformat(value["day"]))

    @classmethod
    def from_json(cls, value: str) -> "TaskSpec":
        return cls.from_dict(json.loads(value))


@dataclass(frozen=True)
class EventSpec:
    title: str
    day: date
    clock: time
    timezone: str
    repeat: Literal["once", "daily", "weekly"] = "once"
    weekdays: tuple[int, ...] = ()
    reminder_minutes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        validate_title(self.title)
        zone(self.timezone)
        if self.clock.tzinfo is not None or self.clock.second or self.clock.microsecond:
            raise UserError("Укажите местное время с точностью до минуты.")
        if self.repeat not in {"once", "daily", "weekly"}:
            raise UserError("Поддерживаются разовые, ежедневные и недельные события.")
        if self.repeat == "weekly" and not self.weekdays:
            raise UserError("Для недельного повтора нужны дни недели.")
        if any(type(d) is not int or not 0 <= d <= 6 for d in self.weekdays):
            raise UserError("Некорректный день недели.")
        if self.repeat != "weekly" and self.weekdays:
            raise UserError("Дни недели применяются только к недельным повторам.")
        if self.reminder_minutes is not None:
            object.__setattr__(self, "reminder_minutes", reminders(self.reminder_minutes))
        object.__setattr__(self, "weekdays", tuple(sorted(set(self.weekdays))))

    def to_dict(self) -> dict:
        result = asdict(self)
        result["day"] = self.day.isoformat()
        result["clock"] = self.clock.strftime("%H:%M")
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, value: dict) -> "EventSpec":
        return cls(
            title=value["title"],
            day=date.fromisoformat(value["day"]),
            clock=parse_time(value["clock"]),
            timezone=value["timezone"],
            repeat=value["repeat"],
            weekdays=tuple(value["weekdays"]),
            reminder_minutes=(
                None if value["reminder_minutes"] is None else tuple(value["reminder_minutes"])
            ),
        )

    @classmethod
    def from_json(cls, value: str) -> "EventSpec":
        return cls.from_dict(json.loads(value))

    def occurs_on(self, day: date) -> bool:
        return day >= self.day and (
            (self.repeat == "once" and day == self.day)
            or self.repeat == "daily"
            or (self.repeat == "weekly" and day.weekday() in self.weekdays)
        )

    def first_after(self, instant: datetime) -> datetime:
        if self.repeat == "once":
            return local_instant(self.day, self.clock, self.timezone)
        day = max(self.day, instant.astimezone(zone(self.timezone)).date())
        for step in range(8):
            candidate = day + timedelta(days=step)
            value = local_instant(candidate, self.clock, self.timezone)
            if self.occurs_on(candidate) and value > instant:
                return value
        raise UserError("Не удалось определить следующее повторение.")

    def as_single(self, day: date) -> "EventSpec":
        return replace(self, day=day, repeat="once", weekdays=())


def read_spec(value: str, kind: str = "event") -> EventSpec | TaskSpec:
    if kind == "task":
        return TaskSpec.from_json(value)
    if kind == "event":
        return EventSpec.from_json(value)
    raise UserError("Неизвестный тип записи.")


@dataclass(frozen=True)
class Occurrence:
    event_id: str
    original_day: date
    start: datetime
    spec: EventSpec

    @property
    def key(self) -> str:
        return f"{self.event_id}:{self.original_day.isoformat()}"


def resolve_occurrence(
    event_id: str, spec: EventSpec, exceptions: dict[date, EventSpec | None], original_day: date
) -> Occurrence | None:
    if original_day in exceptions:
        selected = exceptions[original_day]
    else:
        selected = spec.as_single(original_day) if spec.occurs_on(original_day) else None
    if selected is None:
        return None
    return Occurrence(
        event_id,
        original_day,
        local_instant(selected.day, selected.clock, selected.timezone),
        selected,
    )


def expand(
    event_id: str,
    spec: EventSpec,
    exceptions: dict[date, EventSpec | None],
    start: datetime,
    end: datetime,
) -> list[Occurrence]:
    """Exceptions keep their original identity, even after changing series weekdays."""
    tz = zone(spec.timezone)
    day = max(spec.day, start.astimezone(tz).date() - timedelta(days=1))
    last = end.astimezone(tz).date() + timedelta(days=1)
    result = []
    while day <= last:
        if spec.occurs_on(day) and day not in exceptions:
            instant = local_instant(day, spec.clock, spec.timezone)
            if start <= instant < end:
                result.append(Occurrence(event_id, day, instant, spec.as_single(day)))
        day += timedelta(days=1)
    for original_day, override in exceptions.items():
        if override is not None:
            instant = local_instant(override.day, override.clock, override.timezone)
            if start <= instant < end:
                result.append(Occurrence(event_id, original_day, instant, override))
    return sorted(result, key=lambda row: (row.start, row.key))


def display_time(instant: datetime, timezone: str) -> str:
    local = instant.astimezone(zone(timezone))
    return f"{local:%d.%m.%Y %H:%M} ({timezone}, UTC{local:%z})"


def repeat_text(spec: EventSpec) -> str:
    if spec.repeat == "once":
        return "один раз"
    if spec.repeat == "daily":
        return "каждый день"
    return "каждую неделю: " + ", ".join(WEEKDAYS[d] for d in spec.weekdays)


def reminder_text(values: tuple[int, ...] | list[int]) -> str:
    return ", ".join("в начале" if v == 0 else f"за {v} мин" for v in values) or "выключены"


def split_lines(lines: list[str], limit: int = 3500) -> list[str]:
    parts, current = [], ""
    for line in lines:
        if (
            len(current.encode("utf-16-le")) + len(line.encode("utf-16-le"))
        ) // 2 + 1 > limit and current:
            parts.append(current)
            current = ""
        current += ("\n" if current else "") + line
    if current:
        parts.append(current)
    return parts
