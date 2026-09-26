"""Opt-in, paid live parser check using synthetic examples, without Telegram polling."""

import asyncio
import os
from datetime import UTC, date, datetime

from dotenv import load_dotenv

from .domain import EventSpec, TaskSpec, UserError
from .logging_config import configure_logging
from .parser import OpenAIParser, normalize

CASES = [
    (
        "каждый понедельник 14:00 Content Retro & Planning",
        "Content Retro & Planning",
        "weekly",
        (0,),
        "14:00",
    ),
    ("каждый вторник 14:00 Оля 1-1", "Оля 1-1", "weekly", (1,), "14:00"),
    ("каждую среду 15:00 Креаторская рассылки", "Креаторская рассылки", "weekly", (2,), "15:00"),
    ("завтра 16:00 Штурм // Ориентир 2027", "Штурм // Ориентир 2027", "once", (), "16:00"),
]


async def evaluate() -> None:
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise UserError("Для живой проверки нужен OPENAI_API_KEY. Проверка использует платный API.")
    parser = OpenAIParser(key, os.environ.get("OPENAI_MODEL", "gpt-5.4-mini-2026-03-17"))
    reference = datetime(2026, 9, 24, 8, tzinfo=UTC)
    try:
        for index, (text, title, repeat, weekdays, clock) in enumerate(CASES, 1):
            result = await parser.parse(
                [{"role": "user", "content": text}], reference, "Asia/Yerevan"
            )
            if result.question or len(result.items) != 1:
                raise UserError(f"Пример {index}: вместо события получено уточнение.")
            spec = normalize(result.items[0], reference, "Asia/Yerevan")
            if not isinstance(spec, EventSpec) or (
                spec.title,
                spec.repeat,
                spec.weekdays,
                spec.clock.strftime("%H:%M"),
            ) != (
                title,
                repeat,
                weekdays,
                clock,
            ):
                raise UserError(f"Пример {index}: разбор не совпал с ожидаемым результатом.")
            if repeat == "once" and spec.day != date(2026, 9, 25):
                raise UserError(f"Пример {index}: неправильно распознано «завтра».")
            print(f"Пример {index}: OK")
        for index, (text, title, day) in enumerate(
            (
                ("купить продукты", "купить продукты", date(2026, 9, 24)),
                ("завтра купить продукты", "купить продукты", date(2026, 9, 25)),
            ),
            len(CASES) + 1,
        ):
            result = await parser.parse(
                [{"role": "user", "content": text}], reference, "Asia/Yerevan"
            )
            if result.question or len(result.items) != 1:
                raise UserError(f"Пример {index}: вместо дела получено уточнение.")
            spec = normalize(result.items[0], reference, "Asia/Yerevan")
            if not isinstance(spec, TaskSpec) or spec.title.casefold() != title or spec.day != day:
                raise UserError(f"Пример {index}: дело или его дата не совпали с ожидаемыми.")
            print(f"Пример {index}: OK")
        mixed = await parser.parse(
            [{"role": "user", "content": "завтра купить хлеб\nзавтра 16:00 встреча"}],
            reference,
            "Asia/Yerevan",
        )
        if mixed.question or len(mixed.items) != 2:
            raise UserError("Смешанный пример: ожидались дело и событие.")
        specs = [normalize(item, reference, "Asia/Yerevan") for item in mixed.items]
        if not isinstance(specs[0], TaskSpec) or not isinstance(specs[1], EventSpec):
            raise UserError("Смешанный пример: неверные типы или порядок записей.")
        if (
            any(spec.day != date(2026, 9, 25) for spec in specs)
            or specs[1].clock.strftime("%H:%M") != "16:00"
        ):
            raise UserError("Смешанный пример: неверная дата или время.")
        print("Смешанный пример: OK")
        unsupported = await parser.parse(
            [{"role": "user", "content": "каждый день читать без времени"}],
            reference,
            "Asia/Yerevan",
        )
        if not unsupported.question or unsupported.items:
            raise UserError("Повторяющееся дело: ожидалось объяснение ограничения.")
        print("Неподдерживаемый повтор дела: OK")
    finally:
        await parser.close()


if __name__ == "__main__":
    configure_logging()
    try:
        asyncio.run(evaluate())
    except UserError as exc:
        print(str(exc))
        raise SystemExit(1) from None
