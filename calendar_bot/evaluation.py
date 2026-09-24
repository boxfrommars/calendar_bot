"""Opt-in, paid live parser check using synthetic examples, without Telegram polling."""

import asyncio
import os
from datetime import UTC, date, datetime

from dotenv import load_dotenv

from .domain import UserError
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
            if result.question or len(result.events) != 1:
                raise UserError(f"Пример {index}: вместо события получено уточнение.")
            spec = normalize(result.events[0], reference, "Asia/Yerevan")
            if (spec.title, spec.repeat, spec.weekdays, spec.clock.strftime("%H:%M")) != (
                title,
                repeat,
                weekdays,
                clock,
            ):
                raise UserError(f"Пример {index}: разбор не совпал с ожидаемым результатом.")
            if repeat == "once" and spec.day != date(2026, 9, 25):
                raise UserError(f"Пример {index}: неправильно распознано «завтра».")
            print(f"Пример {index}: OK")
    finally:
        await parser.close()


if __name__ == "__main__":
    configure_logging()
    try:
        asyncio.run(evaluate())
    except UserError as exc:
        print(str(exc))
        raise SystemExit(1) from None
