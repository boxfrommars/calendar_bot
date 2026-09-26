"""Opt-in, paid live parser check using synthetic examples, without Telegram polling."""

import argparse
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


async def evaluate_web_search(parser: OpenAIParser) -> None:
    # Fixed historical fixture: the EURO 2024 final kicked off at 21:00 in Berlin.
    # Only parser output is inspected; no calendar or Telegram state is created.
    reference = datetime(2024, 7, 14, 8, tzinfo=UTC)
    request = "Напомни о сегодняшнем финале мужского футбольного Евро-2024 Испания — Англия"
    for index, messages in enumerate(
        (
            [{"role": "user", "content": request}],
            [
                {"role": "user", "content": request},
                {"role": "assistant", "content": "Во сколько начинается матч?"},
                {"role": "user", "content": "А ты не можешь посмотреть?"},
            ],
        ),
        1,
    ):
        result = await parser.parse(messages, reference, "Asia/Yerevan")
        if result.question or len(result.items) != 1:
            raise UserError(f"Поиск {index}: ожидалось однозначное событие с источником.")
        spec = normalize(result.items[0], reference, "Asia/Yerevan")
        if (
            not isinstance(spec, EventSpec)
            or spec.source is None
            or spec.first_after(reference) != datetime(2024, 7, 14, 19, tzinfo=UTC)
        ):
            raise UserError(f"Поиск {index}: неверное начало финала или отсутствует источник.")
        print(f"Поиск {index}: OK")
    # At 21:47 in Yerevan, 19:45 BST is still future (22:45 in Yerevan).
    # Includes the follow-up from the reported premature 'already passed' refusal.
    timezone_reference = datetime(2026, 9, 26, 17, 47, tzinfo=UTC)
    timezone_request = "Напомни о сегодняшнем матче мужской футбольной Лиги наций Англия — Испания"
    for index, messages in enumerate(
        (
            [{"role": "user", "content": timezone_request}],
            [
                {"role": "user", "content": timezone_request},
                {
                    "role": "assistant",
                    "content": "Это время уже прошло. Укажите будущую дату и время.",
                },
                {"role": "user", "content": "А во сколько было это время?"},
            ],
        ),
        1,
    ):
        result = await parser.parse(messages, timezone_reference, "Asia/Yerevan")
        if result.question or len(result.items) != 1:
            raise UserError(f"Часовой пояс {index}: ожидалось событие, а не пояснение в question.")
        spec = normalize(result.items[0], timezone_reference, "Asia/Yerevan")
        if (
            not isinstance(spec, EventSpec)
            or spec.source is None
            or spec.first_after(timezone_reference) != datetime(2026, 9, 26, 18, 45, tzinfo=UTC)
        ):
            raise UserError(f"Часовой пояс {index}: неверный перевод BST в UTC или нет источника.")
        print(f"Часовой пояс {index}: OK")
    for index, text in enumerate(
        (
            "Напомни о завтрашнем концерте, город и исполнитель пока неизвестны",
            "Напомни о сегодняшнем матче вымышленных команд Кварц-92817 и Базальт-58329",
        ),
        1,
    ):
        result = await parser.parse([{"role": "user", "content": text}], reference, "Asia/Yerevan")
        if result.items or not result.question:
            raise UserError(f"Неоднозначный поиск {index}: ожидалось уточнение без записей.")
        print(f"Неоднозначный поиск {index}: OK")


async def evaluate(*, web_search=False) -> None:
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
        if web_search:
            await evaluate_web_search(parser)
    finally:
        await parser.close()


if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument(
        "--web-search",
        action="store_true",
        help="Also run paid web-search checks for public events.",
    )
    options = arguments.parse_args()
    configure_logging()
    try:
        asyncio.run(evaluate(web_search=options.web_search))
    except UserError as exc:
        print(str(exc))
        raise SystemExit(1) from None
