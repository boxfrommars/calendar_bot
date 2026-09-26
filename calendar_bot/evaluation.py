"""Opt-in, paid live parser check using synthetic examples, without Telegram polling."""

import argparse
import asyncio
import json
import math
import os
import statistics
import time
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .config import DEFAULT_OPENAI_MODEL
from .domain import EventSpec, TaskSpec, UserError
from .logging_config import configure_logging
from .parser import OpenAIParser, normalize


class EvaluationClient:
    """Measure API attempts without retaining prompts, answers, headers or error bodies."""

    def __init__(self, client, *, clock=time.perf_counter):
        self.client = client
        self.clock = clock
        self.responses = SimpleNamespace(parse=self.parse)
        self.attempts = []

    async def parse(self, **kwargs):
        started = self.clock()
        entry = {
            "mode": "create" if kwargs.get("tools") else "edit",
            "requested_model": kwargs["model"],
            "status": "error",
        }
        try:
            response = await self.client.responses.parse(**kwargs)
            usage = getattr(response, "usage", None)
            entry.update(
                status=response.status,
                model=getattr(response, "model", None),
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
                cached_input_tokens=getattr(
                    getattr(usage, "input_tokens_details", None), "cached_tokens", None
                ),
                reasoning_tokens=getattr(
                    getattr(usage, "output_tokens_details", None), "reasoning_tokens", None
                ),
                web_search_calls=sum(
                    output.type == "web_search_call" for output in (response.output or [])
                ),
            )
            return response
        except (Exception, asyncio.CancelledError) as exc:
            entry["error_type"] = type(exc).__name__
            raise
        finally:
            entry["seconds"] = round(self.clock() - started, 4)
            self.attempts.append(entry)

    async def close(self):
        await self.client.close()

    def report(self, *, completed: bool) -> dict:
        def summary(attempts):
            seconds = sorted(entry["seconds"] for entry in attempts)
            totals = {}
            for field in (
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "reasoning_tokens",
                "web_search_calls",
            ):
                values = [entry[field] for entry in attempts if entry.get(field) is not None]
                totals[field] = sum(values) if values else None
            return {
                "attempts": len(attempts),
                "completed_responses": sum(entry["status"] == "completed" for entry in attempts),
                "responses_with_usage": sum(
                    entry.get("input_tokens") is not None for entry in attempts
                ),
                "seconds_total": round(sum(seconds), 4),
                "seconds_p50": statistics.median(seconds) if seconds else None,
                "seconds_p95": seconds[math.ceil(len(seconds) * 0.95) - 1] if seconds else None,
                **totals,
            }

        return {
            "completed": completed,
            "summary": summary(self.attempts),
            "by_mode": {
                mode: summary([entry for entry in self.attempts if entry["mode"] == mode])
                for mode in ("create", "edit")
            },
            "attempts": self.attempts,
        }


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
    ("каждый день 14:00 Зарядка", "Зарядка", "daily", (), "14:00"),
    (
        "каждый понедельник и четверг 14:00 Планирование",
        "Планирование",
        "weekly",
        (0, 3),
        "14:00",
    ),
]


async def evaluate_edits(parser: OpenAIParser, reference: datetime) -> None:
    event = {
        "title": "Планирование & Итоги",
        "day": "2026-09-28",
        "clock": "14:00",
        "timezone": "Asia/Yerevan",
        "repeat": "weekly",
        "weekdays": [0, 3],
        "reminder_minutes": [30, 5],
    }
    task = {"kind": "task", "title": "Купить хлеб", "day": "2026-09-25"}
    for name, base, text, changes in (
        (
            "Название события",
            event,
            "переименуй в Планирование // Команда",
            {"title": "Планирование // Команда"},
        ),
        ("Время серии", event, "теперь в 16:00", {"clock": "16:00"}),
        ("Напоминания серии", event, "отключи напоминания", {"reminder_minutes": []}),
        ("Дата дела", task, "перенеси на послезавтра", {"day": "2026-09-26"}),
    ):
        result = await parser.parse(
            [{"role": "user", "content": text}], reference, "Asia/Yerevan", base
        )
        if result.question or len(result.items) != 1:
            raise UserError(f"{name}: ожидалась одна изменённая запись.")
        spec = normalize(result.items[0], reference, "Asia/Yerevan")
        expected = {key: value for key, value in base.items() if key != "kind"} | changes
        expected_spec = (
            TaskSpec.from_dict(expected)
            if base.get("kind") == "task"
            else EventSpec.from_dict(expected)
        )
        if spec != expected_spec or result.question_sources:
            raise UserError(f"{name}: изменились другие поля или появились источники.")
        print(f"{name}: OK")
    for name, base, text in (
        ("Запрет смены типа дела", task, "преврати дело во встречу завтра в 15:00"),
        (
            "Запрет добавления при правке",
            event,
            "сохрани текущую серию и добавь ещё завтра в 18:00 Обед",
        ),
    ):
        result = await parser.parse(
            [{"role": "user", "content": text}], reference, "Asia/Yerevan", base
        )
        if result.items or not result.question:
            raise UserError(f"{name}: ожидалось уточнение без записей.")
        print(f"{name}: OK")


async def evaluate_constraints(parser: OpenAIParser, reference: datetime) -> None:
    titles = [f"купить товар {index}" for index in range(1, 12)]
    lines = [f"завтра {title}" for title in titles]
    result = await parser.parse(
        [{"role": "user", "content": "\n".join(lines[:10])}], reference, "Asia/Yerevan"
    )
    if result.question or len(result.items) != 10:
        raise UserError("Десять записей: ожидался полный список без уточнения.")
    specs = [normalize(item, reference, "Asia/Yerevan") for item in result.items]
    if any(
        not isinstance(spec, TaskSpec)
        or spec.title.casefold() != title
        or spec.day != date(2026, 9, 25)
        for spec, title in zip(specs, titles[:10], strict=True)
    ):
        raise UserError("Десять записей: неверные типы, названия, даты или порядок.")
    print("Десять записей: OK")
    for name, text in (
        ("Одиннадцать записей", "\n".join(lines)),
        ("Интервал в несколько дней", "каждые два дня в 14:00 Зарядка"),
        ("Интервал в несколько недель", "каждые две недели по понедельникам в 14:00 Планирование"),
        ("Дата окончания серии", "каждый понедельник в 14:00 Планирование до 31 октября 2026 года"),
        ("Число повторений", "каждый понедельник в 14:00 Планирование, всего пять встреч"),
        (
            "Ограничение в смешанном списке",
            "завтра купить хлеб\nкаждые два дня в 14:00 Зарядка\nзавтра 16:00 Встреча",
        ),
    ):
        result = await parser.parse([{"role": "user", "content": text}], reference, "Asia/Yerevan")
        if result.items or not result.question:
            raise UserError(f"{name}: ожидалось уточнение без частичного списка.")
        print(f"{name}: OK")


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


async def evaluate(*, web_search=False, report_path: Path | None = None) -> None:
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise UserError("Для живой проверки нужен OPENAI_API_KEY. Проверка использует платный API.")
    model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
    if not model:
        raise UserError("OPENAI_MODEL не может быть пустым.")
    metrics = EvaluationClient(AsyncOpenAI(api_key=key, timeout=30.0, max_retries=0))
    parser = OpenAIParser(key, model, client=metrics)
    print(f"Модель: {model}; reasoning=low; max_output_tokens=3000")
    reference = datetime(2026, 9, 24, 8, tzinfo=UTC)
    completed = False
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
        await evaluate_constraints(parser, reference)
        await evaluate_edits(parser, reference)
        if web_search:
            await evaluate_web_search(parser)
        completed = True
    finally:
        await parser.close()
        report = metrics.report(completed=completed)
        print("Метрики: " + json.dumps(report["summary"], ensure_ascii=False))
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument(
        "--report", type=Path, help="Write technical metrics as JSON, without prompts or answers."
    )
    arguments.add_argument(
        "--web-search",
        action="store_true",
        help="Also run paid web-search checks for public events.",
    )
    options = arguments.parse_args()
    configure_logging()
    try:
        asyncio.run(evaluate(web_search=options.web_search, report_path=options.report))
    except UserError as exc:
        print(str(exc))
        raise SystemExit(1) from None
