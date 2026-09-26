"""Opt-in, paid live parser check using synthetic examples, without Telegram polling."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import AsyncOpenAI

from .config import DEFAULT_OPENAI_MODEL
from .domain import EventSpec, TaskSpec, UserError
from .logging_config import configure_logging
from .model_trace import ModelTrace
from .parser import (
    COMMON_INSTRUCTIONS,
    CREATE_INSTRUCTIONS,
    EDIT_INSTRUCTIONS,
    OpenAIParser,
    ParseResult,
    normalize,
)


class EvaluationClient:
    """Measure API attempts without retaining prompts, answers, headers or error bodies."""

    def __init__(self, client, *, clock=time.perf_counter):
        self.client = client
        self.clock = clock
        self.responses = SimpleNamespace(parse=self.parse)
        self.attempts = []
        self.case_context = {}

    async def parse(self, **kwargs):
        started = self.clock()
        entry = {
            "mode": "create" if kwargs.get("tools") else "edit",
            "requested_model": kwargs["model"],
            "status": "error",
            **self.case_context,
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


REFERENCE = datetime(2026, 9, 24, 8, tzinfo=UTC)
TIMEZONE = "Asia/Yerevan"


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    messages: list[dict]
    reference: datetime = REFERENCE
    base: dict | None = None
    expected: tuple[dict, ...] = ()
    start_utc: datetime | None = None
    web_search: bool = False


class CheckFailed(Exception):
    """Only fixed check codes; never model output or exception bodies."""


def check_result(case: EvaluationCase, result: ParseResult) -> None:
    expected_count = len(case.expected) if case.start_utc is None else 1
    if not expected_count:
        if result.items or not result.question or not result.question.strip():
            raise CheckFailed("expected_clarification")
        return
    if result.question or len(result.items) != expected_count:
        raise CheckFailed("expected_items")
    if result.question_sources:
        raise CheckFailed("unexpected_question_sources")
    try:
        specs = [normalize(item, case.reference, TIMEZONE) for item in result.items]
    except UserError:
        raise CheckFailed("invalid_item") from None
    if case.start_utc is not None:
        spec = specs[0]
        if not isinstance(spec, EventSpec) or spec.source is None:
            raise CheckFailed("expected_web_event")
        if spec.first_after(case.reference) != case.start_utc:
            raise CheckFailed("wrong_start_utc")
        return
    for spec, expected in zip(specs, case.expected, strict=True):
        fields = spec.to_dict() | {"kind": "task" if isinstance(spec, TaskSpec) else "event"}
        for key, value in expected.items():
            actual = fields.get(key)
            if key == "title" and isinstance(spec, TaskSpec) and case.base is None:
                actual, value = actual.casefold(), value.casefold()
            if actual != value:
                raise CheckFailed(f"field_mismatch:{key}")


def build_cases() -> list[EvaluationCase]:
    cases = []

    def add(id, text, **kwargs):
        messages = [{"role": "user", "content": text}] if isinstance(text, str) else text
        cases.append(EvaluationCase(id, messages, **kwargs))

    for id, text, title, repeat, weekdays, clock in (
        (
            "weekly_monday",
            "каждый понедельник 14:00 Content Retro & Planning",
            "Content Retro & Planning",
            "weekly",
            (0,),
            "14:00",
        ),
        ("weekly_tuesday", "каждый вторник 14:00 Оля 1-1", "Оля 1-1", "weekly", (1,), "14:00"),
        (
            "weekly_wednesday",
            "каждую среду 15:00 Креаторская рассылки",
            "Креаторская рассылки",
            "weekly",
            (2,),
            "15:00",
        ),
        (
            "title_with_year",
            "завтра 16:00 Штурм // Ориентир 2027",
            "Штурм // Ориентир 2027",
            "once",
            (),
            "16:00",
        ),
        ("daily", "каждый день 14:00 Зарядка", "Зарядка", "daily", (), "14:00"),
        (
            "weekly_two_days",
            "каждый понедельник и четверг 14:00 Планирование",
            "Планирование",
            "weekly",
            (0, 3),
            "14:00",
        ),
    ):
        expected = {
            "kind": "event",
            "title": title,
            "repeat": repeat,
            "weekdays": weekdays,
            "clock": clock,
        }
        if repeat == "once":
            expected["day"] = "2026-09-25"
        add(id, text, expected=(expected,))
    for id, text, day in (
        ("task_today", "купить продукты", "2026-09-24"),
        ("task_tomorrow", "завтра купить продукты", "2026-09-25"),
    ):
        add(id, text, expected=({"kind": "task", "title": "купить продукты", "day": day},))
    add(
        "mixed_list",
        "завтра купить хлеб\nзавтра 16:00 встреча",
        expected=(
            {"kind": "task", "day": "2026-09-25"},
            {"kind": "event", "day": "2026-09-25", "clock": "16:00"},
        ),
    )
    add("recurring_task", "каждый день читать без времени")
    titles = [f"купить товар {index}" for index in range(1, 12)]
    lines = [f"завтра {title}" for title in titles]
    add(
        "ten_items",
        "\n".join(lines[:10]),
        expected=tuple(
            {"kind": "task", "title": title, "day": "2026-09-25"} for title in titles[:10]
        ),
    )
    for id, text in (
        ("eleven_items", "\n".join(lines)),
        ("every_two_days", "каждые два дня в 14:00 Зарядка"),
        ("every_two_weeks", "каждые две недели по понедельникам в 14:00 Планирование"),
        ("series_end_date", "каждый понедельник в 14:00 Планирование до 31 октября 2026 года"),
        ("series_count", "каждый понедельник в 14:00 Планирование, всего пять встреч"),
        (
            "mixed_unsupported",
            "завтра купить хлеб\nкаждые два дня в 14:00 Зарядка\nзавтра 16:00 Встреча",
        ),
    ):
        add(id, text)
    event = {
        "title": "Планирование & Итоги",
        "day": "2026-09-28",
        "clock": "14:00",
        "timezone": TIMEZONE,
        "repeat": "weekly",
        "weekdays": [0, 3],
        "reminder_minutes": [30, 5],
    }
    task = {"kind": "task", "title": "Купить хлеб", "day": "2026-09-25"}
    for id, base, text, changes in (
        (
            "edit_title",
            event,
            "переименуй в Планирование // Команда",
            {"title": "Планирование // Команда"},
        ),
        ("edit_time", event, "теперь в 16:00", {"clock": "16:00"}),
        ("edit_reminders", event, "отключи напоминания", {"reminder_minutes": []}),
        ("edit_task_date", task, "перенеси на послезавтра", {"day": "2026-09-26"}),
    ):
        kind = base.get("kind", "event")
        spec = (TaskSpec if kind == "task" else EventSpec).from_dict(base | changes)
        expected = spec.to_dict() | {"kind": kind}
        if kind == "event":
            expected["source"] = None
        add(id, text, base=base, expected=(expected,))
    add("edit_type_rejected", "преврати дело во встречу завтра в 15:00", base=task)
    add("edit_add_rejected", "сохрани текущую серию и добавь ещё завтра в 18:00 Обед", base=event)
    # Historical fixtures: sources and their timezone conventions are linked in docs/development.md.
    for prefix, reference, request, answer, followup, start_utc in (
        (
            "web_euro",
            datetime(2024, 7, 14, 8, tzinfo=UTC),
            "Напомни о сегодняшнем финале мужского футбольного Евро-2024 Испания — Англия",
            "Во сколько начинается матч?",
            "А ты не можешь посмотреть?",
            datetime(2024, 7, 14, 19, tzinfo=UTC),
        ),
        (
            "web_bst",
            datetime(2026, 9, 26, 17, 47, tzinfo=UTC),
            "Напомни о сегодняшнем матче мужской футбольной Лиги наций Англия — Испания",
            "Это время уже прошло. Укажите будущую дату и время.",
            "А во сколько было это время?",
            datetime(2026, 9, 26, 18, 45, tzinfo=UTC),
        ),
    ):
        messages = [{"role": "user", "content": request}]
        add(prefix, messages, reference=reference, start_utc=start_utc, web_search=True)
        add(
            prefix + "_followup",
            [
                *messages,
                {"role": "assistant", "content": answer},
                {"role": "user", "content": followup},
            ],
            reference=reference,
            start_utc=start_utc,
            web_search=True,
        )
    for id, text in (
        ("web_ambiguous", "Напомни о завтрашнем концерте, город и исполнитель пока неизвестны"),
        (
            "web_fictional",
            "Напомни о сегодняшнем матче вымышленных команд Кварц-92817 и Базальт-58329",
        ),
    ):
        add(id, text, reference=datetime(2024, 7, 14, 8, tzinfo=UTC), web_search=True)
    return cases


def select_cases(*, web_search=False, case_ids=()) -> list[EvaluationCase]:
    cases = build_cases()
    selected = set(case_ids)
    if selected - {case.id for case in cases}:
        raise UserError("Неизвестный ID сценария. Используйте --list-cases.")
    if not web_search and any(case.web_search and case.id in selected for case in cases):
        raise UserError("Для веб-сценария укажите --web-search.")
    return [
        case
        for case in cases
        if (web_search or not case.web_search) and (not selected or case.id in selected)
    ]


class EvaluationRun:
    def __init__(
        self, metrics: EvaluationClient, cases: list[EvaluationCase], *, repeat=1, fail_fast=False
    ):
        if repeat < 1 or not cases:
            raise UserError("Нужен непустой набор и положительное число повторов.")
        self.metrics = metrics
        self.cases = cases
        self.fail_fast = fail_fast
        self.results = [
            {"case_id": case.id, "repetition": repetition, "status": "not_run"}
            for repetition in range(1, repeat + 1)
            for case in cases
        ]

    async def run(self, parser: OpenAIParser) -> None:
        cases = {case.id: case for case in self.cases}
        for entry in self.results:
            case = cases[entry["case_id"]]
            self.metrics.case_context = {key: entry[key] for key in ("case_id", "repetition")}
            started = time.perf_counter()
            try:
                result = await parser.parse(case.messages, case.reference, TIMEZONE, case.base)
                entry.update(item_count=len(result.items), has_question=bool(result.question))
                check_result(case, result)
                entry["status"] = "passed"
            except CheckFailed as exc:
                entry.update(status="failed", check=str(exc))
            except asyncio.CancelledError:
                entry["status"] = "cancelled"
                raise
            except Exception as exc:
                entry.update(status="error", error_type=type(exc).__name__)
            finally:
                entry["seconds"] = round(time.perf_counter() - started, 4)
                self.metrics.case_context = {}
            detail = entry.get("check") or entry.get("error_type") or ""
            print(f"{case.id} [{entry['repetition']}]: {entry['status']} {detail}".rstrip())
            if self.fail_fast and entry["status"] != "passed":
                break

    def report(self) -> dict:
        outcomes = {
            status: sum(entry["status"] == status for entry in self.results)
            for status in ("passed", "failed", "error", "cancelled", "not_run")
        }
        report = self.metrics.report(completed=outcomes["passed"] == len(self.results))
        report.update(
            schema_version=2,
            finished=outcomes["not_run"] == 0 and outcomes["cancelled"] == 0,
            outcomes={"planned": len(self.results), **outcomes},
            cases=self.results,
            suite_sha256=hashlib.sha256(
                json.dumps(
                    [asdict(case) for case in self.cases], sort_keys=True, default=str
                ).encode()
            ).hexdigest(),
            prompt_sha256={
                mode: hashlib.sha256((COMMON_INSTRUCTIONS + instructions).encode()).hexdigest()
                for mode, instructions in (
                    ("create", CREATE_INSTRUCTIONS),
                    ("edit", EDIT_INSTRUCTIONS),
                )
            },
        )
        return report


async def evaluate(
    *, web_search=False, report_path: Path | None = None, case_ids=(), repeat=1, fail_fast=False
) -> dict:
    cases = select_cases(web_search=web_search, case_ids=case_ids)
    if repeat < 1:
        raise UserError("Число повторов должно быть положительным.")
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise UserError("Для живой проверки нужен OPENAI_API_KEY. Проверка использует платный API.")
    model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
    if not model:
        raise UserError("OPENAI_MODEL не может быть пустым.")
    metrics = EvaluationClient(AsyncOpenAI(api_key=key, timeout=30.0, max_retries=0))
    trace_dir = os.environ.get("OPENAI_TRACE_DIR", "").strip()
    parser = OpenAIParser(
        key, model, client=metrics, trace=ModelTrace(Path(trace_dir)) if trace_dir else None
    )
    run = EvaluationRun(metrics, cases, repeat=repeat, fail_fast=fail_fast)
    print(
        f"Модель: {model}; reasoning=low; max_output_tokens=3000; сценариев={len(cases)}; повторов={repeat}"
    )
    try:
        await run.run(parser)
    finally:
        try:
            await parser.close()
        finally:
            report = run.report()
            print("Результаты: " + json.dumps(report["outcomes"], ensure_ascii=False))
            print("Метрики: " + json.dumps(report["summary"], ensure_ascii=False))
            if report_path:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                report_path.write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
    return report


def main(argv=None) -> int:
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument(
        "--report", type=Path, help="Write technical JSON without prompts or answers."
    )
    arguments.add_argument(
        "--web-search", action="store_true", help="Include paid web-search scenarios."
    )
    arguments.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        default=[],
        help="Run only this case ID; may be repeated.",
    )
    arguments.add_argument(
        "--repeat", type=int, default=1, help="Run each selected case this many times."
    )
    arguments.add_argument(
        "--fail-fast", action="store_true", help="Stop after the first failed case."
    )
    arguments.add_argument(
        "--list-cases",
        action="store_true",
        help="List case IDs without loading credentials or calling the API.",
    )
    options = arguments.parse_args(argv)
    configure_logging()
    if options.list_cases:
        for case in build_cases():
            group = "web" if case.web_search else "edit" if case.base is not None else "create"
            print(f"{case.id}: {group}")
        return 0
    try:
        report = asyncio.run(
            evaluate(
                web_search=options.web_search,
                report_path=options.report,
                case_ids=options.case_ids,
                repeat=options.repeat,
                fail_fast=options.fail_fast,
            )
        )
    except UserError as exc:
        print(str(exc))
        return 1
    except Exception as exc:
        print(f"Evaluation error: {type(exc).__name__}")
        return 1
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
