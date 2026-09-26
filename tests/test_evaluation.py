import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from calendar_bot.domain import UserError
from calendar_bot.evaluation import (
    CheckFailed,
    EvaluationClient,
    EvaluationRun,
    build_cases,
    check_result,
    evaluate,
    main,
    select_cases,
)
from calendar_bot.parser import ParsedEvent, ParsedTask, ParseResult


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_metrics_include_failures_and_never_copy_private_content(self):
        response = SimpleNamespace(
            status="completed",
            model="gpt-6-luna",
            output=[SimpleNamespace(type="web_search_call", query="SECRET_QUERY")],
            output_parsed="SECRET_REPLY",
            headers="SECRET_HEADERS",
            usage=SimpleNamespace(
                input_tokens=120,
                output_tokens=30,
                input_tokens_details=SimpleNamespace(cached_tokens=100),
                output_tokens_details=SimpleNamespace(reasoning_tokens=10),
            ),
        )
        sdk = SimpleNamespace(
            responses=SimpleNamespace(
                parse=AsyncMock(side_effect=[response, RuntimeError("SECRET_ERROR")])
            ),
            close=AsyncMock(),
        )
        ticks = iter([0.0, 2.0, 3.0, 8.0])
        client = EvaluationClient(sdk, clock=lambda: next(ticks))
        self.assertIs(
            await client.responses.parse(model="gpt-6-luna", tools=[{}], input="SECRET_INPUT"),
            response,
        )
        with self.assertRaises(RuntimeError):
            await client.responses.parse(model="gpt-6-luna", input="SECRET_INPUT")
        report = client.report(completed=False)
        self.assertFalse(report["completed"])
        self.assertNotIn("SECRET", json.dumps(report))
        summary = report["summary"]
        self.assertEqual(summary["attempts"], 2)
        self.assertEqual(summary["completed_responses"], 1)
        self.assertEqual(summary["responses_with_usage"], 1)
        self.assertEqual(summary["seconds_p50"], 3.5)
        self.assertEqual(summary["seconds_p95"], 5)
        self.assertEqual(summary["input_tokens"], 120)
        self.assertEqual(summary["reasoning_tokens"], 10)
        self.assertEqual(summary["web_search_calls"], 1)
        self.assertIsNone(report["by_mode"]["edit"]["input_tokens"])
        await client.close()
        sdk.close.assert_awaited_once()

    async def test_cancelled_attempt_is_recorded_and_propagated(self):
        client = EvaluationClient(
            SimpleNamespace(
                responses=SimpleNamespace(parse=AsyncMock(side_effect=asyncio.CancelledError))
            ),
            clock=lambda: 0.0,
        )
        self.assertIsNone(client.report(completed=False)["summary"]["input_tokens"])
        with self.assertRaises(asyncio.CancelledError):
            await client.responses.parse(model="gpt-6-luna")
        self.assertEqual(client.attempts[0]["error_type"], "CancelledError")
        self.assertIsNone(client.report(completed=False)["summary"]["web_search_calls"])

    async def test_edit_evaluation_rejects_unrequested_changes(self):
        item = ParsedEvent(
            kind="event",
            title="Планирование // Команда",
            date="2026-09-28",
            time="16:00",
            timezone="Asia/Yerevan",
            repeat="weekly",
            weekdays=[0, 3],
            reminders=[30, 5],
            time_source="user",
        )
        parser = SimpleNamespace(
            parse=AsyncMock(return_value=ParseResult(items=[item], question=None))
        )
        case = select_cases(case_ids=["edit_title"])[0]
        with self.assertRaisesRegex(CheckFailed, "field_mismatch:clock"):
            check_result(case, await parser.parse())

    def test_case_selection_does_not_silently_skip_unknown_or_web_cases(self):
        self.assertEqual(len(build_cases()), 31)
        self.assertEqual(len({case.id for case in build_cases()}), 31)
        self.assertEqual(len(select_cases()), 23)
        self.assertEqual(len(select_cases(web_search=True)), 31)
        self.assertEqual(len(select_cases(case_ids=["edit_title", "edit_title"])), 1)
        for ids in (["unknown"], ["edit_title", "unknown"], ["web_euro"]):
            with self.assertRaises(UserError):
                select_cases(case_ids=ids)
        with patch("calendar_bot.evaluation.load_dotenv") as dotenv:
            with patch("calendar_bot.evaluation.AsyncOpenAI") as sdk:
                with (
                    patch("calendar_bot.evaluation.configure_logging"),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(main(["--list-cases"]), 0)
                dotenv.assert_not_called()
                sdk.assert_not_called()

    def test_clarification_and_order_checks_preserve_atomicity(self):
        task = ParsedTask(kind="task", title="купить товар 1", date="2026-09-25")
        clarification = select_cases(case_ids=["eleven_items"])[0]
        for result in (
            ParseResult(items=[task], question="Разделите сообщение"),
            ParseResult(items=[], question="   "),
            ParseResult(items=[], question=None),
        ):
            with self.assertRaisesRegex(CheckFailed, "expected_clarification"):
                check_result(clarification, result)
        check_result(clarification, ParseResult(items=[], question="Разделите сообщение"))
        ten = select_cases(case_ids=["ten_items"])[0]
        items = [
            task.model_copy(update={"title": f"купить товар {index}"}) for index in range(1, 11)
        ]
        check_result(ten, ParseResult(items=items, question=None))
        with self.assertRaisesRegex(CheckFailed, "field_mismatch:title"):
            check_result(ten, ParseResult(items=items[::-1], question=None))

    async def test_run_records_every_repetition_failure_and_error_without_private_text(self):
        case = select_cases(case_ids=["eleven_items"])[0]
        sdk = SimpleNamespace(
            responses=SimpleNamespace(
                parse=AsyncMock(
                    side_effect=[
                        SimpleNamespace(status="completed", output=[]),
                        RuntimeError("SECRET_ERROR"),
                        SimpleNamespace(status="completed", output=[]),
                    ]
                )
            )
        )
        metrics = EvaluationClient(sdk)
        outputs = iter(
            [
                ParseResult(items=[], question=None),
                ParseResult(items=[], question="SECRET_REPLY"),
            ]
        )

        async def parse(*args):
            await metrics.responses.parse(model="gpt-6-luna", tools=[{}], input="SECRET_INPUT")
            return next(outputs)

        run = EvaluationRun(metrics, [case], repeat=3)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            await run.run(SimpleNamespace(parse=parse))
        report = run.report()
        self.assertTrue(report["finished"])
        self.assertFalse(report["completed"])
        self.assertEqual(
            [entry["status"] for entry in report["cases"]], ["failed", "error", "passed"]
        )
        self.assertEqual([entry["repetition"] for entry in report["attempts"]], [1, 2, 3])
        self.assertTrue(all(entry["case_id"] == case.id for entry in report["attempts"]))
        self.assertEqual(report["cases"][0]["check"], "expected_clarification")
        self.assertEqual(report["outcomes"]["passed"], 1)
        self.assertNotIn("SECRET", json.dumps(report) + stdout.getvalue())
        self.assertEqual(metrics.case_context, {})

    async def test_fail_fast_and_cancellation_leave_remaining_cases_not_run(self):
        cases = select_cases(case_ids=["eleven_items", "every_two_days"])
        metrics = EvaluationClient(None)
        for failure, expected in (
            (RuntimeError("SECRET"), "error"),
            (asyncio.CancelledError(), "cancelled"),
        ):
            run = EvaluationRun(metrics, cases, fail_fast=True)
            parser = SimpleNamespace(parse=AsyncMock(side_effect=failure))
            with contextlib.redirect_stdout(io.StringIO()):
                if expected == "cancelled":
                    with self.assertRaises(asyncio.CancelledError):
                        await run.run(parser)
                else:
                    await run.run(parser)
            report = run.report()
            self.assertEqual([entry["status"] for entry in report["cases"]], [expected, "not_run"])
            self.assertFalse(report["finished"])
            self.assertFalse(report["completed"])
            self.assertEqual(metrics.case_context, {})
            parser.parse.assert_awaited_once()

    async def test_report_written_on_cancellation_and_client_closed(self):
        parser = SimpleNamespace(
            parse=AsyncMock(side_effect=asyncio.CancelledError), close=AsyncMock()
        )
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            with (
                patch("calendar_bot.evaluation.load_dotenv"),
                patch.dict("os.environ", {"OPENAI_API_KEY": "test-not-real"}, clear=True),
                patch("calendar_bot.evaluation.AsyncOpenAI"),
                patch("calendar_bot.evaluation.OpenAIParser", return_value=parser),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(asyncio.CancelledError):
                    await evaluate(case_ids=["eleven_items"], report_path=report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["outcomes"]["cancelled"], 1)
            self.assertFalse(report["completed"])
        parser.close.assert_awaited_once()

    def test_cli_exit_code_distinguishes_quality_failure_from_completed_responses(self):
        for passed in (True, False):
            with (
                patch(
                    "calendar_bot.evaluation.evaluate",
                    new=AsyncMock(return_value={"completed": passed}),
                ),
                patch("calendar_bot.evaluation.configure_logging"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(main([]), 0 if passed else 1)

    async def test_invalid_repeat_and_selection_do_not_load_credentials(self):
        for arguments in ({"repeat": 0}, {"repeat": -1}, {"case_ids": ["unknown"]}):
            with (
                patch("calendar_bot.evaluation.load_dotenv") as dotenv,
                patch("calendar_bot.evaluation.AsyncOpenAI") as sdk,
            ):
                with self.assertRaises(UserError):
                    await evaluate(**arguments)
                dotenv.assert_not_called()
                sdk.assert_not_called()

    def test_report_hashes_change_when_case_expectation_changes(self):
        case = select_cases(case_ids=["eleven_items"])[0]
        first = EvaluationRun(EvaluationClient(None), [case]).report()
        second = EvaluationRun(EvaluationClient(None), [replace(case, messages=[])]).report()
        self.assertNotEqual(first["suite_sha256"], second["suite_sha256"])
        self.assertEqual(first["prompt_sha256"], second["prompt_sha256"])
