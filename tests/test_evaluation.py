import asyncio
import json
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from calendar_bot.domain import UserError
from calendar_bot.evaluation import EvaluationClient, evaluate_edits
from calendar_bot.parser import ParsedEvent, ParseResult


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
        with self.assertRaisesRegex(UserError, "другие поля"):
            await evaluate_edits(parser, datetime(2026, 9, 24, 8, tzinfo=UTC))
