import json
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from calendar_bot.domain import UserError
from calendar_bot.parser import OpenAIParser, ParsedEvent, ParseResult


class ParserContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_sdk_uses_structured_schema_and_minimum_context(self):
        expected = ParseResult(
            events=[
                ParsedEvent(
                    title="Штурм // Ориентир 2027",
                    date="2026-09-25",
                    time="16:00",
                    timezone=None,
                    repeat="once",
                    weekdays=[],
                    reminders=None,
                )
            ],
            question=None,
        )
        call = AsyncMock(return_value=SimpleNamespace(status="completed", output_parsed=expected))
        client = SimpleNamespace(responses=SimpleNamespace(parse=call), close=AsyncMock())
        parser = OpenAIParser("test-not-real", "gpt-5.4-mini-2026-03-17", client=client)
        result = await parser.parse(
            [{"role": "user", "content": "завтра 16:00 Штурм // Ориентир 2027"}],
            datetime(2026, 9, 24, 19, 59, tzinfo=UTC),
            "Asia/Yerevan",
        )
        self.assertEqual(result, expected)
        arguments = call.call_args.kwargs
        self.assertFalse(arguments["store"])
        self.assertIs(arguments["text_format"], ParseResult)
        self.assertNotIn("tools", arguments)
        context = json.loads(arguments["input"][1]["content"].split(": ", 1)[1])
        self.assertEqual(context["reference_local"], "2026-09-24T23:59:00+04:00")
        self.assertIsNone(context["selected_event"])

    async def test_refusal_or_incomplete_response_never_creates_event(self):
        for status in ("completed", "incomplete"):
            with self.subTest(status=status):
                client = SimpleNamespace(
                    responses=SimpleNamespace(
                        parse=AsyncMock(
                            return_value=SimpleNamespace(status=status, output_parsed=None)
                        )
                    )
                )
                parser = OpenAIParser("test-not-real", "model", client=client)
                with self.assertRaises(UserError):
                    await parser.parse(
                        [{"role": "user", "content": "текст"}], datetime.now(UTC), "UTC"
                    )
