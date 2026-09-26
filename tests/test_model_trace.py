import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from calendar_bot.config import Config
from calendar_bot.domain import UserError
from calendar_bot.model_trace import ModelTrace
from calendar_bot.parser import OpenAIParser, ParseResult
from tests.test_parser import search_output, web_event


class ModelTraceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "traces"
        self.now = datetime(2026, 9, 26, 17, 47, tzinfo=UTC)
        self.trace = ModelTrace(self.path, clock=lambda: self.now)

    def records(self):
        return [
            json.loads(line)
            for file in self.path.glob("*.jsonl")
            for line in file.read_text(encoding="utf-8").splitlines()
        ]

    async def test_trace_contains_inputs_replies_sources_and_utc_but_not_provider_secrets(self):
        wrong = web_event(date="2026-09-26", time="19:45", timezone="Asia/Yerevan")
        correct = wrong.model_copy(update={"timezone": "Europe/London"})
        responses = [
            SimpleNamespace(
                id="response_test",
                status="completed",
                error="SECRET_ERROR_BODY",
                headers="SECRET_HEADER",
                output_parsed=ParseResult(items=[item], question=None),
                output=[
                    search_output(),
                    SimpleNamespace(type="reasoning", encrypted_content="SECRET_REASONING"),
                    SimpleNamespace(
                        type="message",
                        role="assistant",
                        content=[SimpleNamespace(type="output_text", text="typed result")],
                    ),
                ],
            )
            for item in (wrong, correct)
        ]
        call = AsyncMock(side_effect=responses)
        parser = OpenAIParser(
            "SECRET_API_KEY",
            "model",
            client=SimpleNamespace(responses=SimpleNamespace(parse=call)),
            trace=self.trace,
        )
        with patch("calendar_bot.parser.REASONING_EFFORT", "medium"):
            await parser.parse(
                [{"role": "user", "content": "напомни о матче"}], self.now, "Asia/Yerevan"
            )
        entries = self.records()
        self.assertEqual(
            [entry["kind"] for entry in entries], ["request", "response", "request", "response"]
        )
        self.assertEqual(len(list(self.path.glob("*.jsonl"))), 1)
        for entry, request in zip(entries[::2], call.await_args_list, strict=True):
            self.assertEqual(entry["reasoning"], {"effort": "medium"})
            self.assertEqual(entry["reasoning"], request.kwargs["reasoning"])
        if os.name != "nt":
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o700)
            self.assertEqual(list(self.path.glob("*.jsonl"))[0].stat().st_mode & 0o777, 0o600)
        self.assertEqual(entries[0]["input"][-1]["content"], "напомни о матче")
        self.assertEqual(entries[-1]["calculated"][0]["start_utc"], "2026-09-26T18:45:00+00:00")
        self.assertEqual(entries[-1]["calculated"][0]["start_user"], "2026-09-26T22:45:00+04:00")
        self.assertFalse(entries[1]["calculated"][0]["future_at_reference"])
        self.assertTrue(entries[-1]["calculated"][0]["future_at_reference"])
        self.assertEqual(entries[-1]["output"][0]["sources"], [correct.source.url])
        self.assertNotIn("SECRET", json.dumps(entries))

    async def test_trace_off_by_default_and_config_does_not_create_directory(self):
        config = Config.from_env({"OPENAI_TRACE_DIR": str(self.path)}, require_secrets=False)
        self.assertEqual(config.openai_trace_dir, self.path.resolve())
        self.assertFalse(self.path.exists())
        self.assertIsNone(Config.from_env({}, require_secrets=False).openai_trace_dir)
        parser = OpenAIParser(
            "SECRET",
            "model",
            client=SimpleNamespace(
                responses=SimpleNamespace(
                    parse=AsyncMock(
                        return_value=SimpleNamespace(
                            status="completed",
                            output_parsed=ParseResult(items=[], question="Когда?"),
                            output=[],
                        )
                    )
                )
            ),
        )
        await parser.parse([], self.now, "Asia/Yerevan")
        self.assertFalse(self.path.exists())

    async def test_failure_records_only_safe_error_type(self):
        parser = OpenAIParser(
            "SECRET",
            "model",
            client=SimpleNamespace(
                responses=SimpleNamespace(
                    parse=AsyncMock(side_effect=TimeoutError("SECRET_ERROR_BODY"))
                )
            ),
            trace=self.trace,
        )
        with self.assertRaises(UserError):
            await parser.parse([], self.now, "Asia/Yerevan")
        entries = self.records()
        self.assertEqual(entries[-1]["error_type"], "TimeoutError")
        self.assertNotIn("SECRET", json.dumps(entries))

    async def test_trace_write_failure_does_not_break_parse_or_log_contents(self):
        parser = OpenAIParser(
            "SECRET",
            "model",
            client=SimpleNamespace(
                responses=SimpleNamespace(
                    parse=AsyncMock(
                        return_value=SimpleNamespace(
                            status="completed",
                            output_parsed=ParseResult(items=[], question="Во сколько?"),
                            output=[],
                        )
                    )
                )
            ),
            trace=self.trace,
        )
        with patch.object(self.trace, "_write", side_effect=PermissionError("SECRET_PATH")):
            with self.assertLogs("calendar_bot.model_trace", level="WARNING") as logs:
                result = await parser.parse(
                    [{"role": "user", "content": "PRIVATE_TEXT"}], self.now, "Asia/Yerevan"
                )
        self.assertEqual(result.question, "Во сколько?")
        self.assertIn("PermissionError", " ".join(logs.output))
        self.assertNotIn("SECRET", " ".join(logs.output))
        self.assertNotIn("PRIVATE", " ".join(logs.output))

    async def test_incomplete_tool_response_is_recorded_without_changing_retry_error(self):
        parser = OpenAIParser(
            "SECRET",
            "model",
            client=SimpleNamespace(
                responses=SimpleNamespace(
                    parse=AsyncMock(
                        return_value=SimpleNamespace(
                            status="incomplete",
                            output_parsed=None,
                            output=[SimpleNamespace(type="web_search_call", status="failed")],
                        )
                    )
                )
            ),
            trace=self.trace,
        )
        with self.assertRaisesRegex(UserError, "Не удалось разобрать сообщение"):
            await parser.parse([], self.now, "Asia/Yerevan")
        self.assertEqual(self.records()[-1]["status"], "incomplete")

    async def test_retention_only_prunes_owned_files_and_caps_total_size(self):
        self.path.mkdir()
        old = self.path / f"model-trace-{'a' * 32}.jsonl"
        old.write_text("old")
        os.utime(old, (self.now.timestamp() - 172800,) * 2)
        unrelated = self.path / "keep.txt"
        unrelated.write_text("keep")
        await self.trace.record(self.trace.new_session(), {"kind": "request"})
        self.assertFalse(old.exists())
        self.assertTrue(unrelated.exists())
        for index in range(3):
            file = self.path / f"model-trace-{index:032x}.jsonl"
            file.write_text("x" * 600)
            os.utime(file, (self.now.timestamp() - 60,) * 2)
        with patch("calendar_bot.model_trace.MAX_TOTAL_BYTES", 1024):
            await self.trace.record(self.trace.new_session(), {"kind": "request"})
        self.assertLessEqual(sum(file.stat().st_size for file in self.path.glob("*.jsonl")), 1024)
        self.assertTrue(unrelated.exists())

    async def test_oversized_record_is_explicitly_truncated_valid_json(self):
        with patch("calendar_bot.model_trace.MAX_RECORD_BYTES", 1024):
            await self.trace.record(
                self.trace.new_session(), {"kind": "response", "output": "😀" * 2000}
            )
        self.assertTrue(self.records()[0]["truncated"])
        self.assertLess(list(self.path.glob("*.jsonl"))[0].stat().st_size, 1024)
