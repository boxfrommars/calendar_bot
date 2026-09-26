import unittest
from datetime import UTC, date, datetime, time

from calendar_bot.config import Config, ConfigError
from calendar_bot.domain import (
    EventSource,
    EventSpec,
    UserError,
    expand,
    local_instant,
    parse_reminders,
    zone,
)
from calendar_bot.parser import ParsedEvent, ParsedSource, normalize


class CalendarMathTests(unittest.TestCase):
    def test_public_event_uses_source_zone_with_dst_and_crosses_user_midnight(self):
        source = ParsedSource(title="Организатор", url="https://example.org/schedule")
        reference = datetime(2026, 9, 24, tzinfo=UTC)
        for day, clock, expected in (
            ("2026-09-25", "23:45", datetime(2026, 9, 25, 21, 45, tzinfo=UTC)),
            ("2026-10-24", "20:45", datetime(2026, 10, 24, 18, 45, tzinfo=UTC)),
            ("2026-10-25", "20:45", datetime(2026, 10, 25, 19, 45, tzinfo=UTC)),
        ):
            with self.subTest(day=day):
                spec = normalize(
                    ParsedEvent(
                        kind="event",
                        title="Матч",
                        date=day,
                        time=clock,
                        timezone="Europe/Madrid",
                        repeat="once",
                        weekdays=[],
                        reminders=None,
                        time_source="web",
                        source=source,
                    ),
                    reference,
                    "Asia/Yerevan",
                )
                self.assertEqual(spec.first_after(reference), expected)
                self.assertEqual(EventSpec.from_json(spec.to_json()), spec)
                self.assertEqual(spec.source, source.to_source())
        instant = local_instant(date(2026, 9, 25), time(23, 45), "Europe/Madrid")
        self.assertEqual(instant.astimezone(zone("Asia/Yerevan")).date(), date(2026, 9, 26))

    def test_public_time_requires_source_explicit_zone_date_time_and_one_instant(self):
        fields = dict(
            kind="event",
            title="Матч",
            date="2026-09-25",
            time="20:45",
            timezone="Europe/Berlin",
            repeat="once",
            weekdays=[],
            reminders=None,
            time_source="web",
            source=ParsedSource(title="Организатор", url="https://example.org/schedule"),
        )
        for change in (
            {"source": None},
            {"date": None},
            {"time": None},
            {"timezone": None},
            {"timezone": "Unknown/Timezone"},
            {"repeat": "daily"},
            {"date": "2026-03-29", "time": "02:30"},
            {"date": "2026-10-25", "time": "02:30"},
        ):
            with self.subTest(change=change), self.assertRaises(UserError):
                normalize(ParsedEvent(**(fields | change)), datetime(2026, 1, 1, tzinfo=UTC), "UTC")

    def test_legacy_event_json_has_no_source_and_remains_readable(self):
        spec = EventSpec("Встреча", date(2026, 9, 25), time(12), "UTC")
        self.assertNotIn("source", spec.to_dict())
        self.assertEqual(EventSpec.from_json(spec.to_json()), spec)
        self.assertIsNone(EventSpec.from_dict(spec.to_dict() | {"source": None}).source)

    def test_source_only_allows_valid_http_links(self):
        for url in (
            "javascript:alert(1)",
            "file:///tmp/source",
            "https:///missing-host",
            "https://example.org\n/fake",
            "https://example.org\\@evil.org",
            "https://user:pass@example.org",
            "https://example.org:invalid",
            "https://[bad",
        ):
            with self.subTest(url=url), self.assertRaises(UserError):
                EventSource("Источник", url)
        self.assertEqual(EventSource("Источник", "http://example.org/a?x=1&y=2").title, "Источник")

    def test_original_examples_have_correct_next_occurrences(self):
        reference = datetime(2026, 9, 24, 8, tzinfo=UTC)
        examples = [
            (
                "Content Retro & Planning",
                "weekly",
                [0],
                "14:00",
                None,
                datetime(2026, 9, 28, 10, tzinfo=UTC),
            ),
            ("Оля 1-1", "weekly", [1], "14:00", None, datetime(2026, 9, 29, 10, tzinfo=UTC)),
            (
                "Креаторская рассылки",
                "weekly",
                [2],
                "15:00",
                None,
                datetime(2026, 9, 30, 11, tzinfo=UTC),
            ),
            (
                "Штурм // Ориентир 2027",
                "once",
                [],
                "16:00",
                "2026-09-25",
                datetime(2026, 9, 25, 12, tzinfo=UTC),
            ),
        ]
        for title, repeat, weekdays, clock, day, expected in examples:
            with self.subTest(title=title):
                spec = normalize(
                    ParsedEvent(
                        kind="event",
                        title=title,
                        date=day,
                        time=clock,
                        timezone=None,
                        repeat=repeat,
                        weekdays=weekdays,
                        reminders=None,
                        time_source="user",
                    ),
                    reference,
                    "Asia/Yerevan",
                )
                self.assertEqual(spec.first_after(reference), expected)
                self.assertEqual(spec.title, title)
                self.assertEqual(EventSpec.from_json(spec.to_json()), spec)

    def test_same_weekday_rolls_forward_only_after_time_has_passed(self):
        spec = EventSpec("Retro", date(2026, 9, 28), time(14), "Asia/Yerevan", "weekly", (0,))
        self.assertEqual(
            spec.first_after(datetime(2026, 9, 28, 9, tzinfo=UTC)),
            datetime(2026, 9, 28, 10, tzinfo=UTC),
        )
        self.assertEqual(
            spec.first_after(datetime(2026, 9, 28, 10, tzinfo=UTC)),
            datetime(2026, 10, 5, 10, tzinfo=UTC),
        )

    def test_dst_gap_and_fold_have_explicit_policy(self):
        gap = local_instant(date(2026, 3, 29), time(2, 30), "Europe/Berlin")
        self.assertEqual(gap, datetime(2026, 3, 29, 1, 30, tzinfo=UTC))
        self.assertEqual(gap.astimezone(zone("Europe/Berlin")).hour, 3)
        fold = local_instant(date(2026, 10, 25), time(2, 30), "Europe/Berlin")
        self.assertEqual(fold, datetime(2026, 10, 25, 0, 30, tzinfo=UTC))

    def test_weekdays_and_timezone_survive_dst(self):
        spec = EventSpec(
            "Работа", date(2026, 3, 27), time(9), "Europe/Berlin", "weekly", (0, 1, 2, 3, 4)
        )
        rows = expand(
            "event", spec, {}, datetime(2026, 3, 27, tzinfo=UTC), datetime(2026, 3, 31, tzinfo=UTC)
        )
        self.assertEqual([row.start.hour for row in rows], [8, 7])
        self.assertEqual([row.original_day.weekday() for row in rows], [4, 0])

    def test_overrides_remain_when_weekdays_change(self):
        spec = EventSpec("Серия", date(2026, 9, 24), time(9), "UTC", "weekly", (0,))
        override = EventSpec("Перенос", date(2026, 9, 26), time(11), "UTC")
        rows = expand(
            "id",
            spec,
            {date(2026, 9, 25): override, date(2026, 9, 28): None},
            datetime(2026, 9, 24, tzinfo=UTC),
            datetime(2026, 9, 30, tzinfo=UTC),
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].original_day, date(2026, 9, 25))
        self.assertEqual(rows[0].start, datetime(2026, 9, 26, 11, tzinfo=UTC))

    def test_reminder_validation(self):
        self.assertEqual(parse_reminders("15, 5, 1, 5"), (15, 5, 1))
        self.assertEqual(parse_reminders("нет"), ())
        self.assertIsNone(parse_reminders("общие", inherit=True))
        self.assertEqual(parse_reminders("0"), (0,))
        for invalid in ("-1", "10081", "1,2,3,4,5,6", "пять", ""):
            with self.subTest(value=invalid), self.assertRaises(UserError):
                parse_reminders(invalid)

    def test_missing_time_and_invalid_zone_are_not_guessed(self):
        value = ParsedEvent(
            kind="event",
            title="Встреча",
            date="2026-09-25",
            time=None,
            timezone=None,
            repeat="once",
            weekdays=[],
            reminders=None,
            time_source="user",
        )
        with self.assertRaises(UserError):
            normalize(value, datetime(2026, 9, 24, tzinfo=UTC), "UTC")
        with self.assertRaises(UserError):
            zone("Somewhere/Unknown")

    def test_empty_allowlist_does_not_open_access(self):
        empty = Config.from_env({"BOT_TOKEN": "test", "OPENAI_API_KEY": "test"})
        self.assertEqual(empty.allowed_user_ids, frozenset())
        with self.assertRaises(ConfigError):
            Config.from_env({"ALLOWED_USER_IDS": "not-a-number"}, require_secrets=False)
        config = Config.from_env({"ALLOWED_USER_IDS": "101, 202,101"}, require_secrets=False)
        self.assertEqual(config.allowed_user_ids, frozenset({101, 202}))
