import unittest
from datetime import UTC, date, datetime, time

from calendar_bot.config import Config, ConfigError
from calendar_bot.domain import EventSpec, UserError, expand, local_instant, parse_reminders, zone
from calendar_bot.parser import ParsedEvent, normalize


class CalendarMathTests(unittest.TestCase):
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
                        title=title,
                        date=day,
                        time=clock,
                        timezone=None,
                        repeat=repeat,
                        weekdays=weekdays,
                        reminders=None,
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
            title="Встреча",
            date="2026-09-25",
            time=None,
            timezone=None,
            repeat="once",
            weekdays=[],
            reminders=None,
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
