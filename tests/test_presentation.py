import unittest
from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta

from calendar_bot import presentation as view
from calendar_bot.domain import EventSpec, local_instant
from tests.support import entity_fragments


class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 9, 24, 8, tzinfo=UTC)
        self.spec = EventSpec(
            "Content Retro & Planning", date(2026, 9, 28), time(14), "Asia/Yerevan", "weekly", (0,)
        )

    def test_draft_matches_chosen_style_and_offsets_after_emoji(self):
        content = view.event_card(
            "Новое событие",
            self.spec,
            self.spec.first_after(self.now),
            "Asia/Yerevan",
            (15, 5, 1),
            self.now,
            draft=True,
        )
        kwargs = content.as_kwargs()
        self.assertIsNone(kwargs["parse_mode"])
        self.assertEqual(
            kwargs["text"],
            "📝 Новое событие\n\nContent Retro & Planning\n📅 пн, 28 сентября · 14:00\n"
            "🔁 По понедельникам\n🔔 За 15, 5 и 1 минуту\n\nЕреван · UTC+4\n"
            "Напоминания из ваших настроек. Черновик действует 24 часа.",
        )
        self.assertEqual(
            entity_fragments(kwargs["text"], kwargs["entities"]),
            [
                ("bold", "Новое событие"),
                ("bold", self.spec.title),
                ("bold", "14:00"),
                ("italic", "Ереван · UTC+4"),
                ("italic", "Напоминания из ваших настроек. Черновик действует 24 часа."),
            ],
        )
        self.assertEqual(kwargs["entities"][0].offset, 3)

    def test_event_date_year_and_dst_offset_use_event_instant(self):
        spec = replace(self.spec, day=date(2027, 1, 4), timezone="Europe/Berlin")
        instant = spec.first_after(self.now)
        content = view.event_card(
            "Новое событие", spec, instant, "Asia/Yerevan", (15,), self.now, draft=True
        )
        text, _ = content.render()
        self.assertIn("пн, 4 января 2027 · 14:00", text)
        self.assertIn("Расписание: пн, 4 января 2027 · 14:00 · Europe/Berlin · UTC+1", text)
        self.assertIn("Ваше время: пн, 4 января 2027 · 17:00 · Ереван · UTC+4", text)
        self.assertNotIn("UTC+2", text)
        self.assertEqual(view.timezone_label("Europe/Berlin", self.now), "Europe/Berlin · UTC+2")

    def test_timezone_labels_include_fractional_and_negative_offsets(self):
        for tz, expected in [
            ("UTC", "UTC"),
            ("Europe/Moscow", "Москва · UTC+3"),
            ("Asia/Kathmandu", "Asia/Kathmandu · UTC+5:45"),
            ("America/St_Johns", "America/St_Johns · UTC-2:30"),
        ]:
            with self.subTest(timezone=tz):
                self.assertEqual(view.timezone_label(tz, self.now), expected)

    def test_single_occurrence_uses_custom_reminders_and_both_local_dates(self):
        spec = EventSpec(
            "Разовая <встреча> & _план_",
            date(2026, 9, 25),
            time(0, 30),
            "Asia/Yerevan",
            reminder_minutes=(0,),
        )
        instant = local_instant(spec.day, spec.clock, spec.timezone)
        content = view.event_card(
            "Эта встреча", spec, instant, "Europe/Moscow", (0,), self.now, single=True
        )
        text, entities = content.render()
        self.assertIn("📅 чт, 24 сентября · 23:30", text)
        self.assertIn("Расписание: пт, 25 сентября · 00:30 · Ереван · UTC+4", text)
        self.assertIn("Ваше время: чт, 24 сентября · 23:30 · Москва · UTC+3", text)
        self.assertNotIn("🔁", text)
        self.assertIn("В момент начала", text)
        self.assertIn("Напоминания для этой встречи.", text)
        self.assertIn(("bold", spec.title), entity_fragments(text, entities))

    def test_week_offset_tracks_dst_change(self):
        now = datetime(2026, 10, 24, 6, tzinfo=UTC)
        spec = replace(self.spec, timezone="Europe/Berlin")
        rows = [
            {"spec": spec.to_json(), "start_at": instant.timestamp()}
            for instant in [now, now + timedelta(days=1)]
        ]
        text, entities = view.agenda(rows, "Europe/Berlin", now, 7, 0).render()
        self.assertIn("1. 08:00", text)
        self.assertIn("2. 07:00", text)
        self.assertIn(
            ("italic", "Время: Europe/Berlin · UTC+2 → UTC+1"), entity_fragments(text, entities)
        )

    def test_reminder_keeps_literal_title_and_inflects_minutes(self):
        title = "😀 <b>14:00</b> & _тест_ > 🎯"
        spec = replace(self.spec, title=title, timezone="Europe/Moscow")
        for minutes, label in [
            (1, "Через 1 минуту"),
            (2, "Через 2 минуты"),
            (5, "Через 5 минут"),
            (11, "Через 11 минут"),
            (21, "Через 21 минуту"),
            (0, "Начинается сейчас"),
        ]:
            with self.subTest(minutes=minutes):
                start = self.now + timedelta(minutes=minutes)
                part = view.reminder_part(spec, start, self.now, "Asia/Yerevan")
                text, entities = view.notification_message("reminder", part).render()
                self.assertEqual(text, part)
                fragments = entity_fragments(text, entities)
                self.assertTrue(text.startswith(f"⏰ {label}\n\n"))
                self.assertNotIn(("bold", label), fragments)
                self.assertIn(("bold", title), fragments)
                self.assertIn("Москва · UTC+3", text)
                self.assertIn(("italic", "Ереван · UTC+4"), fragments)

    def test_summary_formats_only_structural_prefixes_in_old_and_new_parts(self):
        title = "😀 <b>план</b> & _проект_ 14:00 > ☀️ События на завтра"
        for part in [
            f"☀️ События на 24.09.2026 · Asia/Yerevan\n\n14:00 — {title}\n15:00 — Время: тест",
            f"14:00 — {title}\n\n15:00 — Время: тест\n\nВремя: Ереван · UTC+4",
        ]:
            with self.subTest(part=part):
                text, entities = view.notification_message("summary", part).render()
                self.assertEqual(text, part)
                fragments = entity_fragments(text, entities)
                self.assertIn(("bold", "14:00"), fragments)
                self.assertIn(("bold", "15:00"), fragments)
                self.assertFalse(any(title in fragment for _, fragment in fragments))

    def test_summary_splits_whole_utf16_lines_and_formats_each_part(self):
        titles = [f"{index} " + "😀" * 175 for index in range(30)]
        rows = [
            {
                "spec": replace(self.spec, title=title).to_json(),
                "start_at": (self.now + timedelta(minutes=index)).timestamp(),
            }
            for index, title in enumerate(titles)
        ]
        parts = view.summary_parts(rows, self.now.date(), "Asia/Yerevan", self.now)
        self.assertGreater(len(parts), 1)
        event_lines = []
        for part in parts:
            self.assertLessEqual(len(part.encode("utf-16-le")) // 2, 3500)
            text, entities = view.notification_message("summary", part).render()
            self.assertEqual(text, part)
            fragments = entity_fragments(text, entities)
            event_lines.extend(line for line in part.splitlines() if " — " in line)
            self.assertTrue(any(kind == "bold" for kind, _ in fragments))
        self.assertEqual([line.split(" — ", 1)[1] for line in event_lines], titles)
        self.assertEqual("\n".join(parts).count("Время: Ереван · UTC+4"), 1)

    def test_summary_labels_foreign_schedule_and_empty_day(self):
        spec = replace(self.spec, timezone="Europe/Moscow")
        rows = [{"spec": spec.to_json(), "start_at": self.now.timestamp()}]
        part = view.summary_parts(rows, self.now.date(), "Asia/Yerevan", self.now)[0]
        self.assertIn("12:00 — Content Retro & Planning", part)
        self.assertIn("По расписанию: 11:00 · Москва · UTC+3", part)
        self.assertIn("Время: Ереван · UTC+4", part)
        empty = view.summary_parts([], self.now.date(), "Asia/Yerevan", self.now)[0]
        self.assertIn("☀️ События на чт, 24 сентября", empty)
        self.assertIn("На сегодня событий нет.", empty)
