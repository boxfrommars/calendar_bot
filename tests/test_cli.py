import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from calendar_bot.storage import check_database


class CommandLineTests(unittest.TestCase):
    def test_explicit_migration_and_readonly_offline_check(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "calendar.sqlite3"
            env = os.environ.copy()
            env.update(
                DATABASE_PATH=str(path),
                BOT_TOKEN="123456:TEST_ONLY_NOT_A_REAL_TOKEN",
                OPENAI_API_KEY="TEST_ONLY_NOT_A_REAL_KEY",
                ALLOWED_USER_IDS="101",
                PYTHONIOENCODING="cp1252",
            )

            def invoke(command):
                return subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "calendar_bot",
                        *command,
                        "--env-file",
                        str(Path(directory) / "absent.env"),
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=30,
                )

            missing = invoke(["check", "--offline"])
            self.assertEqual(missing.returncode, 1)
            self.assertFalse(path.exists())
            help_result = invoke(["--help"])
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn("Личный календарь", help_result.stdout)
            migration = invoke(["migrate"])
            self.assertEqual(migration.returncode, 0, migration.stdout + migration.stderr)
            before = path.read_bytes()
            checked = invoke(["check", "--offline"])
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertEqual(path.read_bytes(), before)
            check_database(path)
            for result in (missing, migration, checked):
                self.assertNotIn(env["BOT_TOKEN"], result.stdout + result.stderr)
                self.assertNotIn(env["OPENAI_API_KEY"], result.stdout + result.stderr)
