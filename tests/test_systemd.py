import logging
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from calendar_bot.config import ConfigError
from calendar_bot.systemd import SystemdNotifier
from tests.polling_support import PRIVATE, MonotonicClock


class SystemdTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch.object(socket, "AF_UNIX", getattr(socket, "AF_UNIX", 1), create=True)
        )

    def env(self, **kwargs):
        return {"WATCHDOG_USEC": "30000000", "NOTIFY_SOCKET": "/synthetic/notify", **kwargs}

    def test_optional_watchdog_and_pid_matching(self):
        with patch("calendar_bot.systemd.socket.socket") as channel:
            for env in (
                {},
                self.env(WATCHDOG_USEC="0"),
                self.env(WATCHDOG_PID=str(os.getpid() + 1)),
            ):
                notifier = SystemdNotifier.from_env(env)
                notifier.ping()
                notifier.stopping()
            channel.assert_not_called()
        self.assertEqual(SystemdNotifier.from_env(self.env()).interval, 5)
        self.assertEqual(SystemdNotifier.from_env(self.env(WATCHDOG_USEC="2000000")).interval, 1)

    def test_protocol_uses_only_known_payloads_and_handles_abstract_socket(self):
        for address, expected in (
            ("/synthetic/notify", "/synthetic/notify"),
            ("@synthetic-notify", "\0synthetic-notify"),
        ):
            with self.subTest(address=address):
                notifier = SystemdNotifier.from_env(
                    self.env(NOTIFY_SOCKET=address, WATCHDOG_PID=str(os.getpid()))
                )
                with patch("calendar_bot.systemd.socket.socket") as factory:
                    channel = factory.return_value.__enter__.return_value
                    notifier.ping()
                    channel.sendto.assert_called_with(b"WATCHDOG=1", expected)
                    notifier.stopping()
                    channel.sendto.assert_called_with(b"STOPPING=1", expected)
                    channel.setblocking.assert_called_with(False)

    def test_configuration_errors_do_not_echo_values(self):
        for env in (
            self.env(WATCHDOG_USEC=PRIVATE),
            self.env(WATCHDOG_USEC="-1"),
            self.env(WATCHDOG_PID=PRIVATE),
            self.env(NOTIFY_SOCKET=""),
            self.env(NOTIFY_SOCKET=PRIVATE),
        ):
            with self.subTest(env=tuple(env)), self.assertRaises(ConfigError) as result:
                SystemdNotifier.from_env(env)
            self.assertNotIn(PRIVATE, str(result.exception))

    def test_explicit_watchdog_on_unsupported_platform_fails_safely(self):
        with patch.object(socket, "AF_UNIX", None):
            with self.assertRaises(ConfigError):
                SystemdNotifier.from_env(self.env())
            self.assertIsNone(SystemdNotifier.from_env({}).address)

    def test_socket_failures_are_safe_and_throttled(self):
        clock = MonotonicClock()
        notifier = SystemdNotifier("/synthetic/notify", 30, clock=clock)
        with patch("calendar_bot.systemd.socket.socket", side_effect=OSError(5, PRIVATE)):
            with self.assertLogs("calendar_bot.systemd", level=logging.ERROR) as captured:
                notifier.ping()
                notifier.ping()
                clock.advance(60)
                notifier.ping()
        self.assertEqual(len(captured.records), 2)
        self.assertIn("errno=5", captured.output[0])
        self.assertNotIn(PRIVATE, "\n".join(captured.output))

    @unittest.skipUnless(sys.platform == "linux", "Linux Unix datagram socket integration")
    def test_real_notify_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            address = str(Path(directory) / "notify")
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as receiver:
                receiver.bind(address)
                receiver.settimeout(1)
                notifier = SystemdNotifier.from_env(self.env(NOTIFY_SOCKET=address))
                notifier.ping()
                self.assertEqual(receiver.recv(128), b"WATCHDOG=1")
                notifier.stopping()
                self.assertEqual(receiver.recv(128), b"STOPPING=1")
