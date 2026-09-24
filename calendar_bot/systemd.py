"""Optional systemd watchdog protocol; no service manager is required locally."""

import logging
import os
import socket
import time
from collections.abc import Mapping

from .config import ConfigError

log = logging.getLogger(__name__)


class SystemdNotifier:
    def __init__(self, address=None, watchdog_seconds=None, *, clock=time.monotonic):
        self.address = address
        self.watchdog_seconds = watchdog_seconds
        self.clock = clock
        self.last_error_at = None

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "SystemdNotifier":
        try:
            usec = int(env.get("WATCHDOG_USEC", "0"))
            pid = int(env.get("WATCHDOG_PID", str(os.getpid())))
            if usec < 0 or pid <= 0:
                raise ValueError
        except ValueError:
            raise ConfigError("Некорректные параметры watchdog systemd.") from None
        if not usec or pid != os.getpid():
            return cls()
        if getattr(socket, "AF_UNIX", None) is None:
            raise ConfigError("Watchdog systemd требует поддержку Unix-сокетов.")
        address = env.get("NOTIFY_SOCKET", "")
        if len(address) < 2 or address[0] not in ("/", "@") or "\0" in address:
            raise ConfigError("Для watchdog systemd нужен корректный NOTIFY_SOCKET.")
        if address.startswith("@"):
            address = "\0" + address[1:]
        return cls(address, usec / 1_000_000)

    @property
    def interval(self) -> float:
        return min(5.0, self.watchdog_seconds / 2) if self.watchdog_seconds else 5.0

    def _send(self, message: bytes) -> None:
        if self.address is None:
            return
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as channel:
                channel.setblocking(False)
                channel.sendto(message, self.address)
        except OSError as exc:
            now = self.clock()
            if self.last_error_at is None or now - self.last_error_at >= 60:
                log.error("systemd_notify_failed type=%s errno=%s", type(exc).__name__, exc.errno)
                self.last_error_at = now

    def ping(self) -> None:
        self._send(b"WATCHDOG=1")

    def stopping(self) -> None:
        self._send(b"STOPPING=1")
