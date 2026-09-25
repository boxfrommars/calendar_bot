"""Observe actual getUpdates attempts without replacing aiogram's retry loop."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

from aiogram.client.session.middlewares.base import BaseRequestMiddleware
from aiogram.methods import GetUpdates

from .health import ERROR_TYPE, health_path
from .systemd import SystemdNotifier

log = logging.getLogger(__name__)

POLLING_TIMEOUT = 10
REQUEST_TIMEOUT = 60
SNAPSHOT_INTERVAL = 5
PROGRESS_MAX_AGE = 90
STARTUP_MAX_AGE = 90
ERROR_LOG_INTERVAL = 60


class PollingMonitor(BaseRequestMiddleware):
    def __init__(self, database_path: Path, instance_id: str, *, clock=time.monotonic):
        self.path = health_path(database_path)
        self.instance_id = instance_id
        self.clock = clock
        self.phase = "starting"
        self.phase_started = clock()
        self.last_started = None
        self.last_completed = None
        self.last_success = None
        self.last_error_type = None
        self.failures = 0
        self.failure_started = None
        self.last_error_log = None
        self.last_write_error = None
        self.stalled = False

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.phase_started = self.clock()
        self.stalled = False
        self.publish()

    async def __call__(self, make_request, bot, method):
        if not isinstance(method, GetUpdates):
            return await make_request(bot, method)
        self.last_started = self.clock()
        try:
            result = await make_request(bot, method)
        except Exception as exc:
            now = self.clock()
            self.last_completed = now
            error_type = type(exc).__name__
            if not ERROR_TYPE.fullmatch(error_type):
                error_type = "UnknownError"
            self.failures += 1
            if self.failure_started is None:
                self.failure_started = now
            if (
                error_type != self.last_error_type
                or self.last_error_log is None
                or now - self.last_error_log >= ERROR_LOG_INTERVAL
            ):
                log.error(
                    "polling_request_failed type=%s failures=%s duration=%.3f",
                    error_type,
                    self.failures,
                    now - self.last_started,
                )
                self.last_error_log = now
            self.last_error_type = error_type
            raise
        else:
            now = self.clock()
            self.last_completed = self.last_success = now
            if self.failures:
                log.info(
                    "polling_recovered failures=%s duration=%.3f",
                    self.failures,
                    now - self.failure_started,
                )
            self.failures = 0
            self.last_error_type = self.failure_started = self.last_error_log = None
            return result

    def progressing(self) -> bool:
        if self.phase == "starting":
            age, limit = self.clock() - self.phase_started, STARTUP_MAX_AGE
        elif self.phase == "polling":
            progress = max(
                t
                for t in (self.phase_started, self.last_started, self.last_completed)
                if t is not None
            )
            age, limit = self.clock() - progress, PROGRESS_MAX_AGE
        else:
            return False
        if age < limit:
            self.stalled = False
            return True
        if not self.stalled:
            log.error("polling_stalled phase=%s duration=%.3f", self.phase, age)
            self.stalled = True
        return False

    def publish(self) -> None:
        snapshot = {
            "version": 1,
            "instance_id": self.instance_id,
            "pid": os.getpid(),
            "phase": self.phase,
            "updated": self.clock(),
            "phase_started": self.phase_started,
            "last_started": self.last_started,
            "last_completed": self.last_completed,
            "last_success": self.last_success,
            "last_error_type": self.last_error_type,
            "failures": self.failures,
        }
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            temporary.write_text(json.dumps(snapshot, allow_nan=False), encoding="utf-8")
            os.replace(temporary, self.path)
        except OSError as exc:
            now = self.clock()
            if self.last_write_error is None or now - self.last_write_error >= ERROR_LOG_INTERVAL:
                log.error("polling_health_write_failed type=%s", type(exc).__name__)
                self.last_write_error = now

    async def run(self, notifier: SystemdNotifier, stop: asyncio.Event) -> None:
        next_snapshot = self.clock()
        while not stop.is_set():
            if self.clock() >= next_snapshot:
                self.publish()
                next_snapshot = self.clock() + SNAPSHOT_INTERVAL
            if self.progressing():
                notifier.ping()
            try:
                await asyncio.wait_for(stop.wait(), timeout=notifier.interval)
            except TimeoutError:
                pass
