"""Read polling health and its current OS-lock owner without loading bot SDKs."""

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .locking import locked_instance

SNAPSHOT_MAX_AGE = 15
SUCCESS_MAX_AGE = 120
ERROR_TYPE = re.compile(r"[A-Za-z_][A-Za-z_0-9]{0,79}\Z")


def health_path(database_path: Path) -> Path:
    return database_path.with_name(database_path.name + ".health.json")


@dataclass(frozen=True)
class HealthResult:
    reason: str
    last_success_age: float | None = None
    error_type: str | None = None

    @property
    def healthy(self) -> bool:
        return self.reason == "ok"

    def describe(self) -> str:
        age = "none" if self.last_success_age is None else f"{self.last_success_age:.3f}"
        status = "healthy" if self.healthy else "unhealthy"
        return (
            f"polling_health status={status} reason={self.reason} "
            f"last_success_age={age} error_type={self.error_type or 'none'}"
        )


def _valid_snapshot(data, now: float) -> bool:
    if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
        return False
    if data.get("phase") not in ("starting", "polling", "stopping", "stopped"):
        return False
    if type(data.get("pid")) is not int or data["pid"] <= 0:
        return False
    if type(data.get("failures")) is not int or data["failures"] < 0:
        return False
    if "last_error_type" not in data:
        return False
    for key in ("updated", "phase_started", "last_started", "last_completed", "last_success"):
        if key not in data:
            return False
        value = data[key]
        if value is None and key.startswith("last_"):
            continue
        if type(value) not in (int, float) or not 0 <= value <= now or not math.isfinite(value):
            return False
        if key != "updated" and value > data["updated"]:
            return False
    error = data.get("last_error_type")
    if data["last_success"] is not None and (
        data["last_completed"] is None or data["last_success"] > data["last_completed"]
    ):
        return False
    return error is None or isinstance(error, str) and ERROR_TYPE.fullmatch(error) is not None


def check_health(database_path: Path, *, clock=time.monotonic) -> HealthResult:
    try:
        identity = locked_instance(database_path)
        if identity is None:
            return HealthResult("not_running")
        with health_path(database_path).open("rb") as file:
            content = file.read(8193)
        if len(content) > 8192:
            return HealthResult("snapshot_invalid")
        data = json.loads(content)
        now = clock()
        if not _valid_snapshot(data, now):
            return HealthResult("snapshot_invalid")
        if data.get("instance_id") != identity or locked_instance(database_path) != identity:
            return HealthResult("instance_changed")
    except FileNotFoundError:
        return HealthResult("snapshot_missing")
    except OSError:
        return HealthResult("state_unreadable")
    except ValueError, UnicodeError, RecursionError:
        return HealthResult("snapshot_invalid")
    success_age = None if data["last_success"] is None else now - data["last_success"]
    error = data["last_error_type"]
    if now - data["updated"] > SNAPSHOT_MAX_AGE:
        reason = "snapshot_stale"
    elif data["phase"] != "polling":
        reason = data["phase"]
    elif success_age is None:
        reason = "no_success"
    elif success_age > SUCCESS_MAX_AGE:
        reason = "last_success_stale"
    else:
        reason = "ok"
    return HealthResult(reason, success_age, error)
