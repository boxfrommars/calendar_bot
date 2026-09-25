"""Compare fresh health processes in two source trees using one Linux Python 3.14.

Only synthetic lock/health files are created. Neither source tree's .env is read.
Run outside unit tests; timings are observations, never pass/fail thresholds.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path


def measure(source: Path, env: dict) -> dict:
    # GNU time measures its own child: wait4 on a Python-launched process can
    # include the larger Python parent's RSS before exec and overstate health's peak.
    with tempfile.TemporaryFile(mode="w+") as statistics_file:
        started = time.perf_counter()
        result = subprocess.run(
            [
                "/usr/bin/time",
                "--format=%U %S %M",
                f"--output=/proc/self/fd/{statistics_file.fileno()}",
                sys.executable,
                "-B",
                "-m",
                "calendar_bot",
                "health",
                "--env-file",
                "/dev/null",
            ],
            cwd=source,
            env=env,
            pass_fds=(statistics_file.fileno(),),
            capture_output=True,
            text=True,
            timeout=60,
        )
        elapsed = time.perf_counter() - started
        if (
            result.returncode != 0
            or "status=healthy reason=ok " not in result.stdout
            or result.stderr
        ):
            raise RuntimeError(
                f"Synthetic health failed: {result.returncode}, {result.stdout!r}, {result.stderr!r}"
            )
        statistics_file.seek(0)
        user, system, peak_rss = statistics_file.read().split()
    return {
        "cpu_seconds": float(user) + float(system),
        "wall_seconds": elapsed,
        "peak_rss_kib": int(peak_rss),
    }


def benchmark(before: Path, after: Path, runs: int, warmups: int) -> dict:
    import fcntl

    # One interpreter and dependency set, with no inherited app configuration.
    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT")
        if key in os.environ
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    measurements = {"before": [], "after": []}
    sources = {"before": before, "after": after}
    with tempfile.TemporaryDirectory(prefix="calendar-health-benchmark-") as directory:
        database = Path(directory) / "synthetic.sqlite3"
        env["DATABASE_PATH"] = str(database)
        identity = uuid.uuid4().hex
        stop = threading.Event()
        snapshot = Path(str(database) + ".health.json")
        temporary = snapshot.with_suffix(".json.tmp")
        phase_started = time.monotonic()

        def publish():
            now = time.monotonic()
            temporary.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "instance_id": identity,
                        "pid": os.getpid(),
                        "phase": "polling",
                        "updated": now,
                        "phase_started": phase_started,
                        "last_started": now,
                        "last_completed": now,
                        "last_success": now,
                        "last_error_type": None,
                        "failures": 0,
                    }
                ),
                encoding="utf-8",
            )
            os.replace(temporary, snapshot)

        def refresh():
            while not stop.wait(1):
                publish()

        with Path(str(database) + ".lock").open("w+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock.write(b"0" + identity.encode("ascii"))
            lock.flush()
            publish()
            # Only the fixture is refreshed here; its CPU is excluded by GNU time.
            refresher = threading.Thread(target=refresh)
            refresher.start()
            try:
                for index in range(warmups + runs):
                    order = ("before", "after") if index % 2 == 0 else ("after", "before")
                    for label in order:
                        sample = measure(sources[label], env)
                        if index >= warmups:
                            measurements[label].append(sample)
            finally:
                stop.set()
                refresher.join()
        if database.exists():
            raise AssertionError("Health created a database")
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=2)
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 14):
        parser.error("Use Linux and the target Python 3.14 interpreter")
    if args.runs < 1 or args.warmups < 0:
        parser.error("runs must be positive and warmups nonnegative")
    before, after = args.before.resolve(), args.after.resolve()
    requirements = (before / "requirements.txt").read_text(encoding="utf-8")
    if requirements != (after / "requirements.txt").read_text(encoding="utf-8"):
        parser.error("Use identical requirements.txt in both trees")
    samples = benchmark(before, after, args.runs, args.warmups)
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "requirements_sha256_lf": hashlib.sha256(requirements.encode()).hexdigest(),
        "installed_packages": {
            dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
        },
        "command": "python -B -m calendar_bot health --env-file /dev/null",
        "measurement": "GNU time: CPU user+sys (0.01 s), peak RSS KiB; perf_counter: wall time",
        "runs_per_tree": args.runs,
        "warmups_per_tree": args.warmups,
        "summary": {
            label: {
                metric: {
                    "mean": statistics.fmean(sample[metric] for sample in values),
                    "min": min(sample[metric] for sample in values),
                    "max": max(sample[metric] for sample in values),
                }
                for metric in ("cpu_seconds", "wall_seconds", "peak_rss_kib")
            }
            for label, values in samples.items()
        },
        "samples": samples,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
