"""Own polling tasks and stop them before handlers, SDKs and storage are closed."""

import asyncio
import logging
import signal
from contextlib import contextmanager

from .polling import POLLING_TIMEOUT

log = logging.getLogger(__name__)
SHUTDOWN_TIMEOUT = 65


@contextmanager
def stop_signals(stop: asyncio.Event):
    loop = asyncio.get_running_loop()
    previous = {}

    def request_stop(signum, frame):
        loop.call_soon_threadsafe(stop.set)

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request_stop)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


async def serve_polling(dispatcher, bot, ui, worker, monitor, begin_shutdown) -> None:
    ready = asyncio.Event()
    worker_stop = asyncio.Event()

    async def started(**kwargs):
        ready.set()

    dispatcher.startup.register(started)
    monitor.set_phase("polling")
    # Do not cancel start_polling: aiogram's own child tasks would survive that.
    polling = asyncio.create_task(
        dispatcher.start_polling(
            bot,
            polling_timeout=POLLING_TIMEOUT,
            allowed_updates=["message", "callback_query"],
            handle_signals=False,
            close_bot_session=False,
        ),
        name="calendar-polling",
    )
    delivery = asyncio.create_task(worker.run(worker_stop), name="calendar-delivery")
    try:
        done, _ = await asyncio.wait((polling, delivery), return_when=asyncio.FIRST_COMPLETED)
        for task, name in ((polling, "polling"), (delivery, "worker")):
            if task in done:
                if not task.cancelled() and (error := task.exception()) is not None:
                    log.error("runtime_task_failed task=%s type=%s", name, type(error).__name__)
                    raise error
                log.error("runtime_task_stopped task=%s", name)
                raise RuntimeError("Unexpected runtime task completion")
    finally:
        begin_shutdown()
        worker_stop.set()
        async with asyncio.timeout(SHUTDOWN_TIMEOUT):
            if not polling.done():
                # stop_polling is valid only after aiogram initialized its stop events.
                readiness = asyncio.create_task(ready.wait())
                try:
                    await asyncio.wait((readiness, polling), return_when=asyncio.FIRST_COMPLETED)
                finally:
                    readiness.cancel()
                    await asyncio.gather(readiness, return_exceptions=True)
                if not polling.done():
                    await dispatcher.stop_polling()
            await asyncio.gather(polling, return_exceptions=True)
            try:
                await ui.shutdown()
            finally:
                await asyncio.gather(delivery, return_exceptions=True)
