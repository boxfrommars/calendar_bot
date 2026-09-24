import asyncio

from aiogram.client.session.base import BaseSession
from aiogram.methods import GetMe, GetUpdates, GetWebhookInfo, SetMyCommands
from aiogram.types import User, WebhookInfo

TOKEN = "123456:TEST_ONLY_NOT_A_REAL_TOKEN"
PRIVATE = "SYNTHETIC_PRIVATE_TEXT https://api.telegram.org/bot" + TOKEN


class MonotonicClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class PollingSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.replies = asyncio.Queue()
        self.requested = asyncio.Event()
        self.completed = asyncio.Event()
        self.calls = []
        self.active = 0
        self.closed = False

    async def make_request(self, bot, method, timeout=None):
        if self.closed:
            raise AssertionError("Request after session close")
        if isinstance(method, GetMe):
            return User(id=123456, is_bot=True, first_name="Synthetic", username="synthetic_bot")
        if isinstance(method, GetWebhookInfo):
            return WebhookInfo(url="", has_custom_certificate=False, pending_update_count=0)
        if isinstance(method, SetMyCommands):
            return True
        if not isinstance(method, GetUpdates):
            raise AssertionError(type(method).__name__)
        self.calls.append((method.model_copy(), timeout))
        self.requested.set()
        self.active += 1
        try:
            reply = await self.replies.get()
            if isinstance(reply, Exception):
                raise reply
            return reply
        finally:
            self.active -= 1
            self.completed.set()

    async def close(self):
        if self.active:
            raise AssertionError("Session closed before polling stopped")
        self.closed = True

    async def stream_content(self, *args, **kwargs):
        if False:
            yield b""


class RecordingNotifier:
    interval = 0.005

    def __init__(self):
        self.pings = 0
        self.stops = 0
        self.pinged = asyncio.Event()

    def ping(self):
        self.pings += 1
        self.pinged.set()

    def stopping(self):
        self.stops += 1
