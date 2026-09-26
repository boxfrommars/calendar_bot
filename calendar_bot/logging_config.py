import logging
import sys
import warnings


def configure_logging() -> None:
    # OpenAI's response parser serializes tool sources. New source variants can
    # make Pydantic put response fragments into a warning on stderr.
    warnings.filterwarnings(
        "ignore",
        message=r"^Pydantic serializer warnings:",
        category=UserWarning,
        module=r"^pydantic\.main$",
    )
    # Windows may default redirected streams to a legacy encoding. Our CLI
    # speaks Russian, so use the same encoding for terminals, pipes and journals.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    # SDK error bodies can contain a token in a URL or user-supplied event text.
    # GetUpdates failures are observed safely by calendar_bot.polling middleware.
    for name in ("httpx", "httpcore", "openai", "aiogram", "aiohttp"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
