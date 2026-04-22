import logging
import colorlog
from logging.handlers import TimedRotatingFileHandler

LOG_FORMAT: str = (
    "%(levelname)s %(name)s %(asctime)-15s %(filename)s:%(lineno)d %(message)s"
)


def make_colorlog_stream_handler(
    log_level: int = logging.INFO,
    format: str | None = LOG_FORMAT,
    filter: logging.Filter | None = None,
) -> logging.Handler:
    handler = colorlog.StreamHandler()
    handler.setLevel(log_level)
    handler.setFormatter(colorlog.ColoredFormatter(f"%(log_color)s{format}"))
    if filter is not None:
        handler.addFilter(filter)
    return handler


def make_logging_file_handler(
    filename: str,
    log_level: int = logging.INFO,
    format: str | None = LOG_FORMAT,
    filter: logging.Filter | None = None,
) -> logging.Handler:
    handler = logging.FileHandler(filename)
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(format))
    if filter is not None:
        handler.addFilter(filter)
    return handler


def make_logging_timed_rotating_file_handler(
    filename: str,
    log_level: int = logging.INFO,
    format: str | None = LOG_FORMAT,
    filter: logging.Filter | None = None,
    when: str = "h",
    interval: int = 1,
    backupCount: int = 0,
    encoding: str | None = None,
    delay: bool = False,
    utc: bool = False,
) -> logging.Handler:
    handler = TimedRotatingFileHandler(
        filename=filename,
        when=when,
        interval=interval,
        backupCount=backupCount,
        encoding=encoding,
        delay=delay,
        utc=utc,
    )
    handler.setLevel(log_level)
    handler.setFormatter(logging.Formatter(format))
    if filter is not None:
        handler.addFilter(filter)
    return handler


def setup_logger(
    logger: logging.Logger = logging.root,
    log_level: int = logging.INFO,
    handlers: list[logging.Handler] = [make_colorlog_stream_handler()],
):
    logger.setLevel(log_level)

    if not logging.root.hasHandlers():
        for handler in handlers:
            logging.root.addHandler(handler)
