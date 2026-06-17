import logging


class PrefixedLogger(logging.LoggerAdapter):
    def __init__(
        self,
        prefix: str,
        name: str | None = None,
        logger: logging.Logger | None = None,
        extra=None,
        merge_extra=False,
    ):
        self.prefix = prefix
        logger = logger if logger is not None else logging.getLogger(name)
        super().__init__(logger, extra, merge_extra)

    def process(self, msg, kwargs):
        return f"{self.prefix} {msg}", kwargs
