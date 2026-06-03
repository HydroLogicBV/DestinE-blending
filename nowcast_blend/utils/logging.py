import logging

log = logging.getLogger(__name__)


def set_log_format() -> None:
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)


def configure_polytope_logging(level=logging.INFO):
    for logger_name in ("polytope", "polytope.api"):
        logging.getLogger(logger_name).setLevel(level)
    for logger_name in ("urllib3", "urllib3.connectionpool"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


class LoggerStream:
    def __init__(self, logger, level=logging.INFO, prefix=""):
        self.logger = logger
        self.level = level
        self.prefix = prefix
        self.buffer = ""

    def write(self, message):
        self.buffer += message
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._log(line)
        return len(message)

    def flush(self):
        if self.buffer:
            self._log(self.buffer)
            self.buffer = ""

    def _log(self, line):
        line = line.strip()
        if line:
            self.logger.log(self.level, "%s%s", self.prefix, line)
