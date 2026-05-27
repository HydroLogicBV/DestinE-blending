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
