# example: logging observer
import abc
import logging
from typing import Optional, Any


class ObserverInterface(abc.ABC):
    @abc.abstractmethod
    def observe(self, instance: Any, data: dict) -> None:
        pass


class LogObserver(ObserverInterface):
    def __init__(
        self, logger: Optional[logging.Logger] = None, level: int = logging.DEBUG
    ):
        self.logger = logger or logging.getLogger()
        # https://docs.python.org/3/library/logging.html#logging-levels
        self.level = level

    def observe(self, instance: Any, data: dict) -> None:
        message = f"{instance.__class__.__name__}: {data}"
        # https://docs.python.org/3/library/logging.html#logging.Logger.log
        self.logger.log(self.level, message)
