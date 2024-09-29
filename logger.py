"""
This module provides a ThreadLogger class that wraps around the loguru.logger.
It allows to bind the log of a thread to a file path.
"""

import threading

from loguru import logger


class ThreadLogger:
    """
    ThreadLogger is a wrapper around loguru.logger.
    It allows to bind the log of a thread to a file path.
    """

    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._logger = logger
            cls._instance._logger.remove()
        return cls._instance

    def bind(self, thread_id: int, path: str, level: str = "INFO") -> None:
        """
        Bind the log of a thread to a file path.
        """
        self._logger.add(
            path,
            filter=lambda record: record["extra"].get("thread") == thread_id,
            level=level,
        )

    def info(self, message: str, *args, **kwargs) -> None:
        """
        Log an info message.
        """
        tid = threading.current_thread().ident
        self._logger.bind(thread=tid).info(message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs) -> None:
        """
        Log a debug message.
        """
        tid = threading.current_thread().ident
        self._logger.bind(thread=tid).debug(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs) -> None:
        """
        Log a warning message.
        """
        tid = threading.current_thread().ident
        self._logger.bind(thread=tid).warning(message, *args, **kwargs)
