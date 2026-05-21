"""
Base class for periodic helper tasks running as daemon threads.
"""

import logging
import threading
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class PeriodicTask(ABC):
    """
    Daemon thread that executes tick() at a fixed interval.

    Provides start/stop lifecycle and graceful shutdown via threading.Event.
    """

    def __init__(self, interval: float, name: str = "PeriodicTask"):
        self._interval = interval
        self._name = name
        self._stop_event = threading.Event()
        self._thread: threading.Thread = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name=self._name
        )
        self._thread.start()
        logger.info(f"[{self._name}] Started (interval={self._interval}s)")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._interval + 10)
        logger.info(f"[{self._name}] Stopped")

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as e:
                logger.warning(f"[{self._name}] Error in tick: {e}", exc_info=True)
            self._stop_event.wait(self._interval)

    @abstractmethod
    def tick(self) -> None:
        """Called periodically. Subclasses implement the actual work here."""
        ...
