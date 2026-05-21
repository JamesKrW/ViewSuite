"""
Subprocess management utilities.
"""

import logging
import os
import signal
import subprocess
import time

logger = logging.getLogger(__name__)


def kill_process_group(process: subprocess.Popen, timeout: float = 10.0) -> None:
    """
    Kill a subprocess and its entire process group.

    Sequence: SIGTERM -> wait -> SIGKILL if still alive.
    The subprocess must have been started with ``start_new_session=True``.
    """
    if process.poll() is not None:
        return

    try:
        pgid = os.getpgid(process.pid)
        logger.info(f"Sending SIGTERM to process group {pgid} (PID {process.pid})")
        os.killpg(pgid, signal.SIGTERM)

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                logger.info(f"Process {process.pid} terminated gracefully")
                return
            time.sleep(0.5)

        logger.info(f"Sending SIGKILL to process group {pgid} (PID {process.pid})")
        os.killpg(pgid, signal.SIGKILL)
        process.wait(timeout=5)
    except ProcessLookupError:
        pass
    except Exception as e:
        logger.warning(f"Error killing process {process.pid}: {e}")
