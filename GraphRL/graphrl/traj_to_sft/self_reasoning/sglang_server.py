"""Launch and wait for a local sglang OpenAI-compatible server.

Usage::

    with SGLangServer(model_path="Qwen/Qwen2.5-VL-7B-Instruct", port=30000) as srv:
        print(srv.base_url)   # "http://127.0.0.1:30000/v1"
        ...                   # run eval

Robustness notes:

  * **Healthy-external check.** When the port is already in use we don't
    blindly attach — we probe ``/health_generate`` (which actually runs
    a tiny generation) to make sure the inference path is live, not just
    the tokenizer-manager front door. A zombie tokenizer_manager from a
    previously-killed run will pass ``/v1/models`` but hang every chat
    completion request, which used to look like our parent process being
    stuck. Now we raise with explicit cleanup instructions instead.

  * **atexit cleanup.** When *this* process launches sglang, we register
    an atexit handler that kills the subprocess group. This catches
    cases where ``__exit__`` doesn't run — uncaught exceptions, sys.exit,
    most signals — but *not* SIGKILL: if you ``kill -9`` the parent,
    nothing in Python runs and you'll need to clean up sglang manually.
"""
from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SGLangServer:
    model_path: str
    port: int = 30000
    host: str = "0.0.0.0"
    dp_size: int = 1
    tp_size: int = 1
    mem_fraction: float = 0.80
    log_path: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)
    ready_timeout: int = 1800
    poll_interval: float = 5.0
    python_bin: str = sys.executable

    # Set after start()
    proc: Optional[subprocess.Popen] = None
    # True once we've registered an atexit cleanup for ``self.proc``.
    _atexit_registered: bool = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> "SGLangServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        if self._port_in_use():
            # Stricter health check than ``/v1/models`` — that endpoint
            # responds 200 even when the inference scheduler is dead, which
            # leads to silent hangs on every chat completion request.
            if not self._inference_healthy():
                raise RuntimeError(
                    f"Port {self.port} is in use but the server doesn't pass "
                    f"/health_generate. This usually means a previous sglang "
                    f"server was killed but its tokenizer_manager survived as "
                    f"a zombie (it still listens but can't run inference). "
                    f"Clean it up before retrying:\n"
                    f"  ps -ef | grep -E 'sglang|multiprocessing.resource_tracker' | "
                    f"grep -v grep | awk '{{print $2}}' | xargs -r kill -9"
                )
            logger.info(
                "Port %d already in use and healthy; attaching as external sglang.",
                self.port,
            )
            return
        cmd = [
            self.python_bin, "-m", "sglang.launch_server",
            "--host", self.host,
            "--port", str(self.port),
            "--model-path", self.model_path,
            "--dp-size", str(self.dp_size),
            "--tp", str(self.tp_size),
            "--trust-remote-code",
            "--log-level", "warning",
            "--mem-fraction-static", str(self.mem_fraction),
            *self.extra_args,
        ]
        logger.info("Launching sglang: %s", " ".join(cmd))
        log_fp = open(self.log_path, "w") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            cmd, stdout=log_fp, stderr=subprocess.STDOUT, start_new_session=True,
        )
        # Belt-and-braces: kill the subprocess group on interpreter shutdown
        # if ``__exit__`` doesn't get a chance to run (uncaught exception,
        # sys.exit, SIGTERM, …). Doesn't help against SIGKILL.
        if not self._atexit_registered:
            atexit.register(self._atexit_cleanup)
            self._atexit_registered = True
        self._wait_ready(external=False)

    def stop(self) -> None:
        try:
            if self.proc is None or self.proc.poll() is not None:
                return
            logger.info("Stopping sglang (pid=%d)", self.proc.pid)
            try:
                pgid = os.getpgid(self.proc.pid)
            except ProcessLookupError:
                self.proc = None
                return
            os.killpg(pgid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(pgid, signal.SIGKILL)
                self.proc.wait(timeout=10)
            self.proc = None
        finally:
            # Already cleaned up: avoid the atexit hook double-firing.
            if self._atexit_registered:
                try:
                    atexit.unregister(self._atexit_cleanup)
                except Exception:
                    pass
                self._atexit_registered = False

    # ------------------------------------------------------------------
    def _atexit_cleanup(self) -> None:
        """Best-effort kill on interpreter shutdown — never raises."""
        try:
            self.stop()
        except Exception:
            pass

    def _port_in_use(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    def _inference_healthy(self) -> bool:
        """Return True iff the server actually runs a tiny generation.

        sglang's ``/health_generate`` endpoint synthesises a short response
        end-to-end, so it fails when only the tokenizer_manager is alive
        and the data-parallel scheduler is dead. ``/v1/models`` is too
        weak — it returns 200 from a half-dead zombie.
        """
        try:
            r = httpx.get(
                f"http://127.0.0.1:{self.port}/health_generate", timeout=15.0,
            )
            return r.status_code == 200
        except httpx.HTTPError:
            return False

    def _wait_ready(self, *, external: bool) -> None:
        deadline = time.time() + self.ready_timeout
        url = f"{self.base_url}/models"
        while time.time() < deadline:
            if not external and self.proc is not None and self.proc.poll() is not None:
                tail = _tail_log(self.log_path, 200)
                raise RuntimeError(f"sglang exited early. Last log:\n{tail}")
            try:
                r = httpx.get(url, timeout=5.0)
                if r.status_code == 200:
                    logger.info("sglang ready at %s", self.base_url)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(self.poll_interval)
        tail = _tail_log(self.log_path, 200)
        raise TimeoutError(f"sglang not ready after {self.ready_timeout}s. Last log:\n{tail}")


def _tail_log(path: Optional[str], n: int) -> str:
    if not path or not os.path.isfile(path):
        return "(no log)"
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-n:])
