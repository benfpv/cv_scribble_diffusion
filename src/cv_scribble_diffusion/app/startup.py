"""Startup helpers and command-line entry points."""

import sys
import threading
import time
from typing import Optional

from cv_scribble_diffusion.app.app import App, logger


class StartupProgress:
    """Small TTY-only progress indicator for slow model startup."""

    def __init__(self, label: str, stream=None, interval_s: float = 0.12, width: int = 22):
        self._label = label
        self._stream = stream if stream is not None else sys.stdout
        self._interval_s = interval_s
        self._width = max(8, width)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started_at = 0.0
        self._tick = 0
        self._last_len = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, _exc, _tb):
        self.stop(success=exc_type is None)
        return False

    @property
    def enabled(self) -> bool:
        isatty = getattr(self._stream, "isatty", None)
        return bool(callable(isatty) and isatty())

    def start(self):
        if not self.enabled:
            return
        self._started_at = time.monotonic()
        self._stop.clear()
        self._draw()
        self._thread = threading.Thread(target=self._run, name="startup-progress", daemon=True)
        self._thread.start()

    def stop(self, success: bool = True):
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s * 2)
        elapsed = time.monotonic() - self._started_at
        message = "Models and UI ready" if success else "Startup failed"
        self._write(f"[Startup] {message} in {elapsed:0.1f}s")
        self._stream.write("\n")
        self._stream.flush()

    def _run(self):
        while not self._stop.wait(self._interval_s):
            self._tick += 1
            self._draw()

    def _draw(self):
        elapsed = max(0.0, time.monotonic() - self._started_at)
        self._write(f"{self._label} {self._bar()} {elapsed:0.1f}s")

    def _bar(self) -> str:
        segment = max(3, self._width // 4)
        cycle = self._width + segment
        head = (self._tick % cycle) - segment
        chars = ["=" if head <= i < head + segment else "." for i in range(self._width)]
        return "[" + "".join(chars) + "]"

    def _write(self, text: str):
        padding = " " * max(0, self._last_len - len(text))
        self._stream.write("\r" + text + padding)
        self._stream.flush()
        self._last_len = len(text)


# Backwards-compatible private alias for tests/older imports during migration.
_StartupProgress = StartupProgress


def main():
    """Launch the OpenCV application."""
    print("...[Startup] Launching cv_scribble_diffusion (loading models and UI)...", flush=True)
    try:
        with StartupProgress("[Startup] Loading models and UI"):
            app = App()
        app.run()
    except Exception:
        logger.exception("Fatal unhandled exception in main thread")
        raise


__all__ = ["StartupProgress", "_StartupProgress", "main"]
