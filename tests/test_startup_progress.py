"""Tests for terminal startup progress rendering."""

import io

import pytest

from main import _StartupProgress


class _TTYBuffer(io.StringIO):
    def isatty(self):
        return True


class _NonTTYBuffer(io.StringIO):
    def isatty(self):
        return False


def test_startup_progress_renders_loading_bar_and_success_message():
    stream = _TTYBuffer()

    progress = _StartupProgress(
        "[Startup] Loading models and UI", stream=stream, interval_s=10.0, width=12,
    )
    progress.start()
    progress.stop(success=True)

    output = stream.getvalue()
    assert "[Startup] Loading models and UI" in output
    assert "[" in output and "]" in output
    assert "Models and UI ready" in output
    assert output.endswith("\n")


def test_startup_progress_reports_failure_before_reraising():
    stream = _TTYBuffer()

    with pytest.raises(RuntimeError):
        with _StartupProgress("[Startup] Loading models and UI", stream=stream, interval_s=10.0):
            raise RuntimeError("boom")

    assert "Startup failed" in stream.getvalue()


def test_startup_progress_is_silent_for_non_tty_output():
    stream = _NonTTYBuffer()

    with _StartupProgress("[Startup] Loading models and UI", stream=stream):
        pass

    assert stream.getvalue() == ""
