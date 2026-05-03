"""Tests for terminal startup progress rendering."""

import io
import runpy

import pytest

from cv_scribble_diffusion.app.startup import _StartupProgress


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


def test_startup_main_constructs_app_inside_progress_and_runs(monkeypatch, capsys):
    from cv_scribble_diffusion.app import startup

    events = []

    class FakeStartupProgress:
        def __init__(self, label):
            events.append(("progress_init", label))

        def __enter__(self):
            events.append("progress_enter")
            return self

        def __exit__(self, exc_type, _exc, _tb):
            events.append(("progress_exit", exc_type))
            return False

    class FakeApp:
        def __init__(self):
            events.append("app_init")

        def run(self):
            events.append("app_run")

    monkeypatch.setattr(startup, "StartupProgress", FakeStartupProgress)
    monkeypatch.setattr(startup, "App", FakeApp)

    startup.main()

    assert events == [
        ("progress_init", "[Startup] Loading models and UI"),
        "progress_enter",
        "app_init",
        ("progress_exit", None),
        "app_run",
    ]
    assert "Launching cv_scribble_diffusion" in capsys.readouterr().out


def test_root_launcher_reexports_packaged_startup_main():
    import main as root_main
    from cv_scribble_diffusion.app import startup

    assert root_main.main is startup.main


def test_package_module_entrypoint_invokes_startup_main(monkeypatch):
    from cv_scribble_diffusion.app import startup

    calls = []

    monkeypatch.setattr(startup, "main", lambda: calls.append("called"))

    runpy.run_module("cv_scribble_diffusion.__main__", run_name="__main__")

    assert calls == ["called"]
