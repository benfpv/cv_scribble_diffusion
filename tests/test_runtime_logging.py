"""Tests for runtime logging configuration behavior."""

import logging
from logging.handlers import RotatingFileHandler

from config import LoggingConfig
import runtime_logging as runtime_logging_module


def test_configure_logging_disabled_is_noop(tmp_path):
    root = logging.getLogger()
    before_handlers = list(root.handlers)
    runtime_logging_module._CONFIGURED = False

    cfg = LoggingConfig(enabled=False, dir=str(tmp_path))
    runtime_logging_module.configure_logging(cfg)

    assert runtime_logging_module._CONFIGURED is False
    assert list(root.handlers) == before_handlers


def test_configure_logging_installs_file_and_console_handlers(tmp_path):
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level

    try:
        runtime_logging_module._CONFIGURED = False
        cfg = LoggingConfig(
            enabled=True,
            dir=str(tmp_path),
            file_name="runtime.log",
            level="DEBUG",
            console_level="INFO",
            max_bytes=1024,
            backup_count=3,
        )
        runtime_logging_module.configure_logging(cfg)

        assert runtime_logging_module._CONFIGURED is True
        assert len(root.handlers) == 2
        assert any(isinstance(h, RotatingFileHandler) for h in root.handlers)
        assert any(type(h) is logging.StreamHandler for h in root.handlers)

        file_handler = next(h for h in root.handlers if isinstance(h, RotatingFileHandler))
        assert file_handler.baseFilename.endswith("runtime.log")
        assert file_handler.maxBytes == 1024
        assert file_handler.backupCount == 3
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        root.handlers[:] = old_handlers
        root.setLevel(old_level)
        runtime_logging_module._CONFIGURED = False


def test_configure_logging_is_idempotent(tmp_path):
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level

    try:
        runtime_logging_module._CONFIGURED = False
        cfg = LoggingConfig(enabled=True, dir=str(tmp_path), file_name="a.log")
        runtime_logging_module.configure_logging(cfg)
        first_handlers = list(root.handlers)

        runtime_logging_module.configure_logging(cfg)

        assert list(root.handlers) == first_handlers
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        root.handlers[:] = old_handlers
        root.setLevel(old_level)
        runtime_logging_module._CONFIGURED = False


def test_parse_level_falls_back_to_default_for_invalid_name():
    default_level = logging.WARNING
    assert runtime_logging_module._parse_level("NOT_A_LEVEL", default_level) == default_level
    assert runtime_logging_module._parse_level("debug", default_level) == logging.DEBUG
