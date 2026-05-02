"""Focused tests for App undo behavior."""

import numpy as np
import cv2
import pytest

from config import AppConfig, UIConfig, InferenceConfig, RevealConfig
from main import App, GenState


def _make_cfg():
    return AppConfig(
        ui=UIConfig(image_size=(64, 64), present_size=(64, 64), interp_fps=30),
        inference=InferenceConfig(image_sizes_ramp=((64, 64),)),
        reveal=RevealConfig(reveal_outro_alpha=0),
    )


@pytest.fixture
def app(monkeypatch, patch_cv_window, mock_pipeline_cls):
    monkeypatch.setattr("main.DiffusionPipeline", mock_pipeline_cls)
    return App(_make_cfg())


def _canvas_point(app, x=8, y=10):
    return app.ui.canvas_x_offset + x, app.ui.canvas_y_offset + y


def test_undo_restores_previous_mask_state(app):
    x, y = _canvas_point(app)
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    app.canvas.commit_active_to_mask()
    app.mouse_callback(cv2.EVENT_LBUTTONUP, x, y, 0, None)
    assert np.any(app.canvas.mask)

    app._undo_last_stroke()

    assert not np.any(app.canvas.mask)
    assert app._gen_state == GenState.IDLE


def test_undo_with_empty_history_is_noop(app):
    before = app.canvas.mask.copy()
    app._undo_last_stroke()
    assert np.array_equal(before, app.canvas.mask)


def test_undo_during_generation_restores_after_reset(monkeypatch, patch_cv_window, slow_pipeline_cls):
    monkeypatch.setattr("main.DiffusionPipeline", slow_pipeline_cls)
    app = App(_make_cfg())

    x, y = _canvas_point(app)
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    app.canvas.commit_active_to_mask()
    app.mouse_callback(cv2.EVENT_LBUTTONUP, x, y, 0, None)
    assert np.any(app.canvas.mask)
    assert len(app._undo_stack) == 1

    app._gen_state = GenState.READY
    import threading
    import time
    t = threading.Thread(target=app.async_diffusion, daemon=True)
    t.start()

    deadline = time.time() + 3.0
    while time.time() < deadline and app._gen_state != GenState.GENERATING:
        time.sleep(0.01)

    app._undo_last_stroke()
    assert app._gen_state == GenState.RESETTING
    assert app._pending_restore is not None

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if app._gen_done.is_set():
            app._gen_done.clear()
            snapshot = app._pending_restore
            app._pending_restore = None
            app._restore_snapshot(snapshot)
            app._reset_ack.set()
            break
        time.sleep(0.01)

    assert not np.any(app.canvas.mask)
    assert app._gen_state == GenState.IDLE

    # Stop the thread to prevent it leaking into subsequent tests.
    app._stop_event.set()
    t.join(timeout=2)


@pytest.mark.parametrize("key_code", [26, ord("z"), ord("Z"), ord("u"), ord("U")])
def test_keyboard_undo_shortcuts_restore_previous_state(app, key_code):
    x, y = _canvas_point(app)
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    app.canvas.commit_active_to_mask()
    app.mouse_callback(cv2.EVENT_LBUTTONUP, x, y, 0, None)
    assert np.any(app.canvas.mask)

    app._handle_keypress(key_code)

    assert not np.any(app.canvas.mask)
    assert app._gen_state == GenState.IDLE


def test_keyboard_shortcuts_map_to_documented_actions(app):
    app.mask_visibility_toggle = False
    app._handle_keypress(9)  # Tab
    assert app.mask_visibility_toggle is True

    start_thickness = app.canvas.brush_thickness
    app._handle_keypress(2555904)  # Right arrow
    assert app.canvas.brush_thickness == start_thickness + 1

    app._handle_keypress(2424832)  # Left arrow
    assert app.canvas.brush_thickness == start_thickness
