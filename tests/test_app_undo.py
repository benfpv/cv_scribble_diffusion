"""Focused tests for App undo behavior."""

import numpy as np
import cv2
import pytest
import torch
from PIL import Image

from config import AppConfig, UIConfig, InferenceConfig, RevealConfig
from main import App, GenState


class _DecodeResult:
    def __init__(self, sample):
        self.sample = sample


class DummyTAESD:
    def decode(self, latents_tensor):
        return _DecodeResult(latents_tensor.clamp(0, 1))

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def parameters(self):
        return iter([torch.tensor([1.0])])


class MockPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self._taesd = DummyTAESD()

    @property
    def taesd(self):
        return self._taesd

    @property
    def taesd_device(self):
        return torch.device("cpu")

    def run_inpaint(self, *args, **kwargs):
        width = kwargs.get("width", 64)
        height = kwargs.get("height", 64)
        return Image.new("RGB", (width, height), (128, 128, 128))


class SlowPipeline(MockPipeline):
    def run_inpaint(self, *args, **kwargs):
        import time
        time.sleep(0.2)
        return super().run_inpaint(*args, **kwargs)


def _make_cfg():
    return AppConfig(
        ui=UIConfig(image_size=(64, 64), present_size=(64, 64), interp_fps=30),
        inference=InferenceConfig(image_sizes_ramp=((64, 64),)),
        reveal=RevealConfig(reveal_outro_alpha=0),
    )


def _patch_cv_window(monkeypatch):
    monkeypatch.setattr("cv2.namedWindow", lambda *a, **kw: None)
    monkeypatch.setattr("cv2.moveWindow", lambda *a, **kw: None)
    monkeypatch.setattr("cv2.setMouseCallback", lambda *a, **kw: None)


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr("main.DiffusionPipeline", MockPipeline)
    _patch_cv_window(monkeypatch)
    return App(_make_cfg())


def test_undo_restores_previous_mask_state(app):
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 20, 50, 0, None)
    app.canvas.commit_active_to_mask()
    app.mouse_callback(cv2.EVENT_LBUTTONUP, 20, 50, 0, None)
    assert np.any(app.canvas.mask)

    app._undo_last_stroke()

    assert not np.any(app.canvas.mask)
    assert app._gen_state == GenState.IDLE


def test_undo_with_empty_history_is_noop(app):
    before = app.canvas.mask.copy()
    app._undo_last_stroke()
    assert np.array_equal(before, app.canvas.mask)


def test_undo_during_generation_restores_after_reset(monkeypatch):
    monkeypatch.setattr("main.DiffusionPipeline", SlowPipeline)
    _patch_cv_window(monkeypatch)
    app = App(_make_cfg())

    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 20, 50, 0, None)
    app.canvas.commit_active_to_mask()
    app.mouse_callback(cv2.EVENT_LBUTTONUP, 20, 50, 0, None)
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


@pytest.mark.parametrize("key_code", [26, ord("z"), ord("Z"), ord("u"), ord("U")])
def test_keyboard_undo_shortcuts_restore_previous_state(app, key_code):
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 20, 50, 0, None)
    app.canvas.commit_active_to_mask()
    app.mouse_callback(cv2.EVENT_LBUTTONUP, 20, 50, 0, None)
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
