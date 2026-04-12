"""Integration tests for App lifecycle, GenState transitions, and thread error handling.

These tests mock the DiffusionPipeline and OpenCV window to run without GPU or display.
"""

import numpy as np
import threading
import time
import pytest
from PIL import Image

import cv2
import torch

from config import AppConfig, UIConfig, InferenceConfig, RevealConfig
from main import App, GenState


# -- helpers -----------------------------------------------------------------

class _DecodeResult:
    def __init__(self, sample):
        self.sample = sample


class DummyTAESD:
    """Minimal TAESD stand-in that returns the input tensor unchanged."""

    def decode(self, latents_tensor):
        return _DecodeResult(latents_tensor.clamp(0, 1))

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def parameters(self):
        return iter([torch.tensor([1.0])])


class MockPipeline:
    """Stand-in for DiffusionPipeline that produces dummy images without GPU."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._taesd = DummyTAESD()

    @property
    def taesd(self):
        return self._taesd

    @property
    def taesd_device(self):
        return torch.device("cpu")

    def run_inpaint(self, init_image, mask_image, control_image,
                    prompt, num_inference_steps, width, height,
                    step_callback=None):
        for i in range(num_inference_steps):
            if step_callback:
                latents = torch.rand(1, 3, height // 8, width // 8)
                step_callback(i, num_inference_steps, latents)
        return Image.new("RGB", (width, height), (128, 128, 128))


class FailingPipeline(MockPipeline):
    """Pipeline that raises on the first call to run_inpaint, then succeeds."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._fail_count = 1

    def run_inpaint(self, *args, **kwargs):
        if self._fail_count > 0:
            self._fail_count -= 1
            raise RuntimeError("Simulated pipeline failure")
        return super().run_inpaint(*args, **kwargs)


class SlowPipeline(MockPipeline):
    """Pipeline that sleeps to make reset-during-generation reproducible."""

    def run_inpaint(self, *args, **kwargs):
        time.sleep(0.2)
        width = kwargs.get("width", 64)
        height = kwargs.get("height", 64)
        return Image.new("RGB", (width, height), (255, 255, 255))


def _make_cfg():
    return AppConfig(
        ui=UIConfig(image_size=(64, 64), present_size=(64, 64), interp_fps=30),
        inference=InferenceConfig(
            min_inference_steps=2,
            max_inference_steps=4,
            rate_inference_steps_change=2,
            image_sizes_ramp=((64, 64),),
        ),
        # Bypass outro animation so async_diffusion doesn't block on display loop.
        reveal=RevealConfig(reveal_outro_alpha=0),
    )


def _patch_cv_window(monkeypatch):
    monkeypatch.setattr("cv2.namedWindow", lambda *a, **kw: None)
    monkeypatch.setattr("cv2.moveWindow", lambda *a, **kw: None)
    monkeypatch.setattr("cv2.setMouseCallback", lambda *a, **kw: None)


@pytest.fixture
def app(monkeypatch):
    """Create an App with mocked pipeline and OpenCV window."""
    monkeypatch.setattr("main.DiffusionPipeline", MockPipeline)
    _patch_cv_window(monkeypatch)
    return App(_make_cfg())


@pytest.fixture
def failing_app(monkeypatch):
    """Create an App whose pipeline fails on first generation."""
    monkeypatch.setattr("main.DiffusionPipeline", FailingPipeline)
    _patch_cv_window(monkeypatch)
    return App(_make_cfg())


def _run_one_generation(app, timeout=10):
    """Start async_diffusion as a daemon, wait for one cycle, then return."""
    done = threading.Event()
    original_set = app._gen_done.set

    def patched_set():
        original_set()
        done.set()

    app._gen_done.set = patched_set
    app._reset_ack.set()

    t = threading.Thread(target=app.async_diffusion, daemon=True)
    t.start()

    # Wait for either successful completion (_gen_done) or error (_thread_error)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if done.is_set() or app._thread_error is not None:
            break
        time.sleep(0.01)

    # Acknowledge so the thread can finish post-generation bookkeeping
    app._gen_done.clear()
    app._reset_ack.set()
    # Allow thread to complete step ramp and reach its sleep
    time.sleep(0.15)


# -- GenState transitions ----------------------------------------------------

def test_initial_state_is_idle(app):
    assert app._gen_state == GenState.IDLE


def test_drawing_transitions_idle_to_ready(app):
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 20, 50, 0, None)
    assert app._gen_state == GenState.READY


def test_drawing_resets_inference_steps_to_min(app):
    app._inference_steps = 10
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 20, 50, 0, None)
    assert app._inference_steps == app.cfg.inference.min_inference_steps


def test_stroke_stops_when_cursor_leaves_canvas(app):
    # Start inside the canvas.
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 20, 50, 0, None)
    assert app.canvas.drawing is True

    # Move outside canvas bounds (left margin) while still pressed.
    app.mouse_callback(cv2.EVENT_MOUSEMOVE, 5, 50, 0, None)
    assert app.canvas.drawing is False

    # Re-entering without a new press should not resume drawing.
    app.mouse_callback(cv2.EVENT_MOUSEMOVE, 25, 55, 0, None)
    assert app.canvas.drawing is False


def test_stroke_preserved_when_cursor_leaves_canvas(app):
    import numpy as np
    # Start drawing inside the canvas.
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 20, 50, 0, None)
    app.mouse_callback(cv2.EVENT_MOUSEMOVE, 30, 60, 0, None)
    
    # Verify pixels were committed to the mask.
    mask_before = app.canvas.mask.copy()
    
    # Move outside canvas bounds—stroke should be committed, not discarded.
    app.mouse_callback(cv2.EVENT_MOUSEMOVE, 5, 50, 0, None)
    
    # Pixels drawn before leaving canvas should be preserved.
    assert np.any(app.canvas.mask > 0), "Stroke should be preserved in persistent mask"
    

def test_exit_requires_confirmation_click(app, monkeypatch):
    monkeypatch.setattr(app.ui, "hit_test", lambda _x, _y: "exit")

    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)
    assert app.exit_triggered is False
    assert app._exit_confirm_stage == 1
    assert app._exit_confirm_until > 0

    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)
    assert app.exit_triggered is False
    assert app._exit_confirm_stage == 2

    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)
    assert app.exit_triggered is True


def test_exit_confirmation_stage_resets_after_timeout(app, monkeypatch):
    monkeypatch.setattr(app.ui, "hit_test", lambda _x, _y: "exit")
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)
    assert app._exit_confirm_stage == 1

    # Expire the stage window and click again: should re-arm at stage 1.
    app._exit_confirm_until = 0.0
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)
    assert app.exit_triggered is False
    assert app._exit_confirm_stage == 1


def test_adjust_max_inference_steps_clamps_runtime_state(app):
    app._inference_steps = 4
    app.current_inference_steps = 4

    app._adjust_max_inference_steps(-2)
    app._adjust_max_inference_steps(-2)

    assert app._max_inference_steps == app.cfg.inference.min_inference_steps + 0
    assert app._inference_steps == app._max_inference_steps
    assert app.current_inference_steps == app._max_inference_steps


def test_save_sets_ui_notice_with_path(app, monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda _p: False)
    monkeypatch.setattr("cv2.imwrite", lambda _p, _img: True)

    app._save_image()
    notice = app._current_ui_notice()

    assert notice is not None
    assert "Saved snapshot:" in notice
    assert "saved_image_2.png" in notice


def test_reset_from_idle_stays_idle(app):
    app.reset_canvas()
    assert app._gen_state == GenState.IDLE


def test_reset_from_ready_goes_idle(app):
    app._gen_state = GenState.READY
    app.reset_canvas()
    assert app._gen_state == GenState.IDLE


def test_reset_from_generating_goes_resetting(app):
    app._gen_state = GenState.GENERATING
    app.reset_canvas()
    assert app._gen_state == GenState.RESETTING


def test_reset_clears_thread_error(app):
    app._thread_error = "some error"
    app.reset_canvas()
    assert app._thread_error is None


# -- thread error handling ---------------------------------------------------

def test_thread_error_recorded_on_pipeline_failure(failing_app):
    """When the pipeline raises, _thread_error is populated and state recovers."""
    app = failing_app
    app.canvas.mask[30:35, 30:35] = 150
    app._gen_state = GenState.READY

    _run_one_generation(app)

    assert app._thread_error is not None
    assert "Simulated pipeline failure" in app._thread_error
    # State should recover to READY (retryable)
    assert app._gen_state == GenState.READY


def test_thread_error_cleared_on_successful_generation(app):
    """A successful generation clears any previous thread error."""
    app._thread_error = "stale error"
    app.canvas.mask[30:35, 30:35] = 150
    app._gen_state = GenState.READY

    _run_one_generation(app)

    assert app._thread_error is None


# -- full generation cycle ---------------------------------------------------

def test_generation_produces_nonzero_frame(app):
    """A full generation cycle produces a visible display frame."""
    app.canvas.mask[20:40, 20:40] = 150
    app._gen_state = GenState.READY

    _run_one_generation(app)

    assert app._gen_count >= 1
    frame = app.animator.get_display_frame()
    assert frame.shape == (64, 64, 3)
    assert np.any(frame > 0)


def test_inference_steps_ramp_after_generation(app):
    """After generation, inference steps increase by the configured rate."""
    app.canvas.mask[20:40, 20:40] = 150
    app._gen_state = GenState.READY
    initial_steps = app._inference_steps

    _run_one_generation(app)

    expected = min(
        initial_steps + app.cfg.inference.rate_inference_steps_change,
        app.cfg.inference.max_inference_steps,
    )
    assert app._inference_steps == expected


def test_generation_ramp_respects_runtime_max_steps(app):
    app.canvas.mask[20:40, 20:40] = 150
    app._gen_state = GenState.READY
    app._max_inference_steps = 3
    app._inference_steps = 3

    _run_one_generation(app)

    assert app._inference_steps == 3


# -- canvas input validation -------------------------------------------------

def test_canvas_clamps_out_of_bounds_coords(app):
    """Drawing at negative or oversized coords does not crash."""
    app.canvas.begin_stroke(-10, -10)
    app.canvas.continue_stroke(9999, 9999)
    app.canvas.end_stroke(9999, -5)
    # No exception raised; drawing state is clean
    assert app.canvas.drawing is False


def test_reset_during_generation_does_not_commit_stale_result(monkeypatch):
    """Resetting mid-generation should keep the reset canvas, not stale output."""
    monkeypatch.setattr("main.DiffusionPipeline", SlowPipeline)
    _patch_cv_window(monkeypatch)
    app = App(_make_cfg())

    app.canvas.mask[20:40, 20:40] = 150
    app._gen_state = GenState.READY

    t = threading.Thread(target=app.async_diffusion, daemon=True)
    t.start()

    deadline = time.time() + 3.0
    while time.time() < deadline and app._gen_state != GenState.GENERATING:
        time.sleep(0.01)

    app.reset_canvas()

    # Mimic display-loop reset acknowledgement in this integration test.
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if app._gen_done.is_set():
            app._gen_done.clear()
            app._gen_state = GenState.IDLE
            app._reset_ack.set()
            break
        time.sleep(0.01)

    assert np.all(np.array(app.canvas.image) == 0)
