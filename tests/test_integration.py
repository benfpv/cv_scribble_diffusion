"""Integration tests for App lifecycle, GenState transitions, and thread error handling.

These tests use the mock pipelines from conftest.py to run without GPU or display.
"""

import numpy as np
import threading
import time
import pytest

import cv2

from config import AppConfig, UIConfig, InferenceConfig, RevealConfig
from main import App, GenState


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


@pytest.fixture
def app(monkeypatch, patch_cv_window, mock_pipeline_cls):
    """Create an App with mocked pipeline and OpenCV window."""
    monkeypatch.setattr("main.DiffusionPipeline", mock_pipeline_cls)
    return App(_make_cfg())


@pytest.fixture
def failing_app(monkeypatch, patch_cv_window, failing_pipeline_cls):
    """Create an App whose pipeline fails on first generation."""
    monkeypatch.setattr("main.DiffusionPipeline", failing_pipeline_cls)
    return App(_make_cfg())


def _run_one_generation(app, timeout=10):
    """Start async_diffusion as a daemon, wait for one cycle, then stop the thread."""
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

    # Stop the thread to prevent it leaking into subsequent tests.
    app._stop_event.set()
    t.join(timeout=2)


def _canvas_point(app, x=8, y=10):
    return app.ui.canvas_x_offset + x, app.ui.canvas_y_offset + y


# -- GenState transitions ----------------------------------------------------

def test_initial_state_is_idle(app):
    assert app._gen_state == GenState.IDLE


def test_drawing_transitions_idle_to_ready(app):
    x, y = _canvas_point(app)
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    assert app._gen_state == GenState.READY


def test_drawing_resets_inference_steps_to_min(app):
    app._inference_steps = 10
    x, y = _canvas_point(app)
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    assert app._inference_steps == app.cfg.inference.min_inference_steps


def test_stroke_stops_when_cursor_leaves_canvas(app):
    # Start inside the canvas.
    x, y = _canvas_point(app)
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    assert app.canvas.drawing is True

    # Move outside canvas bounds (left margin) while still pressed.
    app.mouse_callback(cv2.EVENT_MOUSEMOVE, 5, 50, 0, None)
    assert app.canvas.drawing is False

    # Re-entering without a new press should not resume drawing.
    reenter_x, reenter_y = _canvas_point(app, 13, 15)
    app.mouse_callback(cv2.EVENT_MOUSEMOVE, reenter_x, reenter_y, 0, None)
    assert app.canvas.drawing is False


def test_stroke_preserved_when_cursor_leaves_canvas(app):
    import numpy as np
    # Start drawing inside the canvas.
    x, y = _canvas_point(app)
    move_x, move_y = _canvas_point(app, 18, 20)
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    app.mouse_callback(cv2.EVENT_MOUSEMOVE, move_x, move_y, 0, None)
    
    # Commit active strokes so they land in the persistent mask.
    app.canvas.commit_active_to_mask()
    mask_before = app.canvas.mask.copy()
    assert np.count_nonzero(mask_before) > 0, "Stroke should be in persistent mask"
    
    # Move outside canvas bounds—stroke should be committed, not discarded.
    app.mouse_callback(cv2.EVENT_MOUSEMOVE, 5, 50, 0, None)
    
    # Pixels drawn before leaving canvas must still be present.
    assert np.array_equal(app.canvas.mask >= mask_before, np.ones_like(mask_before, dtype=bool)), \
        "Leaving canvas must not erase previously committed strokes"

def test_exit_requires_confirmation_click(app, monkeypatch):
    monkeypatch.setattr(app.ui, "hit_test", lambda _x, _y: "exit")

    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)
    assert app.exit_triggered is False
    assert app._exit_confirm_stage == 1
    assert app._exit_confirm_until > 0

    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)
    assert app.exit_triggered is True


def test_escape_requires_confirmation(app):
    app._handle_keypress(27)
    assert app.exit_triggered is False
    assert app._exit_confirm_stage == 1

    app._handle_keypress(27)
    assert app.exit_triggered is True


def test_exit_confirmation_can_be_confirmed_across_inputs(app, monkeypatch):
    monkeypatch.setattr(app.ui, "hit_test", lambda _x, _y: "exit")

    app._handle_keypress(27)
    assert app.exit_triggered is False

    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, None)
    assert app.exit_triggered is True


def test_exit_confirmation_cancelled_by_other_action(app):
    app._handle_keypress(27)
    assert app._exit_confirm_stage == 1

    app._handle_keypress(9)  # Tab toggles mask visibility
    assert app.exit_triggered is False
    assert app._exit_confirm_stage == 0
    assert app._exit_confirm_until == 0.0


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


def test_ui_generation_progress_is_zero_when_idle(app):
    app._gen_state = GenState.IDLE
    app.animator._generation_progress = 1.0
    assert app._ui_generation_progress() == 0.0


def test_ui_generation_progress_clamps_when_active(app):
    app._gen_state = GenState.READY
    app.animator._generation_progress = 1.7
    assert app._ui_generation_progress() == 1.0


def test_save_sets_ui_notice_with_path(app, monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda _p: False)
    monkeypatch.setattr("cv2.imwrite", lambda _p, _img: True)

    app._save_image()
    notice = app._current_ui_notice()

    assert notice is not None
    assert "Saved snapshot:" in notice
    assert "saved_image_" in notice
    assert notice.endswith(".png")


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


def test_thread_error_persists_across_successful_generations(app):
    """Thread errors are sticky: only reset/undo clears them so transient
    failures stay visible to the user across subsequent successful runs."""
    app._thread_error = "stale error"
    app.canvas.mask[30:35, 30:35] = 150
    app._gen_state = GenState.READY

    _run_one_generation(app)

    assert app._thread_error == "stale error"


def test_thread_error_cleared_by_reset(app):
    """Resetting the canvas clears any sticky thread error."""
    app._thread_error = "stale error"
    app.reset_canvas()
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
    # MockPipeline produces (128,128,128) fill—at least some pixels should reflect that
    assert frame.mean() > 10


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


def test_reset_during_generation_does_not_commit_stale_result(monkeypatch, patch_cv_window, slow_pipeline_cls):
    """Resetting mid-generation should keep the reset canvas, not stale output."""
    monkeypatch.setattr("main.DiffusionPipeline", slow_pipeline_cls)
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

    # Canvas should be fully blank after reset—no stale pipeline output.
    assert np.all(np.array(app.canvas.image) == 0), \
        "Canvas image must be blank after reset during generation"
    assert np.all(app.canvas.mask == 0), \
        "Canvas mask must be blank after reset during generation"

    # Stop the thread to prevent it leaking into subsequent tests.
    app._stop_event.set()
    t.join(timeout=2)


# -- backoff and consecutive failures ----------------------------------------

def test_consecutive_failure_increments_and_records_error(
    monkeypatch, patch_cv_window, always_failing_pipeline_cls,
):
    """AlwaysFailingPipeline should increment _consecutive_failures each cycle."""
    monkeypatch.setattr("main.DiffusionPipeline", always_failing_pipeline_cls)
    app = App(_make_cfg())
    app.canvas.mask[20:30, 20:30] = 150
    app._gen_state = GenState.READY
    app._reset_ack.set()

    t = threading.Thread(target=app.async_diffusion, daemon=True)
    t.start()

    # Wait until at least 3 failures accumulate
    deadline = time.time() + 10
    while time.time() < deadline and app._consecutive_failures < 3:
        time.sleep(0.05)

    app._stop_event.set()
    t.join(timeout=2)

    assert app._consecutive_failures >= 3
    assert app._thread_error is not None
    assert "Persistent simulated failure" in app._thread_error


def test_consecutive_failure_triggers_ui_notice_after_threshold(
    monkeypatch, patch_cv_window, always_failing_pipeline_cls,
):
    """After _max_consecutive_failures, a UI notice should be set."""
    monkeypatch.setattr("main.DiffusionPipeline", always_failing_pipeline_cls)
    app = App(_make_cfg())
    app._max_consecutive_failures = 2  # lower threshold for faster test
    app.canvas.mask[20:30, 20:30] = 150
    app._gen_state = GenState.READY
    app._reset_ack.set()

    t = threading.Thread(target=app.async_diffusion, daemon=True)
    t.start()

    deadline = time.time() + 10
    while time.time() < deadline and app._consecutive_failures < 2:
        time.sleep(0.05)

    app._stop_event.set()
    t.join(timeout=2)

    assert app._consecutive_failures >= 2
    notice = app._ui_notice
    assert notice is not None
    assert "failing repeatedly" in notice


# -- FPS cycling -------------------------------------------------------------

def test_cycle_fps_wraps_around(app):
    initial_fps = app._display_fps
    opts = app.cfg.ui.display_fps_options
    # Cycle through all options to wrap back
    for _ in range(len(opts)):
        app._cycle_fps()
    assert app._display_fps == initial_fps


def test_cycle_fps_advances(app):
    opts = app.cfg.ui.display_fps_options
    app._cycle_fps()
    assert app._display_fps == opts[app._fps_index]


# -- UI notice expiry --------------------------------------------------------

def test_ui_notice_expires_after_duration(app):
    app._set_ui_notice("test notice", duration_s=0.1)
    assert app._current_ui_notice() == "test notice"
    time.sleep(0.15)
    assert app._current_ui_notice() is None


def test_ui_notice_clears_fields_on_expiry(app):
    app._set_ui_notice("test notice", duration_s=0.1)
    time.sleep(0.15)
    app._current_ui_notice()
    assert app._ui_notice is None
    assert app._ui_notice_until == 0.0


# -- save failure path -------------------------------------------------------

def test_save_failure_sets_notice(app, monkeypatch):
    monkeypatch.setattr("os.path.isfile", lambda _p: False)
    monkeypatch.setattr("cv2.imwrite", lambda _p, _img: False)

    app._save_image()
    notice = app._current_ui_notice()
    assert notice is not None
    assert "Save failed" in notice


def test_save_duplicate_filename_appends_suffix(app, monkeypatch):
    """When the timestamped file exists, a _1 suffix is appended."""
    call_count = 0

    def fake_isfile(path):
        nonlocal call_count
        call_count += 1
        # First call: file exists. Second call (with _1): doesn't exist.
        return call_count <= 1

    monkeypatch.setattr("os.path.isfile", fake_isfile)
    written_paths = []
    monkeypatch.setattr("cv2.imwrite", lambda p, _img: (written_paths.append(p), True)[-1])

    app._save_image()
    assert len(written_paths) == 1
    assert "_1.png" in written_paths[0]


# -- adjust max inference steps (increase) -----------------------------------

def test_adjust_max_inference_steps_increase(app):
    original_max = app._max_inference_steps
    app._adjust_max_inference_steps(2)
    assert app._max_inference_steps == original_max + 2


def test_adjust_max_inference_steps_caps_at_runtime_ceiling(app):
    """Runtime cap is max(cfg.max_inference_steps, 30)."""
    runtime_cap = max(app.cfg.inference.max_inference_steps, 30)
    app._adjust_max_inference_steps(999)
    assert app._max_inference_steps == runtime_cap


# -- trigger_exit unblocks events -------------------------------------------

def test_trigger_exit_sets_stop_event(app):
    app.trigger_exit()
    assert app.exit_triggered is True
    assert app._stop_event.is_set()


def test_trigger_exit_unblocks_reset_ack(app):
    app._reset_ack.clear()
    app.trigger_exit()
    assert app._reset_ack.is_set()


def test_trigger_exit_unblocks_gen_done(app):
    app._gen_done.clear()
    app.trigger_exit()
    assert app._gen_done.is_set()


# -- undo clears thread error -----------------------------------------------

def test_undo_clears_thread_error(app):
    """Undo (via _restore_snapshot) should clear sticky thread errors."""
    x, y = _canvas_point(app)
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    app.canvas.commit_active_to_mask()
    app.mouse_callback(cv2.EVENT_LBUTTONUP, x, y, 0, None)
    app._thread_error = "some error"
    app._undo_last_stroke()
    assert app._thread_error is None


# -- _DiffusionCancelled during generation -----------------------------------

def test_stop_event_during_pipeline_cancels_cleanly(
    monkeypatch, patch_cv_window, mock_pipeline_cls,
):
    """Setting _stop_event mid-pipeline should raise _DiffusionCancelled and exit."""
    from conftest import MockPipeline

    class StopDuringStepPipeline(MockPipeline):
        def run_inpaint(self_pipe, *args, **kwargs):
            step_callback = kwargs.get("step_callback")
            if step_callback:
                import torch
                # Set stop event before the first step callback
                app._stop_event.set()
                step_callback(0, 2, torch.rand(1, 3, 8, 8))
            from PIL import Image
            return Image.new("RGB", (64, 64), (128, 128, 128))

    monkeypatch.setattr("main.DiffusionPipeline", StopDuringStepPipeline)
    app = App(_make_cfg())
    app.canvas.mask[20:30, 20:30] = 150
    app._gen_state = GenState.READY
    app._reset_ack.set()

    t = threading.Thread(target=app.async_diffusion, daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive(), "Thread should exit after _DiffusionCancelled"
    # No error should be recorded — this is a clean cancellation
    assert app._thread_error is None
