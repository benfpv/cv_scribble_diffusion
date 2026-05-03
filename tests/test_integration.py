"""Integration tests for App lifecycle, GenState transitions, and thread error handling.

These tests use the mock pipelines from conftest.py to run without GPU or display.
"""

import numpy as np
import threading
import time
import pytest

import cv2

from cv_scribble_diffusion.config import AppConfig, UIConfig, InferenceConfig, RevealConfig
from cv_scribble_diffusion.app.app import App, GenState, _CLOSING_NOTICE


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
    monkeypatch.setattr("cv_scribble_diffusion.app.app.DiffusionPipeline", mock_pipeline_cls)
    return App(_make_cfg())


@pytest.fixture
def failing_app(monkeypatch, patch_cv_window, failing_pipeline_cls):
    """Create an App whose pipeline fails on first generation."""
    monkeypatch.setattr("cv_scribble_diffusion.app.app.DiffusionPipeline", failing_pipeline_cls)
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


def _prompt_point(app, x=12, y=12):
    x1, y1, _x2, _y2 = app.ui.prompt_box_rect
    return x1 + x, y1 + y


def _replace_prompt(app, text):
    app._begin_prompt_edit(0)
    app._prompt_draft = ""
    app._prompt_cursor = 0
    for ch in text:
        app._handle_keypress(ord(ch))
    app._handle_keypress(13)


# -- GenState transitions ----------------------------------------------------

def test_app_creates_window_with_configured_borderless_flag(monkeypatch, patch_cv_window, mock_pipeline_cls):
    captured = {}
    cfg = _make_cfg()
    cfg.ui.borderless_window = False
    monkeypatch.setattr("cv_scribble_diffusion.app.app.DiffusionPipeline", mock_pipeline_cls)

    def fake_create_app_window(window_name, window_size, borderless=True):
        captured["window_name"] = window_name
        captured["window_size"] = window_size
        captured["borderless"] = borderless
        return False

    monkeypatch.setattr("cv_scribble_diffusion.app.app.create_app_window", fake_create_app_window)
    App(cfg)

    assert captured == {
        "window_name": cfg.ui.window_name,
        "window_size": cfg.ui.window_size,
        "borderless": False,
    }


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


def test_click_without_mousemove_draws_brush_sized_point(app):
    app.canvas.set_brush_thickness(10)
    x, y = _canvas_point(app, 24, 24)

    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    app.mouse_callback(cv2.EVENT_LBUTTONUP, x, y, 0, None)

    ys, xs = np.nonzero(app.canvas.mask)
    assert xs.size > 1
    width = xs.max() - xs.min() + 1
    height = ys.max() - ys.min() + 1
    assert width >= app.canvas.brush_stroke_thickness
    assert height >= app.canvas.brush_stroke_thickness


def test_mouseup_commits_final_segment_without_waiting_for_display_loop(app):
    app.canvas.set_brush_thickness(8)
    x1, y1 = _canvas_point(app, 10, 10)
    x2, y2 = _canvas_point(app, 40, 10)

    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x1, y1, 0, None)
    app.canvas.commit_active_to_mask()
    mask_after_initial_point = app.canvas.mask.copy()
    app.mouse_callback(cv2.EVENT_MOUSEMOVE, x2, y2, 0, None)
    app.mouse_callback(cv2.EVENT_LBUTTONUP, x2, y2, 0, None)

    assert np.count_nonzero(app.canvas.mask) > np.count_nonzero(mask_after_initial_point)
    assert app.canvas.drawing is False
    assert app.canvas.has_active_strokes is False
    assert not np.any(app.canvas.mask_active)


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
    assert app._current_ui_notice() == _CLOSING_NOTICE


def test_escape_requires_confirmation(app):
    app._handle_keypress(27)
    assert app.exit_triggered is False
    assert app._exit_confirm_stage == 1

    app._handle_keypress(27)
    assert app.exit_triggered is True
    assert app._current_ui_notice() == _CLOSING_NOTICE


def test_show_closing_frame_renders_shutdown_notice(app, monkeypatch):
    captured = {}

    def fake_compose(display_frame, canvas_notice=None):
        captured["display_shape"] = display_frame.shape
        captured["canvas_notice"] = canvas_notice
        width, height = app.cfg.ui.window_size
        return np.zeros((height, width, 3), dtype=np.uint8)

    def fake_imshow(window_name, frame):
        captured["window_name"] = window_name
        captured["frame_shape"] = frame.shape

    monkeypatch.setattr(app, "_compose_window_frame", fake_compose)
    monkeypatch.setattr(cv2, "imshow", fake_imshow)
    monkeypatch.setattr(cv2, "waitKeyEx", lambda _delay: -1)

    display_frame = np.zeros((*app.cfg.ui.present_size, 3), dtype=np.uint8)
    app._show_closing_frame(display_frame)

    assert captured["canvas_notice"] == _CLOSING_NOTICE
    assert captured["display_shape"] == display_frame.shape
    assert captured["window_name"] == app.cfg.ui.window_name
    assert captured["frame_shape"] == (app.cfg.ui.window_size[1], app.cfg.ui.window_size[0], 3)


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


# -- prompt editing ----------------------------------------------------------

def test_prompt_box_click_focuses_editor(app):
    x, y = _prompt_point(app)
    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
    assert app._prompt_editing is True


def test_prompt_enter_commits_keyboard_text(app):
    _replace_prompt(app, "neon city")
    assert app._generation_prompt() == "neon city"
    assert app._prompt_editing is False


def test_prompt_escape_cancels_without_arming_exit(app):
    original_prompt = app._generation_prompt()
    app._begin_prompt_edit(0)
    app._insert_prompt_char("x")

    app._handle_keypress(27)

    assert app._generation_prompt() == original_prompt
    assert app._prompt_editing is False
    assert app.exit_triggered is False
    assert app._exit_confirm_stage == 0


def test_prompt_input_is_capped_at_configured_max(app):
    max_chars = app.cfg.ui.prompt_max_chars
    app._begin_prompt_edit(0)
    app._prompt_draft = ""
    app._prompt_cursor = 0

    for _ in range(max_chars + 5):
        app._handle_keypress(ord("x"))

    assert len(app._prompt_draft) == max_chars
    assert app._prompt_cursor == max_chars


def test_prompt_editing_supports_cursor_backspace_and_delete(app):
    _replace_prompt(app, "abcd")
    app._begin_prompt_edit(2)

    app._handle_keypress(8)  # Backspace removes b
    assert app._prompt_draft == "acd"
    assert app._prompt_cursor == 1

    app._handle_keypress(2555904)  # Right arrow to c|d
    app._handle_keypress(3014656)  # Delete removes d
    assert app._prompt_draft == "ac"
    assert app._prompt_cursor == 2


def test_prompt_cleaning_removes_newlines_and_non_ascii(app):
    dirty = "line one\nline two caf\u00e9"
    cleaned = app._clean_prompt_text(dirty)
    assert cleaned == "line one line two caf"


def test_prompt_home_end_keys_move_cursor(app):
    _replace_prompt(app, "abcdef")
    app._begin_prompt_edit(3)

    app._handle_keypress(2359296)  # Home
    assert app._prompt_cursor == 0

    app._handle_keypress(2293760)  # End
    assert app._prompt_cursor == len(app._prompt_draft)


def test_prompt_ctrl_a_selects_all_and_typing_replaces(app):
    _replace_prompt(app, "neon city")
    app._begin_prompt_edit(0)

    app._handle_keypress(1)  # Ctrl+A
    assert app._prompt_selection_bounds() == (0, len("neon city"))

    app._handle_keypress(ord("x"))
    assert app._prompt_draft == "x"
    assert app._prompt_cursor == 1
    assert app._prompt_selection_bounds() is None


def test_prompt_selected_text_can_be_deleted_with_backspace_or_delete(app):
    _replace_prompt(app, "abcdef")
    app._begin_prompt_edit(0)
    app._prompt_cursor = 4
    app._prompt_selection_anchor = 1

    app._handle_keypress(8)
    assert app._prompt_draft == "aef"
    assert app._prompt_cursor == 1
    assert app._prompt_selection_bounds() is None

    app._prompt_cursor = 3
    app._prompt_selection_anchor = 1
    app._handle_keypress(3014656)  # Delete
    assert app._prompt_draft == "a"
    assert app._prompt_cursor == 1
    assert app._prompt_selection_bounds() is None


def test_prompt_arrow_keys_collapse_selection(app):
    _replace_prompt(app, "abcdef")
    app._begin_prompt_edit(0)
    app._prompt_cursor = 5
    app._prompt_selection_anchor = 2

    app._handle_keypress(2424832)  # Left arrow
    assert app._prompt_cursor == 2
    assert app._prompt_selection_bounds() is None

    app._prompt_cursor = 1
    app._prompt_selection_anchor = 4
    app._handle_keypress(2555904)  # Right arrow
    assert app._prompt_cursor == 4
    assert app._prompt_selection_bounds() is None


def test_prompt_double_click_selects_all(app):
    _replace_prompt(app, "blue lake")
    x, y = _prompt_point(app)

    app.mouse_callback(cv2.EVENT_LBUTTONDBLCLK, x, y, 0, None)

    assert app._prompt_editing is True
    assert app._prompt_selection_bounds() == (0, len("blue lake"))


def test_prompt_mouse_drag_selects_and_replaces_text(app, monkeypatch):
    _replace_prompt(app, "crystal forest")
    monkeypatch.setattr(
        app.ui, "prompt_cursor_index",
        lambda text, x, cursor, max_chars: 0 if x < 50 else len(text),
    )
    x1, y1, _x2, _y2 = app.ui.prompt_box_rect
    y = y1 + 12

    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x1 + 1, y, 0, None)
    app.mouse_callback(cv2.EVENT_MOUSEMOVE, 100, y, 0, None)
    app.mouse_callback(cv2.EVENT_LBUTTONUP, 100, y, 0, None)

    assert app._prompt_selection_bounds() == (0, len("crystal forest"))

    for ch in "mist":
        app._handle_keypress(ord(ch))
    assert app._prompt_draft == "mist"
    assert app._prompt_selection_bounds() is None


def test_prompt_replacing_selection_can_edit_at_character_limit(app):
    max_chars = app.cfg.ui.prompt_max_chars
    app._begin_prompt_edit(0)
    app._prompt_draft = "x" * max_chars
    app._prompt_cursor = max_chars
    app._prompt_selection_anchor = max_chars - 1

    app._handle_keypress(ord("y"))

    assert len(app._prompt_draft) == max_chars
    assert app._prompt_draft.endswith("y")
    assert app._prompt_cursor == max_chars
    assert app._prompt_selection_bounds() is None


def test_prompt_clicking_canvas_commits_and_starts_drawing(app):
    app._begin_prompt_edit(0)
    app._prompt_draft = "blue lake"
    app._prompt_cursor = len(app._prompt_draft)
    x, y = _canvas_point(app)

    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)

    assert app._generation_prompt() == "blue lake"
    assert app._prompt_editing is False
    assert app.canvas.drawing is True


def test_prompt_selection_does_not_block_next_canvas_stroke(app):
    _replace_prompt(app, "blue lake")
    app._begin_prompt_edit(0)
    app._select_prompt_all()
    x, y = _canvas_point(app)

    app.mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)

    assert app._generation_prompt() == "blue lake"
    assert app._prompt_editing is False
    assert app._prompt_selection_bounds() is None
    assert app.canvas.drawing is True


def test_prompt_commit_marks_existing_mask_ready(app):
    app.canvas.mask[5:10, 5:10] = 255
    app._gen_state = GenState.IDLE
    app._inference_steps = 10

    _replace_prompt(app, "ink sketch")

    assert app._gen_state == GenState.READY
    assert app._inference_steps == app.cfg.inference.min_inference_steps


def test_generation_uses_committed_runtime_prompt(app):
    _replace_prompt(app, "crystal forest")
    app.canvas.mask[20:30, 20:30] = 150
    app._gen_state = GenState.READY
    app._reset_ack.set()

    _run_one_generation(app)

    assert app.pipe.prompts[-1] == "crystal forest"


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
    monkeypatch.setattr("cv_scribble_diffusion.app.app.DiffusionPipeline", slow_pipeline_cls)
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
    monkeypatch.setattr("cv_scribble_diffusion.app.app.DiffusionPipeline", always_failing_pipeline_cls)
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
    monkeypatch.setattr("cv_scribble_diffusion.app.app.DiffusionPipeline", always_failing_pipeline_cls)
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
    """Runtime cap comes from the explicit config ceiling."""
    runtime_cap = app.cfg.inference.runtime_step_cap
    app._adjust_max_inference_steps(999)
    assert app._max_inference_steps == runtime_cap


def test_adjust_brush_thickness_caps_at_config_max(app):
    app.canvas.set_brush_thickness(app.cfg.ui.max_brush_thickness)
    app._adjust_brush_thickness(1)
    assert app.canvas.brush_thickness == app.cfg.ui.max_brush_thickness


def test_adjust_brush_thickness_caps_at_config_min(app):
    app.canvas.set_brush_thickness(app.cfg.ui.min_brush_thickness)
    app._adjust_brush_thickness(-1)
    assert app.canvas.brush_thickness == app.cfg.ui.min_brush_thickness


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

    monkeypatch.setattr("cv_scribble_diffusion.app.app.DiffusionPipeline", StopDuringStepPipeline)
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
