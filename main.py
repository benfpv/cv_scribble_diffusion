"""cv_scribble_diffusion — real-time scribble-to-image OpenCV application.

Draw on the OpenCV window with your mouse; a ControlNet-guided Stable Diffusion
inpainting pipeline continuously generates imagery around your strokes and reveals
it with an animated wavefront that expands outward from the drawn lines.

Model setup
-----------
SD v1.5 and the scribble ControlNet weights are NOT included in this repository.
Place the model directories in the project root before running (see README.md).
TAESD (madebyollin/taesd) is downloaded automatically on first run via HuggingFace.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import numpy as np
import time
import cv2
import threading
import enum
from collections import deque
from dataclasses import dataclass

from typing import Optional, Tuple

from config import AppConfig
from pipeline import DiffusionPipeline
from canvas import Canvas, CanvasSnapshot
from animator import Animator
from debug import DebugWriter
from generation import (
    decide_crop, make_dilation_kernel, make_control_image,
    compute_dist_map_inputs, build_inpaint_inputs,
)
from runtime_logging import configure_logging, get_logger
from colorspace import rgb_to_bgr
from ui import UIOverlay, StatusInfo, PromptInfo
from windowing import create_app_window


logger = get_logger(__name__)


class GenState(enum.Enum):
    """Explicit generation lifecycle states."""
    IDLE = "idle"              # No generation requested
    READY = "ready"            # Strokes present, should generate
    GENERATING = "generating"  # Pipeline currently running
    RESETTING = "resetting"    # Reset requested, awaiting thread drain


class _DiffusionCancelled(Exception):
    """Raised inside the pipeline step callback to abort cleanly on shutdown."""


_EXIT_CONFIRM_SECONDS = 2.5
_CLOSING_NOTICE = "Closing safely..."


class _StartupProgress:
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


@dataclass
class AppSnapshot:
    """Restorable app state used for undoing the last stroke."""

    canvas: CanvasSnapshot
    prev_gen_mask: np.ndarray
    image_size_index: int
    inference_steps: int


class App:
    """Thin wiring layer: connects Config, Canvas, Pipeline, Animator, and UIOverlay."""

    def __init__(self, cfg: Optional[AppConfig] = None):
        self.cfg = cfg or AppConfig()
        cfg = self.cfg
        configure_logging(cfg.logging)
        logger.info("Initializing App")

        # Sub-systems
        self.pipe = DiffusionPipeline(cfg)
        self.canvas = Canvas(cfg)
        self.animator = Animator(cfg, self.pipe.taesd, self.pipe.taesd_device)
        self.dbg = DebugWriter(cfg.debug)
        self.ui = UIOverlay(cfg.ui)

        # UI state
        self._gen_state = GenState.IDLE
        self._inference_steps = cfg.inference.min_inference_steps
        self._max_inference_steps = cfg.inference.max_inference_steps
        self.current_inference_steps = 1
        self.image_size_index = 0
        self.image_sizes_max_index = len(cfg.inference.image_sizes_ramp) - 1
        self.exit_triggered = False
        self._exit_confirm_stage = 0
        self._exit_confirm_until: float = 0.0
        self.mask_visibility_toggle = True
        self._thread_error: Optional[str] = None
        self._ui_notice: Optional[str] = None
        self._ui_notice_until: float = 0.0
        self._prompt_lock = threading.Lock()
        self._prompt_text = self._clean_prompt_text(cfg.inference.prompt)
        self._prompt_draft = self._prompt_text
        self._prompt_cursor = len(self._prompt_draft)
        self._prompt_editing = False
        self._prompt_selection_anchor: Optional[int] = None
        self._prompt_dragging_selection = False

        # FPS pacing
        fps_opts = cfg.ui.display_fps_options
        self._fps_index = fps_opts.index(cfg.ui.display_fps_default) if cfg.ui.display_fps_default in fps_opts else 0
        self._display_fps = fps_opts[self._fps_index]

        # Generation tracking
        self._gen_count = 0
        self._gen_seq = 0
        self._prev_gen_mask = np.zeros(cfg.ui.image_size, dtype="uint8")
        self._undo_stack: deque[AppSnapshot] = deque(maxlen=30)
        self._pending_restore: Optional[AppSnapshot] = None

        # Cooperative shutdown for the diffusion thread.
        self._stop_event = threading.Event()
        self._diffusion_thread: Optional[threading.Thread] = None

        # Consecutive-failure tracking for exponential backoff in async_diffusion.
        # NOTE: Not reset by reset_canvas() — the pipeline's instability is
        # independent of canvas state. Only a successful generation clears it.
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5

        # Synchronisation between diffusion thread and display loop.
        self._gen_done = threading.Event()
        self._reset_ack = threading.Event()
        self._reset_ack.set()

        # OpenCV window
        borderless_applied = create_app_window(
            cfg.ui.window_name, cfg.ui.window_size, borderless=cfg.ui.borderless_window,
        )
        cv2.setMouseCallback(cfg.ui.window_name, self.mouse_callback)
        logger.info(
            "OpenCV window ready: name=%s size=%s borderless=%s applied=%s",
            cfg.ui.window_name, cfg.ui.window_size, cfg.ui.borderless_window, borderless_applied,
        )

    # -- mouse / keyboard -----------------------------------------------------

    def _handle_keypress(self, key_code: int):
        """Map keyboard input to toolbar-equivalent actions."""
        if key_code < 0:
            return

        # cv2.waitKeyEx returns a platform-specific code. The low byte covers
        # ASCII/control keys across backends.
        low_byte = key_code & 0xFF

        if self._prompt_editing and self._handle_prompt_keypress(key_code, low_byte):
            return

        if low_byte == 27:  # Esc
            self._request_exit(source="Keyboard")
            return

        if low_byte == 32:  # Space
            self._clear_exit_confirmation()
            if self._gen_state != GenState.RESETTING:
                self._announce("Reset Canvas", source="Keyboard")
                self.reset_canvas()
            return
        if low_byte in (13, 10):  # Enter
            self._clear_exit_confirmation()
            self._save_image()
            return
        if low_byte == 9:  # Tab
            self._clear_exit_confirmation()
            self.mask_visibility_toggle = not self.mask_visibility_toggle
            state = "On" if self.mask_visibility_toggle else "Off"
            self._announce(f"Toggle Mask Visibility [{state}]", source="Keyboard")
            return

        # Undo: support Ctrl+Z (26), plain z/Z, and u/U as fallback.
        if low_byte in (26, ord("z"), ord("Z"), ord("u"), ord("U")):
            self._clear_exit_confirmation()
            self._undo_last_stroke()
            return

        # Arrow keys can vary by backend; support common forms.
        if key_code in (2424832, 81):  # Left
            self._clear_exit_confirmation()
            self._adjust_brush_thickness(-1)
            return
        if key_code in (2555904, 83):  # Right
            self._clear_exit_confirmation()
            self._adjust_brush_thickness(1)
            return

    def _handle_prompt_keypress(self, key_code: int, low_byte: int) -> bool:
        """Handle text editing keys while the prompt field is focused."""
        if low_byte == 1:  # Ctrl+A
            self._select_prompt_all()
            return True
        if low_byte == 27:  # Esc
            self._cancel_prompt_edit()
            return True
        if low_byte in (13, 10):  # Enter
            self._commit_prompt_edit()
            return True
        if low_byte == 8:  # Backspace
            if self._delete_prompt_selection():
                return True
            if self._prompt_cursor > 0:
                self._prompt_draft = (
                    self._prompt_draft[:self._prompt_cursor - 1] +
                    self._prompt_draft[self._prompt_cursor:]
                )
                self._prompt_cursor -= 1
            return True
        if low_byte == 127 or key_code == 3014656:  # Delete
            if self._delete_prompt_selection():
                return True
            if self._prompt_cursor < len(self._prompt_draft):
                self._prompt_draft = (
                    self._prompt_draft[:self._prompt_cursor] +
                    self._prompt_draft[self._prompt_cursor + 1:]
                )
            return True
        if key_code == 2424832:  # Left arrow
            selection = self._prompt_selection_bounds()
            if selection is not None:
                self._set_prompt_cursor(selection[0])
            else:
                self._set_prompt_cursor(self._prompt_cursor - 1)
            return True
        if key_code == 2555904:  # Right arrow
            selection = self._prompt_selection_bounds()
            if selection is not None:
                self._set_prompt_cursor(selection[1])
            else:
                self._set_prompt_cursor(self._prompt_cursor + 1)
            return True
        if key_code == 2359296:  # Home
            self._set_prompt_cursor(0)
            return True
        if key_code == 2293760:  # End
            self._set_prompt_cursor(len(self._prompt_draft))
            return True
        if 32 <= low_byte <= 126:
            self._insert_prompt_char(chr(low_byte))
            return True
        return True

    def _insert_prompt_char(self, ch: str):
        """Insert a printable prompt character at the current cursor."""
        self._insert_prompt_text(ch)

    def _insert_prompt_text(self, text: str):
        """Insert printable prompt text, replacing the active selection if any."""
        text = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
        if not text:
            return
        max_chars = self.cfg.ui.prompt_max_chars
        selection = self._prompt_selection_bounds()
        start, end = selection if selection is not None else (self._prompt_cursor, self._prompt_cursor)
        available = max_chars - (len(self._prompt_draft) - (end - start))
        if available <= 0:
            self._set_ui_notice(f"Prompt limit reached ({max_chars} chars)")
            return
        insert_text = text[:available]
        self._prompt_draft = (
            self._prompt_draft[:start] + insert_text +
            self._prompt_draft[end:]
        )
        self._prompt_cursor = start + len(insert_text)
        self._clear_prompt_selection()
        if len(insert_text) < len(text):
            self._set_ui_notice(f"Prompt limit reached ({max_chars} chars)")

    def _begin_prompt_edit(self, cursor: Optional[int] = None):
        """Focus the prompt editor and place the cursor."""
        if not self._prompt_editing:
            with self._prompt_lock:
                self._prompt_draft = self._prompt_text
        self._set_prompt_cursor(len(self._prompt_draft) if cursor is None else cursor)
        self._prompt_dragging_selection = False
        self._prompt_editing = True
        self._set_ui_notice("Editing prompt: Enter applies, Esc cancels", duration_s=4.0)

    def _commit_prompt_edit(self):
        """Commit the prompt draft so future generations use it."""
        new_prompt = self._clean_prompt_text(self._prompt_draft)
        with self._prompt_lock:
            old_prompt = self._prompt_text
            self._prompt_text = new_prompt
        self._prompt_draft = new_prompt
        self._prompt_cursor = len(new_prompt)
        self._prompt_editing = False
        self._clear_prompt_selection()
        self._prompt_dragging_selection = False

        if new_prompt != old_prompt:
            self._inference_steps = self.cfg.inference.min_inference_steps
            self.image_size_index = 0
            if np.any(self.canvas.mask) and self._gen_state == GenState.IDLE:
                self._gen_state = GenState.READY
            self._announce("Prompt updated", source="Prompt")
            self._set_ui_notice(f"Prompt updated ({len(new_prompt)}/{self.cfg.ui.prompt_max_chars})")

    def _cancel_prompt_edit(self):
        """Discard prompt edits and return to the committed prompt."""
        with self._prompt_lock:
            self._prompt_draft = self._prompt_text
        self._prompt_cursor = len(self._prompt_draft)
        self._prompt_editing = False
        self._clear_prompt_selection()
        self._prompt_dragging_selection = False
        self._set_ui_notice("Prompt edit cancelled")

    def _clean_prompt_text(self, text: str) -> str:
        """Keep prompt text single-line, printable ASCII, and within the configured limit."""
        text = text.replace("\r", " ").replace("\n", " ")
        cleaned = "".join(ch for ch in text if 32 <= ord(ch) <= 126)
        return cleaned[:self.cfg.ui.prompt_max_chars]

    def _clamp_prompt_cursor(self, cursor: int) -> int:
        """Clamp a cursor index to the current draft bounds."""
        return max(0, min(cursor, len(self._prompt_draft)))

    def _set_prompt_cursor(self, cursor: int, selecting: bool = False):
        """Move the prompt cursor, optionally extending selection from an anchor."""
        cursor = self._clamp_prompt_cursor(cursor)
        if selecting:
            if self._prompt_selection_anchor is None:
                self._prompt_selection_anchor = self._prompt_cursor
        else:
            self._clear_prompt_selection()
        self._prompt_cursor = cursor

    def _prompt_selection_bounds(self) -> Optional[Tuple[int, int]]:
        """Return active prompt selection as (start, end), or None."""
        if self._prompt_selection_anchor is None:
            return None
        anchor = self._clamp_prompt_cursor(self._prompt_selection_anchor)
        cursor = self._clamp_prompt_cursor(self._prompt_cursor)
        if anchor == cursor:
            return None
        return (min(anchor, cursor), max(anchor, cursor))

    def _clear_prompt_selection(self):
        """Clear any active prompt selection."""
        self._prompt_selection_anchor = None

    def _delete_prompt_selection(self) -> bool:
        """Delete the selected prompt text if a selection is active."""
        selection = self._prompt_selection_bounds()
        if selection is None:
            return False
        start, end = selection
        self._prompt_draft = self._prompt_draft[:start] + self._prompt_draft[end:]
        self._prompt_cursor = start
        self._clear_prompt_selection()
        return True

    def _select_prompt_all(self):
        """Select the complete prompt draft for replacement or deletion."""
        self._prompt_cursor = len(self._prompt_draft)
        self._prompt_selection_anchor = 0 if self._prompt_draft else None

    def _prompt_cursor_from_x(self, x: int) -> int:
        """Translate a window x coordinate to a cursor index for the prompt draft."""
        return self.ui.prompt_cursor_index(
            self._prompt_draft, x, self._prompt_cursor, self.cfg.ui.prompt_max_chars,
        )

    def _update_prompt_selection_from_x(self, x: int):
        """Extend the prompt selection to the cursor location under *x*."""
        self._set_prompt_cursor(self._prompt_cursor_from_x(x), selecting=True)

    def _finish_prompt_selection_drag(self, x: int):
        """End mouse-based prompt selection, clearing empty selections."""
        self._update_prompt_selection_from_x(x)
        self._prompt_dragging_selection = False
        if self._prompt_selection_bounds() is None:
            self._clear_prompt_selection()

    def _generation_prompt(self) -> str:
        """Return the committed prompt for a generation cycle."""
        with self._prompt_lock:
            return self._prompt_text

    def _prompt_info(self) -> Optional[PromptInfo]:
        """Build prompt editor state for UI rendering."""
        if not self.cfg.ui.show_prompt_box:
            return None
        if self._prompt_editing:
            text = self._prompt_draft
            cursor = self._prompt_cursor
        else:
            with self._prompt_lock:
                text = self._prompt_text
            cursor = len(text)
        cursor_visible = self._prompt_editing and int(time.time() * 2) % 2 == 0
        selection = self._prompt_selection_bounds() if self._prompt_editing else None
        return PromptInfo(
            text=text,
            editing=self._prompt_editing,
            cursor=cursor,
            cursor_visible=cursor_visible,
            max_chars=self.cfg.ui.prompt_max_chars,
            selection_start=selection[0] if selection is not None else 0,
            selection_end=selection[1] if selection is not None else 0,
        )

    def mouse_callback(self, event, x, y, flags, param):
        """OpenCV mouse callback: toolbar hits and canvas strokes."""
        cfg = self.cfg
        canvas = self.canvas
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if self.ui.prompt_hit_test(x, y):
                self._clear_exit_confirmation()
                self._begin_prompt_edit(self._prompt_cursor_from_x(x) if self._prompt_editing else None)
                self._select_prompt_all()
                self._prompt_dragging_selection = False
                return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.ui.prompt_hit_test(x, y):
                self._clear_exit_confirmation()
                if self._prompt_editing:
                    text = self._prompt_draft
                    cursor = self._prompt_cursor
                else:
                    with self._prompt_lock:
                        text = self._prompt_text
                    cursor = len(text)
                cursor = self.ui.prompt_cursor_index(text, x, cursor, cfg.ui.prompt_max_chars)
                self._begin_prompt_edit(cursor)
                self._prompt_selection_anchor = self._prompt_cursor
                self._prompt_dragging_selection = True
                return

            if self._prompt_editing:
                self._commit_prompt_edit()

            action = self.ui.hit_test(x, y)
            if action == "exit":
                self._request_exit(source="Toolbar")
                return

            self._clear_exit_confirmation()
            if action == "reset":
                if self._gen_state != GenState.RESETTING:
                    self._announce("Reset Canvas", source="Toolbar")
                    self.reset_canvas()
                return
            elif action == "mask":
                self.mask_visibility_toggle = not self.mask_visibility_toggle
                state = "On" if self.mask_visibility_toggle else "Off"
                self._announce(f"Toggle Mask Visibility [{state}]", source="Toolbar")
                return
            elif action == "save":
                self._save_image()
                return
            elif action == "undo":
                self._undo_last_stroke()
                return
            elif action == "brush_dec":
                self._adjust_brush_thickness(-1)
                return
            elif action == "brush_inc":
                self._adjust_brush_thickness(1)
                return
            elif action == "steps_dec":
                self._adjust_max_inference_steps(-2)
                return
            elif action == "steps_inc":
                self._adjust_max_inference_steps(2)
                return
            elif action == "fps":
                self._cycle_fps()
                return

            coords = self.ui.canvas_coords(x, y)
            if coords is not None:
                if not canvas.drawing:
                    self._undo_stack.append(self._snapshot_state())
                canvas.begin_stroke(*coords)
                self._inference_steps = cfg.inference.min_inference_steps
                if self._gen_state == GenState.IDLE:
                    self._gen_state = GenState.READY
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._prompt_dragging_selection:
                self._update_prompt_selection_from_x(x)
                return
            if canvas.drawing:
                coords = self.ui.canvas_coords(x, y)
                if coords is not None:
                    canvas.continue_stroke(*coords)
                    self._inference_steps = cfg.inference.min_inference_steps
                    if self._gen_state == GenState.IDLE:
                        self._gen_state = GenState.READY
                else:
                    # If the drag leaves the canvas, preserve the stroke and terminate it.
                    canvas.end_stroke(canvas.prev_x, canvas.prev_y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self._prompt_dragging_selection:
                self._finish_prompt_selection_drag(x)
                return
            coords = self.ui.canvas_coords(x, y)
            if coords is not None:
                canvas.end_stroke(*coords)
            elif canvas.drawing:
                # Mouse-up outside canvas should still finalize drawing state.
                canvas.end_stroke(canvas.prev_x, canvas.prev_y)

    def reset_canvas(self):
        """Clear all stroke data, imagery, and animation state."""
        logger.warning("Resetting canvas state")
        self._undo_stack.clear()
        self._pending_restore = None
        self.canvas.reset()
        self.animator.reset()
        self._inference_steps = self.cfg.inference.min_inference_steps
        self._gen_count = 0
        self._gen_seq = 0
        self._prev_gen_mask = np.zeros(self.cfg.ui.image_size, dtype="uint8")
        self._thread_error = None
        if self._gen_state == GenState.GENERATING:
            self._gen_state = GenState.RESETTING
        else:
            self._gen_state = GenState.IDLE
            self.animator.set_display_frame(
                np.zeros((*self.cfg.ui.present_size, 3), dtype="uint8"))

    def trigger_exit(self):
        """Signal the application to exit."""
        self.exit_triggered = True
        self._stop_event.set()
        self._set_ui_notice(_CLOSING_NOTICE, duration_s=10.0)
        # Unblock any thread waiting on these events so it can observe stop.
        self._reset_ack.set()
        self._gen_done.set()
        logger.info("Exit triggered")

    def _clear_exit_confirmation(self):
        """Cancel any pending two-step exit confirmation."""
        self._exit_confirm_stage = 0
        self._exit_confirm_until = 0.0

    def _request_exit(self, source: str):
        """Arm or confirm exit through a shared two-step UX."""
        now = time.time()
        if now > self._exit_confirm_until:
            self._clear_exit_confirmation()

        if self._exit_confirm_stage >= 1:
            self._announce(_CLOSING_NOTICE, source=source)
            self._clear_exit_confirmation()
            self.trigger_exit()
            return

        self._exit_confirm_stage = 1
        self._exit_confirm_until = now + _EXIT_CONFIRM_SECONDS
        notice = "Press Esc or click EXIT again to quit"
        self._announce(notice, source=source)
        self._set_ui_notice(notice, duration_s=_EXIT_CONFIRM_SECONDS)

    def _cycle_fps(self):
        """Cycle through the available FPS presets."""
        opts = self.cfg.ui.display_fps_options
        self._fps_index = (self._fps_index + 1) % len(opts)
        self._display_fps = opts[self._fps_index]
        self.animator.set_interp_fps(self._display_fps)
        self._announce(f"Display FPS: {self._display_fps}", source="Toolbar")

    def _save_image(self):
        """Save the current generated image to the working directory."""
        ucfg = self.cfg.ui


        # Timestamped filename avoids fragile counter math and the missing _1 case.
        image_save_name = time.strftime("saved_image_%Y%m%d_%H%M%S.png")
        # If the user spam-saves within the same second, append a short suffix.
        if os.path.isfile(image_save_name):
            stem, ext = os.path.splitext(image_save_name)
            n = 1
            while os.path.isfile(f"{stem}_{n}{ext}"):
                n += 1
            image_save_name = f"{stem}_{n}{ext}"

        self._announce(f"Saving {image_save_name}", source="Toolbar")
        saved_path = os.path.abspath(image_save_name)
        img = cv2.resize(
            rgb_to_bgr(np.array(self.canvas.image)),
            ucfg.present_size, interpolation=cv2.INTER_NEAREST)
        ok = cv2.imwrite(image_save_name, img)
        if ok:
            self._set_ui_notice(f"Saved snapshot: {saved_path}")
        else:
            self._set_ui_notice("Save failed: could not write file")

    def _adjust_brush_thickness(self, delta: int):
        """Increase or decrease brush thickness from toolbar controls."""
        self.canvas.set_brush_thickness(self.canvas.brush_thickness + delta)
        self._announce(
            f"Brush Thickness: {self.canvas.brush_thickness}", source="Toolbar",
        )

    def _announce(self, message: str, source: str = "User"):
        """Single sink for user-facing actions: prints to stdout and logs at INFO.

        Keeps console output and structured logs in one place so the message
        can't drift between them and so future routing (e.g., status bar) is
        a one-line change.
        """
        line = f"...[{source}] {message}..."
        print(line)
        logger.info("[%s] %s", source, message)

    def _set_ui_notice(self, message: str, duration_s: float = 3.0):
        """Display a short-lived status message in the bottom bar."""
        self._ui_notice = message
        self._ui_notice_until = time.time() + max(duration_s, 0.1)

    def _current_ui_notice(self) -> Optional[str]:
        """Return active status message if still within its visibility window."""
        if self._ui_notice and time.time() <= self._ui_notice_until:
            return self._ui_notice
        self._ui_notice = None
        self._ui_notice_until = 0.0
        return None

    def _ui_generation_progress(self) -> float:
        """Progress value for the UI progress bar.

        Keep the no-flicker behavior between back-to-back READY/GENERATING
        cycles, but force a clear idle bar when no generation is requested.
        """
        progress = self.animator.generation_progress
        if self._gen_state == GenState.IDLE:
            return 0.0
        return max(0.0, min(progress, 1.0))

    def _compose_window_frame(self, display_frame: np.ndarray,
                              canvas_notice: Optional[str] = None) -> np.ndarray:
        """Build a complete UI frame for normal display or shutdown notice."""
        cfg = self.cfg
        icfg = cfg.inference

        is_generating = self._gen_state == GenState.GENERATING
        if time.time() > self._exit_confirm_until:
            self._clear_exit_confirmation()
        exit_armed = self._exit_confirm_stage > 0
        exit_active = exit_armed or self.exit_triggered
        button_states = {
            "exit": exit_active,
            "reset": self._gen_state == GenState.RESETTING,
            "save": False,
            "mask": self.mask_visibility_toggle,
            "undo": False,
            "brush_dec": False,
            "brush_inc": False,
            "steps_dec": False,
            "steps_inc": False,
            "fps": False,
        }
        button_labels = {
            "exit": "QUIT?" if exit_active else "EXIT",
            "undo": "UNDO",
            "steps_dec": "MAX-",
            "steps_inc": "MAX+",
            "fps": "FPS",
            "brush_dec": "THIN",
            "brush_inc": "THICK",
        }
        status = StatusInfo(
            quality=self.current_inference_steps if (is_generating or self._gen_count > 0) else 0,
            quality_min=icfg.min_inference_steps,
            quality_max=self._max_inference_steps,
            gen_count=self._gen_count,
            display_fps=self._display_fps,
            brush_thickness=self.canvas.brush_thickness,
            ui_notice=self._current_ui_notice(),
            thread_error=self._thread_error,
        )
        return self.ui.compose_frame(
            display_frame,
            self.canvas.mask_active,
            self.canvas.mask_present,
            self.mask_visibility_toggle,
            self.canvas.has_active_strokes,
            self._ui_generation_progress(),
            button_states,
            button_labels,
            status=status,
            prompt=self._prompt_info(),
            canvas_notice=canvas_notice,
        )

    def _show_closing_frame(self, display_frame: np.ndarray):
        """Publish one final visible frame before shutdown cleanup can block."""
        frame = self._compose_window_frame(display_frame, canvas_notice=_CLOSING_NOTICE)
        cv2.imshow(self.cfg.ui.window_name, frame)
        cv2.waitKeyEx(1)

    def _adjust_max_inference_steps(self, delta: int):
        """Adjust the runtime max number of diffusion steps."""
        icfg = self.cfg.inference
        runtime_cap = icfg.runtime_step_cap
        new_max = max(icfg.min_inference_steps, min(self._max_inference_steps + delta, runtime_cap))
        if new_max == self._max_inference_steps:
            return
        self._max_inference_steps = new_max
        self._inference_steps = min(self._inference_steps, self._max_inference_steps)
        self.current_inference_steps = min(self.current_inference_steps, self._max_inference_steps)
        self._announce(
            f"Max Diffusion Steps: {self._max_inference_steps}", source="Toolbar",
        )

    def _snapshot_state(self) -> AppSnapshot:
        """Capture restorable application state before a user stroke."""
        return AppSnapshot(
            canvas=self.canvas.snapshot(),
            prev_gen_mask=self._prev_gen_mask.copy(),
            image_size_index=self.image_size_index,
            inference_steps=self._inference_steps,
        )

    def _restore_snapshot(self, snapshot: AppSnapshot):
        """Restore a previous application state for undo."""
        self.canvas.restore(snapshot.canvas)
        self._prev_gen_mask = snapshot.prev_gen_mask.copy()
        self.image_size_index = snapshot.image_size_index
        self._inference_steps = snapshot.inference_steps
        self._thread_error = None
        self.animator.reset()
        display = cv2.resize(
            rgb_to_bgr(np.array(self.canvas.image)),
            self.cfg.ui.present_size,
            interpolation=cv2.INTER_LINEAR,
        )
        self.animator.set_display_frame(display)
        self._gen_state = GenState.READY if np.any(self.canvas.mask) else GenState.IDLE

    def _undo_last_stroke(self):
        """Undo the most recent stroke by restoring the prior app snapshot."""
        if not self._undo_stack:
            self._announce("Undo unavailable", source="Toolbar")
            return

        snapshot = self._undo_stack.pop()
        if self._gen_state == GenState.GENERATING:
            self._pending_restore = snapshot
            self._gen_state = GenState.RESETTING
            return

        self._restore_snapshot(snapshot)
        self._announce("Undo last stroke", source="Toolbar")

    # -- diffusion thread -----------------------------------------------------

    def async_diffusion(self):
        """Background thread: runs the pipeline whenever strokes are present."""
        cfg = self.cfg
        icfg = cfg.inference
        ucfg = cfg.ui
        while not self._stop_event.is_set():
            try:
                if self._gen_state != GenState.READY:
                    time.sleep(1)
                    continue

                self._gen_state = GenState.GENERATING
                # Note: _thread_error is intentionally NOT cleared here. It is
                # only cleared on reset/undo so transient pipeline failures
                # remain visible to the user across subsequent successful runs.
                self._gen_seq += 1
                mask_gray = self.canvas.mask.copy()

                plan = decide_crop(
                    mask_gray, ucfg.image_size,
                    icfg.crop_pad, icfg.crop_alignment,
                    icfg.crop_area_threshold, icfg.crop_min_dim,
                )
                if plan is None:
                    logger.debug("Generation skipped: empty mask")
                    self._gen_state = GenState.IDLE
                    time.sleep(0.05)
                    continue
                logger.info(
                    "Generation cycle #%s starting: steps=%s use_crop=%s region=%s",
                    self._gen_seq, self._inference_steps, plan.use_crop, plan.region,
                )

                init_pil = self.canvas.image
                control_pil = make_control_image(mask_gray)
                inference_steps = self._inference_steps
                self.current_inference_steps = inference_steps
                kernel = make_dilation_kernel(icfg.mask_dilate)

                def step_cb(s, total_steps, l):
                    if self._stop_event.is_set():
                        # Best-effort cancellation: raising here aborts the diffusers loop.
                        raise _DiffusionCancelled()
                    if self._gen_state != GenState.RESETTING:
                        logger.debug("Pipeline step callback: step=%s/%s", s + 1, total_steps)
                        self.animator.on_step(s, self.current_inference_steps, l)
                    else:
                        logger.debug("Skipped step callback due to reset request")

                # Build distance map and inpaint inputs from helpers.
                dist_inputs = compute_dist_map_inputs(
                    mask_gray, self._prev_gen_mask, plan, ucfg, kernel,
                )
                if dist_inputs is None:
                    dist_map = None
                    logger.info("No delta strokes; using global crossfade reveal")
                else:
                    dist_map = self.animator.make_dist_map(
                        dist_inputs.delta_mask, dist_inputs.delta_dilated,
                        dist_inputs.out_size, dist_inputs.cx, dist_inputs.cy,
                    )

                ramp_size = (
                    None if plan.use_crop
                    else icfg.image_sizes_ramp[self.image_size_index]
                )
                inpaint = build_inpaint_inputs(
                    init_pil, mask_gray, control_pil, plan, kernel,
                    image_sizes_ramp_size=ramp_size,
                )

                self.animator.prepare_generation(
                    plan.region if plan.use_crop else None, dist_map,
                )

                # Debug artifacts mirror the previous per-branch tags.
                if plan.use_crop:
                    cw, ch = inpaint.width, inpaint.height
                    self.dbg.save_annotated_crop(init_pil, plan.region,
                        f"steps={inference_steps} size={cw}x{ch}")
                    self.dbg.save("crop_init", inpaint.init_image)
                    self.dbg.save("crop_control", inpaint.control_image)
                    self.dbg.save("crop_mask", inpaint.inpaint_mask)
                else:
                    self.dbg.save("full_init", inpaint.init_image)
                    self.dbg.save("full_control", inpaint.control_image)
                    self.dbg.save("full_mask", inpaint.inpaint_mask)

                prompt = self._generation_prompt()
                gen_start_time = time.time()
                result = self.pipe.run_inpaint(
                    inpaint.init_image, inpaint.inpaint_mask, inpaint.control_image,
                    prompt, inference_steps,
                    width=inpaint.width, height=inpaint.height,
                    step_callback=step_cb,
                )

                if self._gen_state == GenState.RESETTING:
                    logger.info("Reset requested during generation; skipping commit")
                    self._reset_ack.clear()
                    self._gen_done.set()
                    self._reset_ack.wait()
                    self._gen_done.clear()
                    time.sleep(0.05)
                    continue

                if plan.use_crop:
                    self.dbg.save("crop_result", result)
                    self.canvas.patch_image(plan.region, result, icfg.crop_feather_px)
                else:
                    self.dbg.save("full_result", result)
                    self.canvas.image = result.resize(ucfg.image_size)
                    self.image_size_index = min(
                        self.image_size_index + 1, self.image_sizes_max_index,
                    )

                final_frame = cv2.resize(
                    rgb_to_bgr(np.array(self.canvas.image)),
                    ucfg.present_size, interpolation=cv2.INTER_LINEAR)
                generation_duration = time.time() - gen_start_time
                logger.info("Generation inference complete in %.3fs; staging outro", generation_duration)
                self.animator.start_outro(final_frame, generation_duration)
                self.animator.wait_for_outro()
                logger.info("Outro completed")
                self._gen_state = GenState.READY
                self._gen_count += 1
                self._prev_gen_mask = mask_gray.copy()

                # Signal the display loop and wait for it to acknowledge
                self._reset_ack.clear()
                self._gen_done.set()
                self._reset_ack.wait()
                self._gen_done.clear()

                self._inference_steps = min(
                    self._inference_steps + icfg.rate_inference_steps_change,
                    self._max_inference_steps)
                self._consecutive_failures = 0
                time.sleep(0.1)
            except _DiffusionCancelled:
                logger.info("Diffusion cancelled by stop event")
                break
            except Exception as exc:
                logger.exception("Unhandled exception in diffusion thread")
                self._consecutive_failures += 1
                self._thread_error = str(exc)
                if self._consecutive_failures >= self._max_consecutive_failures:
                    self._set_ui_notice(
                        f"Pipeline failing repeatedly ({self._consecutive_failures}x); see logs",
                        duration_s=10.0,
                    )
                if self._gen_state == GenState.GENERATING:
                    self._gen_state = GenState.READY
                if self._gen_state == GenState.RESETTING:
                    self._gen_done.set()
                else:
                    self._gen_done.clear()
                self._reset_ack.set()
                # Exponential backoff capped at 4s.
                backoff = min(0.25 * (2 ** (self._consecutive_failures - 1)), 4.0)
                self._stop_event.wait(timeout=backoff)

    # -- display loop ---------------------------------------------------------

    def run(self):
        """Start the diffusion thread and enter the OpenCV display/event loop."""
        cfg = self.cfg
        ucfg = cfg.ui
        icfg = cfg.inference
        self._diffusion_thread = threading.Thread(
            target=self.async_diffusion, name="diffusion", daemon=True,
        )
        self._diffusion_thread.start()

        while True:
            if self.canvas.commit_active_to_mask():
                self.image_size_index = 0
                self._inference_steps = icfg.min_inference_steps
                if self._gen_state == GenState.IDLE:
                    self._gen_state = GenState.READY

            if self._gen_state == GenState.RESETTING:
                if self._gen_done.wait(timeout=0):
                    self._gen_done.clear()
                    if self._pending_restore is not None:
                        snapshot = self._pending_restore
                        self._pending_restore = None
                        self._restore_snapshot(snapshot)
                    else:
                        self._gen_state = GenState.IDLE
                        self.animator.set_display_frame(
                            np.zeros((*ucfg.present_size, 3), dtype="uint8"))
                    self._reset_ack.set()
            elif self._gen_done.is_set():
                self._gen_done.clear()
                self._reset_ack.set()

            display_frame = self.animator.get_display_frame()
            window_frame = self._compose_window_frame(display_frame)

            cv2.imshow(ucfg.window_name, window_frame)
            key_code = cv2.waitKeyEx(1)
            self._handle_keypress(key_code)

            if self.exit_triggered:
                self._show_closing_frame(display_frame)
                break
            time.sleep(1.0 / self._display_fps)

        self._stop_event.set()
        self.animator.stop()
        if self._diffusion_thread is not None:
            logger.info("Waiting for diffusion thread to stop")
            self._diffusion_thread.join(timeout=2.0)
            if self._diffusion_thread.is_alive():
                logger.warning("Diffusion thread did not stop within timeout")
        logger.info("Application shutdown")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("...[Startup] Launching cv_scribble_diffusion (loading models and UI)...", flush=True)
    try:
        with _StartupProgress("[Startup] Loading models and UI"):
            app = App()
        app.run()
    except Exception:
        logger.exception("Fatal unhandled exception in main thread")
        raise