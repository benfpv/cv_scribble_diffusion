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

import numpy as np
import time
import cv2
import threading
from collections import deque
from dataclasses import dataclass

from typing import Optional

from cv_scribble_diffusion.config import AppConfig
from cv_scribble_diffusion.generation.pipeline import DiffusionPipeline
from cv_scribble_diffusion.ui.canvas import Canvas, CanvasSnapshot
from cv_scribble_diffusion.ui.animator import Animator
from cv_scribble_diffusion.infra.debug import DebugWriter
from cv_scribble_diffusion.infra.runtime_logging import configure_logging, get_logger
from cv_scribble_diffusion.utils.colorspace import rgb_to_bgr
from cv_scribble_diffusion.ui.overlay import UIOverlay
from cv_scribble_diffusion.ui.windowing import create_app_window
from cv_scribble_diffusion.app.prompt_editor import PromptEditor
from cv_scribble_diffusion.app.state import GenState, GenerationState
from cv_scribble_diffusion.app.frame_composer import FrameComposer, FrameState
from cv_scribble_diffusion.app.input_controller import InputController
from cv_scribble_diffusion.generation.worker import GenerationWorker


logger = get_logger(__name__)


_EXIT_CONFIRM_SECONDS = 2.5
_CLOSING_NOTICE = "Closing safely..."


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
        self._frame_composer = FrameComposer(cfg, self.ui)

        # UI state
        self._state = GenerationState(
            inference_steps=cfg.inference.min_inference_steps)
        self._max_inference_steps = cfg.inference.max_inference_steps
        self.image_sizes_max_index = len(cfg.inference.image_sizes_ramp) - 1
        self.exit_triggered = False
        self._exit_confirm_stage = 0
        self._exit_confirm_until: float = 0.0
        self.mask_visibility_toggle = True
        self._ui_notice: Optional[str] = None
        self._ui_notice_until: float = 0.0
        self.prompt = PromptEditor(
            initial_prompt=cfg.inference.prompt,
            max_chars=cfg.ui.prompt_max_chars,
            cursor_index_fn=lambda text, x, cursor, max_chars: (
                self.ui.prompt_cursor_index(text, x, cursor, max_chars)
            ),
            notify=self._set_ui_notice,
            on_commit=self._on_prompt_committed,
        )

        # FPS pacing
        fps_opts = cfg.ui.display_fps_options
        self._fps_index = fps_opts.index(cfg.ui.display_fps_default) if cfg.ui.display_fps_default in fps_opts else 0
        self._display_fps = fps_opts[self._fps_index]

        # Generation tracking
        self._undo_stack: deque[AppSnapshot] = deque(maxlen=30)
        self._pending_restore: Optional[AppSnapshot] = None

        # Background diffusion worker owns the thread, the shutdown event, and
        # the display-loop handshake events.
        self.worker = GenerationWorker(
            cfg=cfg,
            canvas=self.canvas,
            animator=self.animator,
            pipeline=self.pipe,
            dbg=self.dbg,
            state=self._state,
            prompt_provider=self.prompt.committed,
            notify=self._set_ui_notice,
            max_steps_provider=lambda: self._max_inference_steps,
            image_sizes_max_index=self.image_sizes_max_index,
        )

        # OpenCV window
        self._input = InputController(self)
        borderless_applied = create_app_window(
            cfg.ui.window_name, cfg.ui.window_size, borderless=cfg.ui.borderless_window,
        )
        cv2.setMouseCallback(cfg.ui.window_name, self.mouse_callback)
        logger.info(
            "OpenCV window ready: name=%s size=%s borderless=%s applied=%s",
            cfg.ui.window_name, cfg.ui.window_size, cfg.ui.borderless_window, borderless_applied,
        )

    # -- diffusion worker delegation (preserves prior App surface) ------------

    @property
    def _stop_event(self) -> threading.Event:
        return self.worker._stop_event

    @property
    def _gen_done(self) -> threading.Event:
        return self.worker._gen_done

    @property
    def _reset_ack(self) -> threading.Event:
        return self.worker._reset_ack

    @property
    def _prev_gen_mask(self) -> np.ndarray:
        return self.worker.prev_gen_mask

    @_prev_gen_mask.setter
    def _prev_gen_mask(self, value: np.ndarray):
        self.worker.prev_gen_mask = value

    @property
    def _consecutive_failures(self) -> int:
        return self.worker._consecutive_failures

    @_consecutive_failures.setter
    def _consecutive_failures(self, value: int):
        self.worker._consecutive_failures = value

    @property
    def _max_consecutive_failures(self) -> int:
        return self.worker._max_consecutive_failures

    @_max_consecutive_failures.setter
    def _max_consecutive_failures(self, value: int):
        self.worker._max_consecutive_failures = value

    def async_diffusion(self):
        """Run the background diffusion loop (delegates to GenerationWorker)."""
        self.worker.run()

    # -- generation state (lock-guarded, delegated to GenerationState) --------

    @property
    def _gen_state(self) -> GenState:
        return self._state.gen_state

    @_gen_state.setter
    def _gen_state(self, value: GenState):
        self._state.gen_state = value

    @property
    def _inference_steps(self) -> int:
        return self._state.inference_steps

    @_inference_steps.setter
    def _inference_steps(self, value: int):
        self._state.inference_steps = value

    @property
    def current_inference_steps(self) -> int:
        return self._state.current_inference_steps

    @current_inference_steps.setter
    def current_inference_steps(self, value: int):
        self._state.current_inference_steps = value

    @property
    def image_size_index(self) -> int:
        return self._state.image_size_index

    @image_size_index.setter
    def image_size_index(self, value: int):
        self._state.image_size_index = value

    @property
    def _thread_error(self) -> Optional[str]:
        return self._state.thread_error

    @_thread_error.setter
    def _thread_error(self, value: Optional[str]):
        self._state.thread_error = value

    @property
    def _gen_count(self) -> int:
        return self._state.gen_count

    @_gen_count.setter
    def _gen_count(self, value: int):
        self._state.gen_count = value

    # -- mouse / keyboard -----------------------------------------------------

    def _handle_keypress(self, key_code: int):
        """Map keyboard input to toolbar-equivalent actions."""
        self._input.handle_keypress(key_code)

    def _on_prompt_committed(self, changed: bool, new_prompt: str):
        """Apply generation side-effects when the prompt changes on commit."""
        if not changed:
            return
        self._inference_steps = self.cfg.inference.min_inference_steps
        self.image_size_index = 0
        if np.any(self.canvas.mask):
            self._state.transition(GenState.IDLE, GenState.READY)
        self._announce("Prompt updated", source="Prompt")
        self._set_ui_notice(
            f"Prompt updated ({len(new_prompt)}/{self.cfg.ui.prompt_max_chars})")

    def mouse_callback(self, event, x, y, flags, param):
        """OpenCV mouse callback: toolbar hits and canvas strokes."""
        self._input.handle_mouse(event, x, y, flags, param)

    def reset_canvas(self):
        """Clear all stroke data, imagery, and animation state."""
        logger.warning("Resetting canvas state")
        self._undo_stack.clear()
        self._pending_restore = None
        self.canvas.reset()
        self.animator.reset()
        self._inference_steps = self.cfg.inference.min_inference_steps
        self._gen_count = 0
        self.worker.reset_tracking()
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
        if time.time() > self._exit_confirm_until:
            self._clear_exit_confirmation()
        exit_armed = self._exit_confirm_stage > 0
        gen_state = self._gen_state
        state = FrameState(
            is_generating=gen_state == GenState.GENERATING,
            is_resetting=gen_state == GenState.RESETTING,
            exit_active=exit_armed or self.exit_triggered,
            mask_visibility=self.mask_visibility_toggle,
            current_inference_steps=self.current_inference_steps,
            gen_count=self._gen_count,
            max_inference_steps=self._max_inference_steps,
            generation_progress=self._ui_generation_progress(),
            display_fps=self._display_fps,
            brush_thickness=self.canvas.brush_thickness,
            mask_active=self.canvas.mask_active,
            mask_present=self.canvas.mask_present,
            has_active_strokes=self.canvas.has_active_strokes,
            prompt_info=self.prompt.info(self.cfg.ui.show_prompt_box),
            ui_notice=self._current_ui_notice(),
            thread_error=self._thread_error,
            canvas_notice=canvas_notice,
        )
        return self._frame_composer.compose(display_frame, state)

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

    # -- display loop ---------------------------------------------------------

    def run(self):
        """Start the diffusion thread and enter the OpenCV display/event loop."""
        cfg = self.cfg
        ucfg = cfg.ui
        icfg = cfg.inference
        self.worker.start()

        while True:
            if self.canvas.commit_active_to_mask():
                self.image_size_index = 0
                self._inference_steps = icfg.min_inference_steps
                self._state.transition(GenState.IDLE, GenState.READY)

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
        self.worker.stop(join_timeout=2.0)
        logger.info("Application shutdown")
        cv2.destroyAllWindows()

