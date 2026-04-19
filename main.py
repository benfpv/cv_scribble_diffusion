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
import ctypes
import threading
import enum
from collections import deque
from dataclasses import dataclass

from typing import Optional

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
from ui import UIOverlay, StatusInfo


logger = get_logger(__name__)


class GenState(enum.Enum):
    """Explicit generation lifecycle states."""
    IDLE = "idle"              # No generation requested
    READY = "ready"            # Strokes present, should generate
    GENERATING = "generating"  # Pipeline currently running
    RESETTING = "resetting"    # Reset requested, awaiting thread drain


class _DiffusionCancelled(Exception):
    """Raised inside the pipeline step callback to abort cleanly on shutdown."""


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
        cv2.namedWindow(cfg.ui.window_name)
        self._center_window()
        cv2.setMouseCallback(cfg.ui.window_name, self.mouse_callback)
        logger.info("OpenCV window ready: name=%s size=%s", cfg.ui.window_name, cfg.ui.window_size)

    def _center_window(self):
        """Center the OpenCV window on the primary screen.

        Windows-only via ``ctypes.windll.user32``. On non-Windows platforms
        the screen-size lookup raises and we fall through to placement at
        ``(0, 0)``, which most window managers will then reposition.
        """
        ww, wh = self.cfg.ui.window_size
        sw = ww
        sh = wh
        try:
            user32 = ctypes.windll.user32
            sw = int(user32.GetSystemMetrics(0))
            sh = int(user32.GetSystemMetrics(1))
        except Exception:
            pass

        x = max(0, (sw - ww) // 2)
        y = max(0, (sh - wh) // 2)
        cv2.moveWindow(self.cfg.ui.window_name, x, y)
        logger.info("Centered window at (%s, %s) on screen %sx%s", x, y, sw, sh)

    # -- mouse / keyboard -----------------------------------------------------

    def _handle_keypress(self, key_code: int):
        """Map keyboard input to toolbar-equivalent actions."""
        if key_code < 0:
            return

        # cv2.waitKeyEx returns a platform-specific code. The low byte covers
        # ASCII/control keys across backends.
        low_byte = key_code & 0xFF

        if low_byte == 27:  # Esc
            self.trigger_exit()
            return

        if low_byte == 32:  # Space
            if self._gen_state != GenState.RESETTING:
                self._announce("Reset Canvas", source="Keyboard")
                self.reset_canvas()
            return
        if low_byte in (13, 10):  # Enter
            self._save_image()
            return
        if low_byte == 9:  # Tab
            self.mask_visibility_toggle = not self.mask_visibility_toggle
            state = "On" if self.mask_visibility_toggle else "Off"
            self._announce(f"Toggle Mask Visibility [{state}]", source="Keyboard")
            return

        # Undo: support Ctrl+Z (26), plain z/Z, and u/U as fallback.
        if low_byte in (26, ord("z"), ord("Z"), ord("u"), ord("U")):
            self._undo_last_stroke()
            return

        # Arrow keys can vary by backend; support common forms.
        if key_code in (2424832, 81):  # Left
            self._adjust_brush_thickness(-1)
            return
        if key_code in (2555904, 83):  # Right
            self._adjust_brush_thickness(1)
            return

    def mouse_callback(self, event, x, y, flags, param):
        """OpenCV mouse callback: toolbar hits and canvas strokes."""
        cfg = self.cfg
        canvas = self.canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            action = self.ui.hit_test(x, y)
            if action == "exit":
                now = time.time()
                if now > self._exit_confirm_until:
                    self._exit_confirm_stage = 0

                if self._exit_confirm_stage >= 2:
                    self._announce("Exiting", source="Toolbar")
                    self._exit_confirm_stage = 0
                    self._exit_confirm_until = 0.0
                    self.trigger_exit()
                else:
                    self._exit_confirm_stage += 1
                    self._exit_confirm_until = now + 2.5
                    remaining = 3 - self._exit_confirm_stage
                    if remaining == 1:
                        notice = "Click EXIT once more to confirm"
                    else:
                        notice = f"Click EXIT {remaining} more times to confirm"
                    self._announce(notice, source="Toolbar")
                    self._set_ui_notice(notice, duration_s=2.5)
                return

            self._exit_confirm_stage = 0
            self._exit_confirm_until = 0.0
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
            if canvas.drawing:
                coords = self.ui.canvas_coords(x, y)
                if coords is not None:
                    canvas.continue_stroke(*coords)
                    self._inference_steps = cfg.inference.min_inference_steps
                    if self._gen_state == GenState.IDLE:
                        self._gen_state = GenState.READY
                else:
                    # If the drag leaves the canvas, commit what we've drawn and terminate.
                    canvas.commit_active_to_mask()
                    canvas.end_stroke(canvas.prev_x, canvas.prev_y)
        elif event == cv2.EVENT_LBUTTONUP:
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
        # Unblock any thread waiting on these events so it can observe stop.
        self._reset_ack.set()
        self._gen_done.set()
        logger.info("Exit triggered")

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

    def _adjust_max_inference_steps(self, delta: int):
        """Adjust the runtime max number of diffusion steps."""
        icfg = self.cfg.inference
        runtime_cap = max(icfg.max_inference_steps, 30)
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

                gen_start_time = time.time()
                result = self.pipe.run_inpaint(
                    inpaint.init_image, inpaint.inpaint_mask, inpaint.control_image,
                    icfg.prompt, inference_steps,
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

            is_generating = self._gen_state == GenState.GENERATING
            if time.time() > self._exit_confirm_until:
                self._exit_confirm_stage = 0
            exit_armed = self._exit_confirm_stage > 0
            button_states = {
                "exit": exit_armed,
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
                "exit": f"EXIT {self._exit_confirm_stage}/3" if exit_armed else "EXIT",
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
            window_frame = self.ui.compose_frame(
                display_frame,
                self.canvas.mask_active,
                self.canvas.mask_present,
                self.mask_visibility_toggle,
                self.canvas.has_active_strokes,
                self.animator.generation_progress,
                is_generating,
                button_states,
                button_labels,
                status=status,
            )

            cv2.imshow(ucfg.window_name, window_frame)
            key_code = cv2.waitKeyEx(1)
            self._handle_keypress(key_code)

            if self.exit_triggered:
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
    print("...[Startup] Launching cv_scribble_diffusion (loading models and UI)...")
    try:
        app = App()
        app.run()
    except Exception:
        logger.exception("Fatal unhandled exception in main thread")
        raise