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
from PIL import Image
import threading

from typing import Optional

from config import AppConfig
from pipeline import DiffusionPipeline
from canvas import Canvas
from animator import Animator
from debug import DebugWriter
from runtime_logging import configure_logging, get_logger
from ui import UIOverlay, StatusInfo


logger = get_logger(__name__)


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
        self.num_inference_steps = -1
        self.current_inference_steps = 1
        self.image_size_index = 0
        self.image_sizes_max_index = len(cfg.inference.image_sizes_ramp) - 1
        self.exit_triggered = False
        self.mask_visibility_toggle = True
        self._is_generating = False

        # FPS pacing
        fps_opts = cfg.ui.display_fps_options
        self._fps_index = fps_opts.index(cfg.ui.display_fps_default) if cfg.ui.display_fps_default in fps_opts else 0
        self._display_fps = fps_opts[self._fps_index]

        # Generation tracking
        self._gen_count = 0
        self._gen_seq = 0
        self._prev_gen_mask = np.zeros(cfg.ui.image_size, dtype="uint8")

        # Synchronisation between diffusion thread and display loop.
        self._gen_done = threading.Event()
        self._reset_ack = threading.Event()
        self._reset_ack.set()
        self._reset_requested = False

        # OpenCV window
        cv2.namedWindow(cfg.ui.window_name)
        self._center_window()
        cv2.setMouseCallback(cfg.ui.window_name, self.mouse_callback)
        logger.info("OpenCV window ready: name=%s size=%s", cfg.ui.window_name, cfg.ui.window_size)

    def _center_window(self):
        """Center the OpenCV window on the primary screen."""
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

    def mouse_callback(self, event, x, y, flags, param):
        """OpenCV mouse callback: toolbar hits and canvas strokes."""
        cfg = self.cfg
        canvas = self.canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            action = self.ui.hit_test(x, y)
            if action == "exit":
                print("...[Toolbar] Exiting...")
                logger.info("Toolbar exit requested")
                self.trigger_exit()
                return
            elif action == "reset":
                if not self._reset_requested:
                    print("...[Toolbar] Reset Canvas...")
                    logger.info("Toolbar reset requested")
                    self.reset_canvas()
                return
            elif action == "mask":
                self.mask_visibility_toggle = not self.mask_visibility_toggle
                state = "On" if self.mask_visibility_toggle else "Off"
                print(f"...[Toolbar] Toggle Mask Visibility [{state}]...")
                logger.info("Mask visibility toggled: %s", state)
                return
            elif action == "fps":
                self._cycle_fps()
                return

            coords = self.ui.canvas_coords(x, y)
            if coords is not None:
                canvas.begin_stroke(*coords)
                self.num_inference_steps = cfg.inference.min_inference_steps
        elif event == cv2.EVENT_MOUSEMOVE:
            if canvas.drawing:
                coords = self.ui.canvas_coords(x, y)
                if coords is not None:
                    canvas.continue_stroke(*coords)
                    self.num_inference_steps = cfg.inference.min_inference_steps
        elif event == cv2.EVENT_LBUTTONUP:
            coords = self.ui.canvas_coords(x, y)
            if coords is not None:
                canvas.end_stroke(*coords)

    def reset_canvas(self):
        """Clear all stroke data, imagery, and animation state."""
        logger.warning("Resetting canvas state")
        self.canvas.reset()
        self.animator.reset()
        self.num_inference_steps = -1
        self._reset_requested = True
        self._gen_count = 0
        self._gen_seq = 0
        self._prev_gen_mask = np.zeros(self.cfg.ui.image_size, dtype="uint8")

    def trigger_exit(self):
        """Signal the application to exit."""
        self.exit_triggered = True
        logger.info("Exit triggered")

    def _cycle_fps(self):
        """Cycle through the available FPS presets."""
        opts = self.cfg.ui.display_fps_options
        self._fps_index = (self._fps_index + 1) % len(opts)
        self._display_fps = opts[self._fps_index]
        self.animator.set_interp_fps(self._display_fps)
        print(f"...[Toolbar] Display FPS: {self._display_fps}...")
        logger.info("Display FPS changed to %s", self._display_fps)

    # -- diffusion thread -----------------------------------------------------

    def async_diffusion(self):
        """Background thread: runs the pipeline whenever strokes are present."""
        cfg = self.cfg
        icfg = cfg.inference
        ucfg = cfg.ui
        while True:
            try:
                if self.num_inference_steps > 1:
                    self._gen_seq += 1
                    self._is_generating = True
                    mask_gray = self.canvas.mask.copy()
                    nonzero = cv2.findNonZero(mask_gray)
                    logger.info(
                        "Generation cycle #%s starting: steps=%s nonzero_mask=%s",
                        self._gen_seq,
                        self.num_inference_steps,
                        nonzero is not None,
                    )

                    # Decide crop region
                    use_crop = False
                    cx1 = cy1 = 0
                    cx2, cy2 = ucfg.image_size
                    if nonzero is not None:
                        bx, by, bw, bh = cv2.boundingRect(nonzero)
                        pad = icfg.crop_pad
                        rx1 = max(0, bx - pad)
                        ry1 = max(0, by - pad)
                        rx2 = min(ucfg.image_size[0], bx + bw + pad)
                        ry2 = min(ucfg.image_size[1], by + bh + pad)
                        rw = (rx2 - rx1) // 8 * 8
                        rh = (ry2 - ry1) // 8 * 8
                        rx2 = rx1 + rw
                        ry2 = ry1 + rh
                        if rw * rh < int(icfg.crop_area_threshold * ucfg.image_size[0] * ucfg.image_size[1]) and rw >= icfg.crop_min_dim and rh >= icfg.crop_min_dim:
                            use_crop = True
                            cx1, cy1, cx2, cy2 = rx1, ry1, rx2, ry2
                    logger.info("Crop decision: use_crop=%s crop=%s", use_crop, (cx1, cy1, cx2, cy2))

                    init_pil = self.canvas.image
                    control_np = np.where(mask_gray > 0, np.uint8(255), np.uint8(0))
                    control_pil = Image.fromarray(cv2.cvtColor(control_np, cv2.COLOR_GRAY2RGB))
                    inference_steps = self.num_inference_steps
                    self.current_inference_steps = inference_steps

                    # Dilate
                    dil = icfg.mask_dilate * 2 + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
                    sx, sy = ucfg.display_scale

                    def step_cb(s, total_steps, l):
                        if not self._reset_requested:
                            logger.debug("Pipeline step callback: step=%s/%s", s + 1, total_steps)
                            self.animator.on_step(s, self.current_inference_steps, l)
                        else:
                            logger.debug("Skipped step callback due to reset request")

                    gen_start_time = time.time()

                    if use_crop:
                        crop = (cx1, cy1, cx2, cy2)
                        cw, ch = cx2 - cx1, cy2 - cy1
                        stroke_crop = mask_gray[cy1:cy2, cx1:cx2]
                        dil_crop = cv2.dilate(stroke_crop, kernel)
                        inpaint_mask = Image.fromarray(np.where(dil_crop > 0, np.uint8(255), np.uint8(0)))
                        px1, py1 = int(cx1 * sx), int(cy1 * sy)
                        px2, py2 = int(cx2 * sx), int(cy2 * sy)
                        delta_crop = cv2.subtract(stroke_crop, self._prev_gen_mask[cy1:cy2, cx1:cx2])
                        delta_count = int(cv2.countNonZero(delta_crop))
                        logger.info("Generation #%s crop delta pixels=%s", self._gen_seq, delta_count)
                        if delta_count > 0:
                            delta_dil_crop = cv2.dilate(delta_crop, kernel)
                            dist_map = self.animator.make_dist_map(
                                delta_crop, delta_dil_crop, (px2 - px1, py2 - py1), cw // 2, ch // 2)
                        else:
                            # No newly-added strokes: refine with global crossfade.
                            dist_map = None
                            logger.info("No delta strokes in crop; using global crossfade reveal")
                        self.animator.prepare_generation(crop, dist_map)
                        self.dbg.save_annotated_crop(init_pil, crop,
                            f"steps={inference_steps} size={cw}x{ch}")
                        self.dbg.save("crop_init", init_pil.crop(crop))
                        self.dbg.save("crop_control", control_pil.crop(crop))
                        self.dbg.save("crop_mask", inpaint_mask)
                        result_crop = self.pipe.run_inpaint(
                            init_pil.crop(crop), inpaint_mask,
                            control_pil.crop(crop),
                            icfg.prompt, inference_steps, width=cw, height=ch,
                            step_callback=step_cb,
                        )
                        self.dbg.save("crop_result", result_crop)
                        self.canvas.patch_image(crop, result_crop, icfg.crop_pad // 2)
                    else:
                        image_size = icfg.image_sizes_ramp[self.image_size_index]
                        dilated_full = cv2.dilate(mask_gray, kernel)
                        inpaint_mask = Image.fromarray(cv2.resize(
                            np.where(dilated_full > 0, np.uint8(255), np.uint8(0)),
                            image_size, interpolation=cv2.INTER_NEAREST))
                        h_f, w_f = mask_gray.shape
                        delta_full = cv2.subtract(mask_gray, self._prev_gen_mask)
                        delta_count = int(cv2.countNonZero(delta_full))
                        logger.info("Generation #%s full delta pixels=%s", self._gen_seq, delta_count)
                        if delta_count > 0:
                            delta_dilated = cv2.dilate(delta_full, kernel)
                            dist_map = self.animator.make_dist_map(
                                delta_full, delta_dilated, ucfg.present_size, w_f // 2, h_f // 2)
                        else:
                            # No newly-added strokes: refine with global crossfade.
                            dist_map = None
                            logger.info("No delta strokes in full frame; using global crossfade reveal")
                        self.animator.prepare_generation(None, dist_map)
                        self.dbg.save("full_init", init_pil.resize(image_size))
                        self.dbg.save("full_control", control_pil.resize(image_size, Image.NEAREST))
                        self.dbg.save("full_mask", inpaint_mask)
                        result = self.pipe.run_inpaint(
                            init_pil.resize(image_size), inpaint_mask,
                            control_pil.resize(image_size, Image.NEAREST),
                            icfg.prompt, inference_steps,
                            width=image_size[0], height=image_size[1],
                            step_callback=step_cb,
                        )
                        self.dbg.save("full_result", result)
                        self.canvas.image = result.resize(ucfg.image_size)
                        self.image_size_index = min(self.image_size_index + 1, self.image_sizes_max_index)

                    final_frame = cv2.resize(
                        cv2.cvtColor(np.array(self.canvas.image), cv2.COLOR_RGB2BGR),
                        ucfg.present_size, interpolation=cv2.INTER_LINEAR)
                    generation_duration = time.time() - gen_start_time
                    logger.info("Generation inference complete in %.3fs; staging outro", generation_duration)
                    self.animator.start_outro(final_frame, generation_duration)
                    self.animator.wait_for_outro()
                    logger.info("Outro completed")
                    self._is_generating = False
                    self._gen_count += 1
                    self._prev_gen_mask = mask_gray.copy()

                    # Signal the display loop and wait for it to acknowledge
                    self._reset_ack.clear()
                    self._gen_done.set()
                    self._reset_ack.wait()
                    self._gen_done.clear()

                    self.num_inference_steps = min(
                        self.num_inference_steps + icfg.rate_inference_steps_change,
                        icfg.max_inference_steps)
                    time.sleep(0.1)
                else:
                    time.sleep(1)
            except Exception:
                logger.exception("Unhandled exception in diffusion thread")
                self._is_generating = False
                self._gen_done.clear()
                self._reset_ack.set()
                time.sleep(0.25)

    # -- display loop ---------------------------------------------------------

    def run(self):
        """Start the diffusion thread and enter the OpenCV display/event loop."""
        cfg = self.cfg
        ucfg = cfg.ui
        icfg = cfg.inference
        threading.Thread(target=self.async_diffusion, daemon=True).start()

        while True:
            if self.canvas.commit_active_to_mask():
                self.image_size_index = 0
                self.num_inference_steps = icfg.min_inference_steps
                logger.debug("Committed active strokes to persistent mask")

            if self._reset_requested:
                if self._gen_done.wait(timeout=0):
                    self._gen_done.clear()
                    self._reset_requested = False
                    self.animator.set_display_frame(
                        np.zeros((*ucfg.present_size, 3), dtype="uint8"))
                    self._reset_ack.set()
            elif self._gen_done.is_set():
                self._gen_done.clear()
                self._reset_ack.set()

            display_frame = self.animator.get_display_frame()

            button_states = {
                "exit": self.exit_triggered,
                "reset": self._reset_requested,
                "mask": self.mask_visibility_toggle,
                "fps": False,
            }
            button_labels = {
                "fps": str(self._display_fps),
            }
            status = StatusInfo(
                quality=self.current_inference_steps if (self._is_generating or self._gen_count > 0) else 0,
                quality_min=icfg.min_inference_steps,
                quality_max=icfg.max_inference_steps,
                gen_count=self._gen_count,
            )
            window_frame = self.ui.compose_frame(
                display_frame,
                self.canvas.mask_active,
                self.canvas.mask_present,
                self.mask_visibility_toggle,
                self.canvas.has_active_strokes,
                self.animator.generation_progress,
                self._is_generating,
                button_states,
                button_labels,
                status=status,
            )

            cv2.imshow(ucfg.window_name, window_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # Esc
                print("...[Esc] Exiting...")
                logger.info("Escape exit requested")
                self.trigger_exit()
            elif key == 9:  # Tab
                self.mask_visibility_toggle = not self.mask_visibility_toggle
                state = "On" if self.mask_visibility_toggle else "Off"
                print(f"...[Tab] Toggle Mask Visibility [{state}]...")
                logger.info("Mask visibility toggled from keyboard: %s", state)
            elif key == 13:  # Enter
                save_count = 1
                image_save_name = None
                for _ in range(ucfg.image_store_limit_count - save_count + 1):
                    save_count += 1
                    image_save_name = f"saved_image_{save_count}.png"
                    if not os.path.isfile(image_save_name):
                        break
                if save_count >= ucfg.image_store_limit_count + 1:
                    print(f"...[Enter] Saving failed - image count ({save_count}) exceeds limit ({ucfg.image_store_limit_count})...")
                else:
                    print(f"...[Enter] Saving {image_save_name}...")
                    logger.info("Saving image to %s", image_save_name)
                    img = cv2.resize(
                        cv2.cvtColor(np.array(self.canvas.image), cv2.COLOR_RGB2BGR),
                        ucfg.present_size, interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(image_save_name, img)
            elif key == 32:  # Space
                if not self._reset_requested:
                    print("...[SpaceBar] Reset Canvas...")
                    self.reset_canvas()
            elif key == 81:  # Left Arrow
                new = self.canvas.brush_thickness - 1
                self.canvas.set_brush_thickness(new)
                print(f"...[Left] [-] Brush Thickness to {self.canvas.brush_thickness}...")
            elif key == 83:  # Right Arrow
                new = self.canvas.brush_thickness + 1
                self.canvas.set_brush_thickness(new)
                print(f"...[Right] [+] Brush Thickness to {self.canvas.brush_thickness}...")

            if self.exit_triggered:
                break
            time.sleep(1.0 / self._display_fps)

        self.animator.stop()
        logger.info("Application shutdown")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("...[Startup] Launching cv_scribble_diffusion (loading models and UI)...")
    app = App()
    app.run()