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
from PIL import Image
import threading

from typing import Optional

from config import AppConfig
from pipeline import DiffusionPipeline
from canvas import Canvas
from animator import Animator


class App:
    """Thin wiring layer: connects Config, Canvas, Pipeline, and Animator."""

    def __init__(self, cfg: Optional[AppConfig] = None):
        self.cfg = cfg or AppConfig()
        cfg = self.cfg

        # Sub-systems
        self.pipe = DiffusionPipeline(cfg)
        self.canvas = Canvas(cfg)
        self.animator = Animator(cfg, self.pipe.taesd, self.pipe.taesd_device)

        # UI state
        self.inhibit_this_mouse_event = False
        self.present_size_half = (cfg.ui.present_size[0] // 2, cfg.ui.present_size[1] // 2)
        self.num_inference_steps = -1
        self.current_inference_steps = 1
        self.image_size_index = 0
        self.image_sizes_max_index = len(cfg.inference.image_sizes_ramp) - 1
        self.exit_triggered = False
        self.mask_visibility_toggle = True

        # Synchronisation between diffusion thread and display loop.
        # _gen_done  : set by diffusion thread when a generation finishes.
        # _reset_ack : set by display loop once the reset frame has been shown,
        #              telling the diffusion thread it can resume normally.
        self._gen_done = threading.Event()
        self._reset_ack = threading.Event()
        self._reset_ack.set()          # not in reset state initially
        self._reset_requested = False  # flag: display loop should clear the canvas next frame

        # Pen-control locations / colours
        h = self.present_size_half
        self.pen_controls_exit_loc_begin = [h[0] - 62, 2]
        self.pen_controls_exit_loc_end = [h[0] - 22, 12]
        self.pen_controls_exit_color = [100, 100, 100]
        self.pen_controls_exit_color_triggered = [180, 180, 180]
        self.pen_controls_reset_loc_begin = [h[0] - 20, 2]
        self.pen_controls_reset_loc_end = [h[0] + 20, 12]
        self.pen_controls_reset_color_static = [50, 50, 150]
        self.pen_controls_reset_color = [50, 50, 150]
        self.pen_controls_reset_color_triggered = [80, 80, 240]
        self.pen_controls_visibility_loc_begin = [h[0] + 22, 2]
        self.pen_controls_visibility_loc_end = [h[0] + 62, 12]
        self.pen_controls_visibility_color_on = [50, 200, 50]
        self.pen_controls_visibility_color = [50, 200, 50]
        self.pen_controls_visibility_color_off = [50, 120, 50]

        # OpenCV window
        cv2.namedWindow(cfg.ui.window_name)
        cv2.moveWindow(cfg.ui.window_name, 300, 0)
        cv2.setMouseCallback(cfg.ui.window_name, self.mouse_callback)

    # -- mouse / keyboard -----------------------------------------------------

    def mouse_callback(self, event, x, y, flags, param):
        """OpenCV mouse callback: pen-control clicks, stroke drawing, and stroke commit."""
        cfg = self.cfg
        canvas = self.canvas
        if event == cv2.EVENT_LBUTTONDOWN:
            if cfg.ui.pen_controls_active and y < 50:
                if self.pen_controls_exit_loc_begin[0] < x < self.pen_controls_exit_loc_end[0] and self.pen_controls_exit_loc_begin[1] < y < self.pen_controls_exit_loc_end[1]:
                    print("...[Pen Controls] Exiting...")
                    self.trigger_exit()
                    self.inhibit_this_mouse_event = True
                elif self.pen_controls_reset_loc_begin[0] < x < self.pen_controls_reset_loc_end[0] and self.pen_controls_reset_loc_begin[1] < y < self.pen_controls_reset_loc_end[1]:
                    if not self._reset_requested:
                        print("...[Pen Controls] Reset Canvas (changes take effect on next present)...")
                        self.reset_canvas()
                        self.pen_controls_reset_color = self.pen_controls_reset_color_triggered
                        self.inhibit_this_mouse_event = True
                elif self.pen_controls_visibility_loc_begin[0] < x < self.pen_controls_visibility_loc_end[0] and self.pen_controls_visibility_loc_begin[1] < y < self.pen_controls_visibility_loc_end[1]:
                    self.mask_visibility_toggle = not self.mask_visibility_toggle
                    state = "On" if self.mask_visibility_toggle else "Off"
                    print(f"...[Pen Controls] Toggle Mask Visibility [{state}]...")
                    self.pen_controls_visibility_color = (
                        self.pen_controls_visibility_color_on if self.mask_visibility_toggle
                        else self.pen_controls_visibility_color_off
                    )
                    self.inhibit_this_mouse_event = True
            if not self.inhibit_this_mouse_event:
                canvas.begin_stroke(x, y)
                self.num_inference_steps = cfg.inference.min_inference_steps
            else:
                self.inhibit_this_mouse_event = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if canvas.drawing:
                canvas.continue_stroke(x, y)
                self.num_inference_steps = cfg.inference.min_inference_steps
        elif event == cv2.EVENT_LBUTTONUP:
            canvas.end_stroke(x, y)

    def reset_canvas(self):
        """Clear all stroke data, imagery, and animation state."""
        self.canvas.reset()
        self.animator.reset()
        self.num_inference_steps = -1
        self._reset_requested = True

    def trigger_exit(self):
        """Signal the application to exit."""
        self.exit_triggered = True
        self.pen_controls_exit_color = self.pen_controls_exit_color_triggered
        frame = self.animator.get_display_frame()
        cv2.rectangle(frame, self.pen_controls_exit_loc_begin,
                      self.pen_controls_exit_loc_end, self.pen_controls_exit_color, -1)
        self.animator.set_display_frame(frame)
        cv2.imshow(self.cfg.ui.window_name, frame)

    # -- diffusion thread -----------------------------------------------------

    def async_diffusion(self):
        """Background thread: runs the pipeline whenever strokes are present."""
        cfg = self.cfg
        icfg = cfg.inference
        ucfg = cfg.ui
        while True:
            if self.num_inference_steps > 1:
                mask_gray = self.canvas.mask.copy()
                nonzero = cv2.findNonZero(mask_gray)

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

                init_pil = self.canvas.image
                control_np = np.where(mask_gray > 0, np.uint8(255), np.uint8(0))
                control_pil = Image.fromarray(cv2.cvtColor(control_np, cv2.COLOR_GRAY2RGB))
                inference_steps = self.num_inference_steps
                self.current_inference_steps = inference_steps

                # Dilate
                dil = icfg.mask_dilate * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
                sx, sy = ucfg.display_scale

                def step_cb(s, ts, l):
                    if not self._reset_requested:
                        self.animator.on_step(s, self.current_inference_steps, l)

                if use_crop:
                    crop = (cx1, cy1, cx2, cy2)
                    cw, ch = cx2 - cx1, cy2 - cy1
                    stroke_crop = mask_gray[cy1:cy2, cx1:cx2]
                    dil_crop = cv2.dilate(stroke_crop, kernel)
                    inpaint_mask = Image.fromarray(np.where(dil_crop > 0, np.uint8(255), np.uint8(0)))
                    px1, py1 = int(cx1 * sx), int(cy1 * sy)
                    px2, py2 = int(cx2 * sx), int(cy2 * sy)
                    dist_map = self.animator.make_dist_map(
                        stroke_crop, dil_crop, (px2 - px1, py2 - py1), cw // 2, ch // 2)
                    self.animator.prepare_generation(crop, dist_map)
                    result_crop = self.pipe.run_inpaint(
                        init_pil.crop(crop), inpaint_mask,
                        control_pil.crop(crop),
                        icfg.prompt, inference_steps, width=cw, height=ch,
                        step_callback=step_cb,
                    )
                    self.canvas.patch_image(crop, result_crop, icfg.crop_pad // 2)
                else:
                    image_size = icfg.image_sizes_ramp[self.image_size_index]
                    dilated_full = cv2.dilate(mask_gray, kernel)
                    inpaint_mask = Image.fromarray(cv2.resize(
                        np.where(dilated_full > 0, np.uint8(255), np.uint8(0)),
                        image_size, interpolation=cv2.INTER_NEAREST))
                    h_f, w_f = mask_gray.shape
                    dist_map = self.animator.make_dist_map(
                        mask_gray, dilated_full, ucfg.present_size, w_f // 2, h_f // 2)
                    self.animator.prepare_generation(None, dist_map)
                    result = self.pipe.run_inpaint(
                        init_pil.resize(image_size), inpaint_mask,
                        control_pil.resize(image_size, Image.NEAREST),
                        icfg.prompt, inference_steps,
                        width=image_size[0], height=image_size[1],
                        step_callback=step_cb,
                    )
                    self.canvas.image = result.resize(ucfg.image_size)
                    self.image_size_index = min(self.image_size_index + 1, self.image_sizes_max_index)

                final_frame = cv2.resize(
                    cv2.cvtColor(np.array(self.canvas.image), cv2.COLOR_RGB2BGR),
                    ucfg.present_size, interpolation=cv2.INTER_LINEAR)
                self.animator.start_outro(final_frame)

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

            if self._reset_requested:
                # Wait until the in-flight generation signals done, then ack
                if self._gen_done.wait(timeout=0):
                    self._gen_done.clear()
                    self._reset_requested = False
                    self.animator.set_display_frame(
                        np.zeros((*ucfg.present_size, 3), dtype="uint8"))
                    self.pen_controls_reset_color = self.pen_controls_reset_color_static
                    self._reset_ack.set()
            elif self._gen_done.is_set():
                # Normal (non-reset) generation finished — acknowledge immediately
                self._gen_done.clear()
                self._reset_ack.set()

            display_frame = self.animator.get_display_frame()
            if np.any(self.canvas.mask_active) and not self.mask_visibility_toggle:
                display_frame = cv2.add(display_frame, cv2.cvtColor(self.canvas.mask_active, cv2.COLOR_GRAY2BGR))
            if self.mask_visibility_toggle:
                display_frame = cv2.add(display_frame, self.canvas.mask_present)
                self.pen_controls_visibility_color = self.pen_controls_visibility_color_on

            if ucfg.pen_controls_active:
                cv2.rectangle(display_frame, self.pen_controls_exit_loc_begin, self.pen_controls_exit_loc_end, self.pen_controls_exit_color, -1)
                cv2.rectangle(display_frame, self.pen_controls_reset_loc_begin, self.pen_controls_reset_loc_end, self.pen_controls_reset_color, -1)
                cv2.rectangle(display_frame, self.pen_controls_visibility_loc_begin, self.pen_controls_visibility_loc_end, self.pen_controls_visibility_color, -1)

            cv2.imshow(ucfg.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # Esc
                print("...[Esc] Exiting...")
                self.trigger_exit()
            elif key == 9:  # Tab
                self.mask_visibility_toggle = not self.mask_visibility_toggle
                state = "On" if self.mask_visibility_toggle else "Off"
                print(f"...[Tab] Toggle Mask Visibility [{state}]...")
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
                    img = cv2.resize(
                        cv2.cvtColor(np.array(self.canvas.image), cv2.COLOR_RGB2BGR),
                        ucfg.present_size, interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(image_save_name, img)
            elif key == 32:  # Space
                if not self._reset_requested:
                    print("...[SpaceBar] Reset Canvas (changes take effect on next present)...")
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
            time.sleep(0.01)

        self.animator.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = App()
    app.run()