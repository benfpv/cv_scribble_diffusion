"""Reveal-wavefront animation and threaded compositing.

Noise generation and reveal math live in reveal.py as pure functions.
"""

import time
import threading

import numpy as np
import cv2
import torch

from config import AppConfig
from reveal import build_dist_map, compute_reveal


class Animator:
    """Owns the display frame, threading primitives, and all compositing logic."""

    def __init__(self, cfg: AppConfig, taesd, taesd_device):
        self.cfg = cfg
        self._taesd = taesd
        self._taesd_device = taesd_device

        # Threading
        self._frame_lock = threading.Lock()
        self._interp_stop = threading.Event()

        ps = cfg.ui.present_size

        # Display state
        self.image_present = np.zeros((*ps, 3), dtype="uint8")
        self.prev_image_present = np.zeros((*ps, 3), dtype="uint8")

        # Per-generation state
        self._crop_region = None          # (x1, y1, x2, y2) in image_size coords
        self._dist_map_norm = None        # float32 at present-size
        self._last_display_alpha = 0.0
        self._prev_latents = None         # CPU float32 numpy
        self._last_interp_f32 = None      # last EMA frame
        self._step_wall_time = 1.0
        self._step_wall_ts = 0.0

    # -- public helpers -------------------------------------------------------

    def get_display_frame(self) -> np.ndarray:
        with self._frame_lock:
            return self.image_present.copy()

    def set_display_frame(self, frame: np.ndarray):
        with self._frame_lock:
            self.image_present = frame

    def reset(self):
        """Reset all animation state for a fresh generation cycle."""
        ps = self.cfg.ui.present_size
        with self._frame_lock:
            self.image_present = np.zeros((*ps, 3), dtype="uint8")
        self.prev_image_present = np.zeros((*ps, 3), dtype="uint8")
        self._crop_region = None
        self._dist_map_norm = None
        self._interp_stop.set()
        self._prev_latents = None
        self._last_interp_f32 = None
        self._last_display_alpha = 0.0
        self._step_wall_ts = 0.0

    def stop(self):
        """Signal all running threads to stop."""
        self._interp_stop.set()

    # -- distance map ---------------------------------------------------------

    def make_dist_map(self, stroke_mask: np.ndarray, dilated_mask: np.ndarray,
                       out_size: tuple, cx: int, cy: int) -> np.ndarray:
        """Delegate distance-map construction to reveal.py."""
        rcfg = self.cfg.reveal
        return build_dist_map(
            stroke_mask, dilated_mask, out_size, cx, cy,
            rcfg.reveal_mode, rcfg.stochastic_noise_strength,
        )

    # -- step callback --------------------------------------------------------

    def on_step(self, step: int, total_steps: int, latents_tensor):
        """Called by the pipeline step callback. Hands latents to a new interp thread."""
        now = time.time()
        if self._step_wall_ts > 0:
            self._step_wall_time = now - self._step_wall_ts
        self._step_wall_ts = now

        self._interp_stop.set()

        alpha = (step + 1) / max(total_steps, 1)
        edge = self.cfg.reveal.reveal_edge
        curr_latents_np = latents_tensor.float().cpu().numpy()
        prev_latents_np = self._prev_latents if self._prev_latents is not None else curr_latents_np

        if self.cfg.reveal.interp_fps > 0:
            self._interp_stop = threading.Event()
            snap_dist = self._dist_map_norm.copy() if self._dist_map_norm is not None else None
            snap_crop = self._crop_region
            snap_prev = self.prev_image_present.copy()
            snap_alpha_s = self._last_display_alpha
            snap_alpha_e = alpha
            snap_duration = max(self._step_wall_time, 0.05)
            threading.Thread(
                target=self._interp_thread_latent_lerp,
                args=(self._interp_stop, prev_latents_np, curr_latents_np,
                      snap_dist, snap_crop, snap_prev,
                      snap_alpha_s, snap_alpha_e, edge, snap_duration),
                daemon=True,
            ).start()
        self._prev_latents = curr_latents_np

    # -- outro launcher -------------------------------------------------------

    def start_outro(self, final_frame: np.ndarray):
        """Launch the post-generation outro reveal thread."""
        rcfg = self.cfg.reveal
        # Capture the pre-generation background BEFORE overwriting — the outro
        # composites new pixels expanding into this older image.
        pre_gen_prev = self.prev_image_present.copy()
        self.prev_image_present = final_frame.copy()

        if rcfg.reveal_outro_alpha <= 0 or self._dist_map_norm is None:
            self.set_display_frame(final_frame)
            return

        self._interp_stop.set()
        self._interp_stop = threading.Event()

        outro_dist = self._dist_map_norm.copy()
        outro_crop = self._crop_region
        outro_edge = rcfg.reveal_edge
        sx, sy = self.cfg.ui.display_scale

        if outro_crop is not None:
            cx1, cy1, cx2, cy2 = outro_crop
            px1, py1 = int(cx1 * sx), int(cy1 * sy)
            px2, py2 = int(cx2 * sx), int(cy2 * sy)
            outro_decoded = final_frame[py1:py2, px1:px2].astype(np.float32)
            with self._frame_lock:
                outro_start_f32 = self.image_present[py1:py2, px1:px2].astype(np.float32)
        else:
            outro_decoded = final_frame.astype(np.float32)
            with self._frame_lock:
                outro_start_f32 = self.image_present.astype(np.float32)

        outro_alpha_s = self._last_display_alpha

        threading.Thread(
            target=self._interp_thread,
            args=(self._interp_stop, outro_decoded, outro_dist, outro_crop,
                  pre_gen_prev, outro_alpha_s, outro_alpha_s + rcfg.reveal_outro_alpha,
                  outro_edge, rcfg.reveal_outro_duration, outro_start_f32),
            daemon=True,
        ).start()

    # -- prepare for a new generation -----------------------------------------

    def prepare_generation(self, crop_region, dist_map):
        """Set crop region / dist map and reset per-generation latent state."""
        self._crop_region = crop_region
        self._dist_map_norm = dist_map
        self._last_display_alpha = -self.cfg.reveal.reveal_edge / 2
        self._prev_latents = None
        self._step_wall_ts = 0.0

    # -- internal: compositing ------------------------------------------------

    def _decoded_to_frame_f32(self, decoded_uint8, crop_region):
        """Resize decoded uint8 RGB array to display-size float32 BGR."""
        ucfg = self.cfg.ui
        if crop_region is not None:
            cx1, cy1, cx2, cy2 = crop_region
            sx, sy = ucfg.display_scale
            pw = int(cx2 * sx) - int(cx1 * sx)
            ph = int(cy2 * sy) - int(cy1 * sy)
            return cv2.resize(cv2.cvtColor(decoded_uint8, cv2.COLOR_RGB2BGR),
                              (pw, ph), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return cv2.resize(cv2.cvtColor(decoded_uint8, cv2.COLOR_RGB2BGR),
                          ucfg.present_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def _apply_reveal(self, decoded_bgr_f32, alpha, edge, dist_map,
                      crop_region, prev_img, stop_event=None):
        """Composite decoded_bgr_f32 into image_present using the reveal wavefront."""
        rcfg = self.cfg.reveal
        white_w = float(np.clip(1.0 - alpha / max(rcfg.reveal_white_steps, 1e-6), 0.0, 1.0))
        source = decoded_bgr_f32 * (1.0 - white_w) + 255.0 * white_w if white_w > 0 else decoded_bgr_f32

        if crop_region is not None and dist_map is not None:
            cx1, cy1, cx2, cy2 = crop_region
            sx, sy = self.cfg.ui.display_scale
            px1, py1 = int(cx1 * sx), int(cy1 * sy)
            px2, py2 = int(cx2 * sx), int(cy2 * sy)
            reveal = compute_reveal(dist_map, alpha, edge)[:, :, np.newaxis]
            prev_crop = prev_img[py1:py2, px1:px2].astype(np.float32)
            composited = (source * reveal + prev_crop * (1.0 - reveal)).astype(np.uint8)
            result = prev_img.copy()
            result[py1:py2, px1:px2] = composited
        elif dist_map is not None:
            reveal = compute_reveal(dist_map, alpha, edge)[:, :, np.newaxis]
            result = (source * reveal + prev_img.astype(np.float32) * (1.0 - reveal)).astype(np.uint8)
        else:
            result = cv2.addWeighted(prev_img, 1.0 - alpha, source.astype(np.uint8), alpha, 0)

        if stop_event is not None and stop_event.is_set():
            return
        with self._frame_lock:
            self.image_present = result

    # -- internal: threads ----------------------------------------------------

    def _interp_thread_latent_lerp(self, stop_event, prev_latents_np, curr_latents_np,
                                   dist_map, crop_region, prev_img,
                                   alpha_s, alpha_e, edge, step_duration):
        """Sub-step latent interpolation: lerp, decode with TAESD, composite via reveal."""
        rcfg = self.cfg.reveal
        frame_dt = 1.0 / max(rcfg.interp_fps, 1)
        start_t = time.time()
        taesd = self._taesd
        device = self._taesd_device
        diff_np = curr_latents_np - prev_latents_np
        smooth = rcfg.latent_interp_smooth
        last_f32 = self._last_interp_f32

        while not stop_event.is_set():
            elapsed = time.time() - start_t
            t = min(elapsed / step_duration, 1.0)
            sub_alpha = alpha_s + (alpha_e - alpha_s) * t

            lerped_np = prev_latents_np + diff_np * t
            lerped_t = torch.from_numpy(lerped_np).to(device=device, dtype=torch.float16)
            with torch.no_grad():
                decoded = taesd.decode(lerped_t).sample.clamp(0, 1)
            decoded_np = decoded.cpu().permute(0, 2, 3, 1).float().numpy()[0]
            decoded_uint8 = (decoded_np * 255).astype(np.uint8)
            decoded_bgr_f32 = self._decoded_to_frame_f32(decoded_uint8, crop_region)

            if last_f32 is None or last_f32.shape != decoded_bgr_f32.shape:
                last_f32 = decoded_bgr_f32
            else:
                decoded_bgr_f32 = decoded_bgr_f32 * (1.0 - smooth) + last_f32 * smooth
                last_f32 = decoded_bgr_f32

            self._last_interp_f32 = last_f32
            self._last_display_alpha = sub_alpha
            self._apply_reveal(decoded_bgr_f32, sub_alpha, edge,
                               dist_map, crop_region, prev_img, stop_event=stop_event)

            if t >= 1.0:
                break
            next_tick = start_t + (elapsed // frame_dt + 1) * frame_dt
            sleep_t = next_tick - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _interp_thread(self, stop_event, decoded_f32, dist_map, crop_region,
                       prev_img, alpha_s, alpha_e, edge, step_duration, start_f32=None):
        """Animate the reveal wavefront over a fixed decoded frame (with optional cross-fade)."""
        frame_dt = 1.0 / max(self.cfg.reveal.interp_fps, 1)
        start_t = time.time()

        while not stop_event.is_set():
            elapsed = time.time() - start_t
            t = min(elapsed / step_duration, 1.0)
            sub_alpha = alpha_s + (alpha_e - alpha_s) * t
            frame = start_f32 * (1.0 - t) + decoded_f32 * t if start_f32 is not None else decoded_f32
            self._apply_reveal(frame, sub_alpha, edge, dist_map, crop_region, prev_img, stop_event=stop_event)
            if t >= 1.0:
                break
            next_tick = start_t + (elapsed // frame_dt + 1) * frame_dt
            sleep_t = next_tick - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)


