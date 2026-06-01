"""Reveal-wavefront animation and deterministic display compositing.

Noise generation and reveal math live in reveal.py as pure functions.
The Animator keeps a small amount of state and advances animation from the
display loop instead of spawning per-step worker threads.
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from cv_scribble_diffusion.config import AppConfig
from cv_scribble_diffusion.generation.reveal import (
    build_dist_map, compute_reveal, compute_outro_duration,
)
from cv_scribble_diffusion.ui.latent_decoder import LatentDecoder
from cv_scribble_diffusion.ui.reveal_compositor import RevealCompositor
from cv_scribble_diffusion.infra.runtime_logging import get_logger


logger = get_logger(__name__)


@dataclass
class _AnimationPhase:
    """Single in-flight visual transition driven by the display loop."""

    kind: str
    start_time: float
    duration: float
    alpha_start: float
    alpha_end: float
    preview_start: np.ndarray
    preview_end: np.ndarray
    prev_image: np.ndarray
    dist_map: Optional[np.ndarray]
    crop_region: Optional[Tuple[int, int, int, int]]
    final_frame: Optional[np.ndarray] = None


class Animator:
    """Owns the display frame and reveal state machine."""

    def __init__(self, cfg: AppConfig, taesd, taesd_device):
        self.cfg = cfg
        self._decoder = LatentDecoder(cfg, taesd, taesd_device)
        self._compositor = RevealCompositor(cfg)

        self._frame_lock = threading.Lock()
        self._outro_done = threading.Event()
        self._outro_done.set()

        ps = cfg.ui.present_size
        self._front = np.zeros((*ps, 3), dtype="uint8")
        self._back = np.zeros((*ps, 3), dtype="uint8")
        self.prev_image_present = np.zeros((*ps, 3), dtype="uint8")

        self._generation_progress: float = 0.0
        self._crop_region: Optional[Tuple[int, int, int, int]] = None
        self._dist_map_norm: Optional[np.ndarray] = None
        self._last_display_alpha = 0.0
        self._last_source_f32: Optional[np.ndarray] = None
        self._last_source_region: Optional[Tuple[int, int, int, int]] = None
        self._step_wall_time = 1.0
        self._step_wall_ts = 0.0
        self._phase: Optional[_AnimationPhase] = None

    # -- public helpers -------------------------------------------------------

    def get_display_frame(self) -> np.ndarray:
        with self._frame_lock:
            self._advance_phase_locked(time.time())
            return self._front

    def set_display_frame(self, frame: np.ndarray):
        with self._frame_lock:
            copied = frame.copy()
            self._back = copied
            self._front = self._back
            self._last_source_f32 = copied.astype(np.float32)
            self._last_source_region = None

    @property
    def generation_progress(self) -> float:
        with self._frame_lock:
            return self._generation_progress

    def reset(self):
        """Reset all animation state for a fresh generation cycle."""
        ps = self.cfg.ui.present_size
        with self._frame_lock:
            self._phase = None
            self._back = np.zeros((*ps, 3), dtype="uint8")
            self._front = self._back
            self.prev_image_present = np.zeros((*ps, 3), dtype="uint8")
            self._crop_region = None
            self._dist_map_norm = None
            self._last_display_alpha = 0.0
            self._last_source_f32 = None
            self._last_source_region = None
            self._step_wall_ts = 0.0
            self._generation_progress = 0.0
            self._outro_done.set()
        logger.info("Animator reset")

    def stop(self):
        """Stop animation progression and unblock any waits."""
        with self._frame_lock:
            self._phase = None
            self._outro_done.set()
        logger.info("Animator stopped")

    def wait_for_outro(self, timeout: float = None) -> bool:
        """Block until the current outro (if any) is finished."""
        return self._outro_done.wait(timeout=timeout)

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
        """Decode the current preview once and stage a deterministic step phase."""
        now = time.time()
        if self._step_wall_ts > 0:
            self._step_wall_time = now - self._step_wall_ts
        self._step_wall_ts = now

        raw_alpha = (step + 1) / max(total_steps, 1)

        with self._frame_lock:
            self._generation_progress = raw_alpha
            crop_region = self._crop_region
            has_dist_map = self._dist_map_norm is not None

        # No delta reveal map means this is a refinement pass: avoid step-time
        # TAESD preview compositing across the full frame, and wait for outro.
        if not has_dist_map:
            logger.debug(
                "Skipping step phase: no dist_map (refinement pass) step=%s/%s",
                step + 1,
                total_steps,
            )
            return

        preview_f32 = self._decode_latents_to_frame_f32(latents_tensor, crop_region)

        rcfg = self.cfg.reveal
        threshold = rcfg.reveal_start_threshold
        if raw_alpha < threshold:
            with self._frame_lock:
                self._last_source_f32 = preview_f32
                self._last_source_region = crop_region
            logger.debug(
                "Skipping visible step phase: step=%s/%s raw_alpha=%.3f threshold=%.3f",
                step + 1,
                total_steps,
                raw_alpha,
                threshold,
            )
            return

        alpha = (raw_alpha - threshold) / max(1.0 - threshold, 1e-6)
        duration = min(
            max(self._step_wall_time, rcfg.step_phase_min_duration),
            rcfg.step_phase_max_duration,
        )

        with self._frame_lock:
            start_preview = self._current_preview_locked(now, preview_f32)
            alpha_start = self._current_alpha_locked(now)
            dist_map = self._dist_map_norm.copy() if self._dist_map_norm is not None else None
            prev_img = self.prev_image_present.copy()
            self._phase = _AnimationPhase(
                kind="step",
                start_time=now,
                duration=duration,
                alpha_start=alpha_start,
                alpha_end=alpha,
                preview_start=start_preview,
                preview_end=preview_f32,
                prev_image=prev_img,
                dist_map=dist_map,
                crop_region=crop_region,
            )
            self._last_source_f32 = preview_f32
            self._last_source_region = crop_region
            self._outro_done.set()
        logger.debug(
            "Staged step phase: step=%s/%s alpha=%.3f duration=%.3fs crop=%s",
            step + 1,
            total_steps,
            alpha,
            duration,
            crop_region,
        )

    # -- outro launcher -------------------------------------------------------

    def start_outro(self, final_frame: np.ndarray, generation_duration: float):
        """Stage the post-generation outro reveal phase."""
        rcfg = self.cfg.reveal
        now = time.time()

        with self._frame_lock:
            dist_map = self._dist_map_norm.copy() if self._dist_map_norm is not None else None
            crop_region = self._crop_region
            pre_gen_prev = self.prev_image_present.copy()

            if rcfg.reveal_outro_alpha <= 0:
                copied = final_frame.copy()
                self._back = copied
                self._front = self._back
                self._last_source_f32 = copied.astype(np.float32)
                self._last_source_region = None
                self._phase = None
                self._generation_progress = 1.0
                self._update_prev_image(copied, dist_map=dist_map, crop_region=crop_region)
                self._outro_done.set()
                logger.debug("Outro bypassed: outro disabled")
                return

            outro_preview = self._frame_to_phase_preview(final_frame, crop_region)
            start_preview = self._current_preview_locked(now, outro_preview)
            alpha_start = self._current_alpha_locked(now)
            outro_duration = compute_outro_duration(
                generation_duration,
                rcfg.reveal_outro_duration_ratio,
                rcfg.reveal_outro_min_duration,
                rcfg.reveal_outro_max_duration,
            )
            if dist_map is None:
                # Refinement pass: global crossfade should start immediately
                # and finish quickly, without reveal-wave delay semantics.
                alpha_start = max(0.0, alpha_start)
                alpha_end = 1.0
                outro_duration = max(
                    rcfg.refinement_crossfade_min_duration,
                    min(outro_duration, rcfg.refinement_crossfade_max_duration),
                )
                logger.info(
                    "Refinement outro profile: alpha_start=%.3f alpha_end=%.3f duration=%.3fs",
                    alpha_start,
                    alpha_end,
                    outro_duration,
                )
            else:
                full_coverage_alpha = 1.0 + rcfg.reveal_edge
                alpha_end = max(full_coverage_alpha, alpha_start + rcfg.reveal_outro_alpha)

            self._phase = _AnimationPhase(
                kind="outro",
                start_time=now,
                duration=outro_duration,
                alpha_start=alpha_start,
                alpha_end=alpha_end,
                preview_start=start_preview,
                preview_end=outro_preview,
                prev_image=pre_gen_prev,
                dist_map=dist_map,
                crop_region=crop_region,
                final_frame=final_frame.copy(),
            )
            self._outro_done.clear()
        logger.info(
            "Staged outro: duration=%.3fs alpha_start=%.3f alpha_end=%.3f crop=%s",
            outro_duration,
            alpha_start,
            alpha_end,
            crop_region,
        )

    # -- prepare for a new generation -----------------------------------------

    def prepare_generation(self, crop_region, dist_map):
        """Set crop region / dist map and reset per-generation reveal state."""
        with self._frame_lock:
            self._crop_region = crop_region
            self._dist_map_norm = dist_map
            if dist_map is None:
                self._last_display_alpha = 0.0
            else:
                self._last_display_alpha = -self.cfg.reveal.reveal_edge / 2
            self._last_source_f32 = None
            self._last_source_region = None
            self._step_wall_ts = 0.0
            self._generation_progress = 0.0
            self._phase = None
            self._outro_done.set()
        logger.info("Prepared generation: crop=%s dist_map_shape=%s", crop_region, None if dist_map is None else dist_map.shape)

    def set_interp_fps(self, fps: int):
        """Update the configured display interpolation rate at runtime."""
        self.cfg.ui.interp_fps = fps

    def _update_prev_image(self, final_frame: np.ndarray,
                           dist_map: Optional[np.ndarray] = None,
                           crop_region: Optional[Tuple[int, int, int, int]] = None):
        """Selectively update prev_image_present only in delta-revealed areas."""
        dist_map = self._dist_map_norm if dist_map is None else dist_map
        crop_region = self._crop_region if crop_region is None else crop_region
        if dist_map is None:
            self.prev_image_present = final_frame.copy()
            return self.prev_image_present.copy()

        rcfg = self.cfg.reveal
        # Commit fully across the dilated inpaint boundary (dist <= 1) while
        # keeping outside pixels (dist == 2) untouched.
        commit_alpha = 1.0 + rcfg.reveal_edge / 2.0
        mask = compute_reveal(dist_map, commit_alpha, rcfg.reveal_edge)
        r3 = mask[:, :, np.newaxis]

        if crop_region is not None:
            px1, py1, px2, py2 = self._present_crop_bounds(crop_region)
            self.prev_image_present[py1:py2, px1:px2] = (
                final_frame[py1:py2, px1:px2].astype(np.float32) * r3
                + self.prev_image_present[py1:py2, px1:px2].astype(np.float32) * (1.0 - r3)
            ).astype(np.uint8)
        else:
            self.prev_image_present = (
                final_frame.astype(np.float32) * r3
                + self.prev_image_present.astype(np.float32) * (1.0 - r3)
            ).astype(np.uint8)
        return self.prev_image_present.copy()

    # -- internal: state evaluation ------------------------------------------

    def _advance_phase_locked(self, now: float):
        """Advance the active phase, update the front buffer, and finish phases."""
        phase = self._phase
        if phase is None:
            return

        t = self._phase_progress(phase, now)
        alpha = phase.alpha_start + (phase.alpha_end - phase.alpha_start) * t
        preview = self._compositor.blend_preview(phase.preview_start, phase.preview_end, t)
        result = self._compose_reveal(preview, alpha, phase.dist_map, phase.crop_region, phase.prev_image)
        self._last_display_alpha = alpha

        if t >= 1.0:
            if phase.kind == "outro" and phase.final_frame is not None:
                final_frame = phase.final_frame.copy()
                self._last_source_f32 = phase.preview_end
                self._last_source_region = phase.crop_region
                self._phase = None
                self._generation_progress = 1.0
                merged = self._update_prev_image(final_frame, dist_map=phase.dist_map, crop_region=phase.crop_region)
                self._back = merged
                self._front = self._back
                self._outro_done.set()
                logger.info("Completed outro and committed merged frame")
                return

            self._last_source_f32 = phase.preview_end
            self._last_source_region = phase.crop_region
            self._phase = None
            logger.debug("Completed step phase")

        self._back = result
        self._front = self._back

    def _current_preview_locked(self, now: float, fallback: np.ndarray) -> np.ndarray:
        """Current decoded preview in phase-space for continuity across updates."""
        phase = self._phase
        if phase is not None:
            t = self._phase_progress(phase, now)
            current = self._compositor.blend_preview(phase.preview_start, phase.preview_end, t)
            if current.shape == fallback.shape:
                return current
        if (
            self._last_source_f32 is not None
            and self._last_source_f32.shape == fallback.shape
            and self._last_source_region == self._crop_region
        ):
            return self._last_source_f32.copy()
        return fallback.copy()

    def _current_alpha_locked(self, now: float) -> float:
        """Current display alpha, including unfinished phase progress."""
        phase = self._phase
        if phase is None:
            return self._last_display_alpha
        t = self._phase_progress(phase, now)
        return phase.alpha_start + (phase.alpha_end - phase.alpha_start) * t

    @staticmethod
    def _phase_progress(phase: _AnimationPhase, now: float) -> float:
        duration = max(phase.duration, 1e-6)
        return min(max((now - phase.start_time) / duration, 0.0), 1.0)

    def _frame_to_phase_preview(self, frame: np.ndarray,
                                crop_region: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        """Extract the phase-local preview from a full present-size frame."""
        if crop_region is None:
            return frame.astype(np.float32)
        px1, py1, px2, py2 = self._present_crop_bounds(crop_region)
        return frame[py1:py2, px1:px2].astype(np.float32)

    def _present_crop_bounds(self, crop_region: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Map an image-space crop to present-space pixel bounds."""
        return self._compositor.present_crop_bounds(crop_region)

    # -- internal: decoding / compositing ------------------------------------

    def _decode_latents_to_frame_f32(self, latents_tensor, crop_region):
        """Decode current latents to a display-sized float32 BGR preview."""
        return self._decoder.decode_to_frame_f32(latents_tensor, crop_region)

    def _compose_reveal(self, decoded_bgr_f32, alpha, dist_map,
                        crop_region, prev_img) -> np.ndarray:
        """Composite a preview frame into the previous image using reveal math."""
        return self._compositor.compose_reveal(
            decoded_bgr_f32, alpha, dist_map, crop_region, prev_img)


