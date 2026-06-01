"""Pure reveal-wavefront compositing math.

Stateless given an AppConfig: blends a decoded preview into the previous
display frame using the reveal distance map (or a global crossfade when no
map is present). Extracted from the Animator so the compositing rules are
isolated from the threaded phase state machine.
"""

from typing import Optional, Tuple

import numpy as np
import cv2

from cv_scribble_diffusion.utils.geometry import present_bounds
from cv_scribble_diffusion.config import AppConfig
from cv_scribble_diffusion.generation.reveal import compute_reveal, ease_progress


class RevealCompositor:
    """Composite preview frames into the previous image using reveal math."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

    def present_crop_bounds(self, crop_region: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Map an image-space crop to present-space pixel bounds."""
        return present_bounds(crop_region, self.cfg.ui.display_scale)

    @staticmethod
    def blend_preview(start: np.ndarray, end: np.ndarray, t: float) -> np.ndarray:
        if start.shape != end.shape:
            return end
        return start * (1.0 - t) + end * t

    def compose_reveal(self, decoded_bgr_f32: np.ndarray, alpha: float,
                       dist_map: Optional[np.ndarray],
                       crop_region: Optional[Tuple[int, int, int, int]],
                       prev_img: np.ndarray) -> np.ndarray:
        """Composite a preview frame into the previous image using reveal math."""
        rcfg = self.cfg.reveal
        # Keep white preflash only for dist-map reveal modes; for global
        # crossfade (dist_map is None) this manifests as full-frame flicker.
        if dist_map is None:
            white_w = 0.0
        else:
            white_w = float(np.clip(1.0 - alpha / max(rcfg.reveal_white_steps, 1e-6), 0.0, 1.0))
        source = decoded_bgr_f32 * (1.0 - white_w) + 255.0 * white_w if white_w > 0 else decoded_bgr_f32
        eased_alpha = ease_progress(alpha, rcfg.reveal_ease_power)
        edge = rcfg.reveal_edge

        if crop_region is not None and dist_map is not None:
            px1, py1, px2, py2 = self.present_crop_bounds(crop_region)
            reveal = compute_reveal(dist_map, eased_alpha, edge)[:, :, np.newaxis]
            prev_crop = prev_img[py1:py2, px1:px2].astype(np.float32)
            composited = (source * reveal + prev_crop * (1.0 - reveal)).astype(np.uint8)
            result = prev_img.copy()
            result[py1:py2, px1:px2] = composited
            return result
        if dist_map is not None:
            reveal = compute_reveal(dist_map, eased_alpha, edge)[:, :, np.newaxis]
            return (source * reveal + prev_img.astype(np.float32) * (1.0 - reveal)).astype(np.uint8)
        weight = min(max(eased_alpha, 0.0), 1.0)
        if crop_region is not None:
            px1, py1, px2, py2 = self.present_crop_bounds(crop_region)
            result = prev_img.copy()
            prev_crop = prev_img[py1:py2, px1:px2]
            blended_crop = cv2.addWeighted(prev_crop, 1.0 - weight, source.astype(np.uint8), weight, 0)
            result[py1:py2, px1:px2] = blended_crop
            return result
        return cv2.addWeighted(prev_img, 1.0 - weight, source.astype(np.uint8), weight, 0)
