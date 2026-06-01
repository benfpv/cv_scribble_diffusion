"""TAESD latent decoding to display-sized BGR previews.

Isolates the GPU/TAESD decode + resize concern from the Animator's reveal
state machine so the two can evolve (and be tested) independently.
"""

from typing import Optional, Tuple

import numpy as np
import cv2
import torch

from cv_scribble_diffusion.utils.colorspace import rgb_to_bgr
from cv_scribble_diffusion.utils.geometry import present_bounds
from cv_scribble_diffusion.config import AppConfig


class LatentDecoder:
    """Decode diffusion latents into display-sized float32 BGR frames."""

    def __init__(self, cfg: AppConfig, taesd, taesd_device):
        self.cfg = cfg
        self._taesd = taesd
        self._taesd_device = taesd_device

    def decode_to_frame_f32(self, latents_tensor,
                            crop_region: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        """Decode current latents once to a display-sized float32 BGR preview."""
        lerped_t = latents_tensor.to(device=self._taesd_device, dtype=torch.float16)
        with torch.no_grad():
            decoded = self._taesd.decode(lerped_t).sample.clamp(0, 1)
        decoded_np = decoded.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        decoded_uint8 = (decoded_np * 255).astype(np.uint8)
        return self.decoded_to_frame_f32(decoded_uint8, crop_region)

    def decoded_to_frame_f32(self, decoded_uint8: np.ndarray,
                             crop_region: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        """Resize decoded uint8 RGB array to display-size float32 BGR."""
        ucfg = self.cfg.ui
        if crop_region is not None:
            px1, py1, px2, py2 = present_bounds(crop_region, ucfg.display_scale)
            pw = px2 - px1
            ph = py2 - py1
            return cv2.resize(rgb_to_bgr(decoded_uint8),
                              (pw, ph), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return cv2.resize(rgb_to_bgr(decoded_uint8),
                          ucfg.present_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
