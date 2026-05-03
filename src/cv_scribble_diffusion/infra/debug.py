"""Optional debug-image writer.

All public methods are no-ops when DebugConfig.enabled is False, so call sites
need no guard clauses.
"""

import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

from cv_scribble_diffusion.config import DebugConfig
from cv_scribble_diffusion.utils.colorspace import rgb_to_bgr
from cv_scribble_diffusion.infra.runtime_logging import get_logger


logger = get_logger(__name__)


class DebugWriter:
    """Writes tagged debug images (and optional info text) to a folder."""

    def __init__(self, cfg: DebugConfig):
        self._cfg = cfg
        self._counter = 0
        if cfg.enabled:
            os.makedirs(cfg.dir, exist_ok=True)
            logger.info("Writing debug images to %s", cfg.dir)

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def save(self, tag: str, image, info: str = ""):
        """Save a PIL Image or numpy array with an auto-incrementing filename.

        Parameters
        ----------
        tag   : short label embedded in the filename, e.g. "crop_init"
        image : PIL Image or numpy uint8 array (HW or HWC RGB/BGR)
        info  : optional text written to a sibling .txt file
        """
        if not self._cfg.enabled:
            return
        self._counter += 1
        ts = time.strftime("%H%M%S")
        stem = os.path.join(self._cfg.dir, f"{self._counter:04d}_{ts}_{tag}")
        _write_image(stem + ".png", image)
        logger.debug("Saved debug artifact %s.png", stem)
        if info:
            with open(stem + ".txt", "w") as fh:
                fh.write(info)

    def save_annotated_crop(self, canvas_pil: Image.Image, crop, info: str = ""):
        """Save the full canvas PIL image with the crop rectangle overlaid in green."""
        if not self._cfg.enabled:
            return
        annotated = canvas_pil.copy().convert("RGB")
        draw = ImageDraw.Draw(annotated)
        cx1, cy1, cx2, cy2 = crop
        draw.rectangle([cx1, cy1, cx2 - 1, cy2 - 1], outline=(0, 255, 0), width=2)
        self.save("crop_box", annotated, info)


# ---------------------------------------------------------------------------

def _write_image(path: str, image):
    """Write PIL Image or numpy array to *path*."""
    if isinstance(image, Image.Image):
        image.save(path)
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            cv2.imwrite(path, image)
        elif image.ndim == 3 and image.shape[2] == 3:
            # Assume RGB (from PIL/diffusers); convert to BGR for OpenCV
            cv2.imwrite(path, rgb_to_bgr(image))
        else:
            cv2.imwrite(path, image)
