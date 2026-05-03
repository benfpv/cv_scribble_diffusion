"""Canvas state: stroke masks, display image, brush logic, and feathered paste-back."""

from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image

from colorspace import gray_to_bgr
from config import AppConfig


@dataclass
class CanvasSnapshot:
    """Restorable canvas state captured before a user edit."""

    mask: np.ndarray
    mask_present: np.ndarray
    image: Image.Image


class Canvas:
    """Owns all mask/image buffers and exposes drawing, reset, and compositing helpers."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._init_buffers()

        # Brush derived values
        self.brush_thickness = cfg.ui.clamp_brush_thickness(cfg.ui.brush_thickness)
        self._update_brush_geometry()

        # Drawing state
        self.drawing = False
        self.has_active_strokes = False
        self.prev_x = -1
        self.prev_y = -1
        self._last_committed_active = np.zeros(cfg.ui.present_size, dtype="uint8")

    # -- buffer management ----------------------------------------------------

    def _init_buffers(self):
        ucfg = self.cfg.ui
        self.mask = np.zeros(ucfg.image_size, dtype="uint8")
        self.mask_active = np.zeros(ucfg.present_size, dtype="uint8")
        self.mask_present = np.zeros((*ucfg.present_size, 3), dtype="uint8")
        self.image = Image.new("RGB", ucfg.image_size, (0, 0, 0))

    def reset(self):
        """Clear all stroke data and imagery. Does NOT touch threading state."""
        self.drawing = False
        self.has_active_strokes = False
        self._init_buffers()
        self._last_committed_active = np.zeros(self.cfg.ui.present_size, dtype="uint8")

    def snapshot(self) -> CanvasSnapshot:
        """Capture restorable canvas state for undo."""
        return CanvasSnapshot(
            mask=self.mask.copy(),
            mask_present=self.mask_present.copy(),
            image=self.image.copy(),
        )

    def restore(self, snapshot: CanvasSnapshot):
        """Restore a previously captured canvas state."""
        self.mask = snapshot.mask.copy()
        self.mask_present = snapshot.mask_present.copy()
        self.image = snapshot.image.copy()
        self.mask_active = np.zeros(self.cfg.ui.present_size, dtype="uint8")
        self._last_committed_active = np.zeros(self.cfg.ui.present_size, dtype="uint8")
        self.drawing = False
        self.has_active_strokes = False
        self.prev_x = -1
        self.prev_y = -1

    # -- coordinate validation ------------------------------------------------

    def _clamp_coords(self, x: int, y: int) -> tuple:
        """Clamp coordinates to valid present_size bounds."""
        w, h = self.cfg.ui.present_size
        return max(0, min(x, w - 1)), max(0, min(y, h - 1))

    # -- drawing --------------------------------------------------------------

    def begin_stroke(self, x: int, y: int):
        x, y = self._clamp_coords(x, y)
        self.drawing = True
        self.has_active_strokes = True
        self._last_committed_active.fill(0)
        cv2.circle(self.mask_active, (x, y), self.brush_point_radius, 150, -1)
        self.prev_x = x
        self.prev_y = y

    def continue_stroke(self, x: int, y: int):
        if not self.drawing:
            return
        x, y = self._clamp_coords(x, y)
        cv2.line(self.mask_active, (self.prev_x, self.prev_y), (x, y), 150, self.brush_stroke_thickness)
        self.prev_x = x
        self.prev_y = y

    def end_stroke(self, x: int, y: int):
        x, y = self._clamp_coords(x, y)
        self.commit_active_to_mask()
        self.drawing = False
        self.has_active_strokes = False
        self.prev_x = x
        self.prev_y = y
        self.mask_active = np.zeros(self.cfg.ui.present_size, dtype="uint8")
        self._last_committed_active = np.zeros(self.cfg.ui.present_size, dtype="uint8")

    def commit_active_to_mask(self):
        """Burn any active strokes into the persistent mask (called each display loop)."""
        if self.has_active_strokes:
            ucfg = self.cfg.ui
            delta_active = cv2.subtract(self.mask_active, self._last_committed_active)
            if not np.any(delta_active):
                return False
            resized = cv2.resize(delta_active, ucfg.image_size, interpolation=cv2.INTER_NEAREST)
            self.mask = cv2.add(self.mask, resized)
            self.mask_present = cv2.resize(
                gray_to_bgr(self.mask),
                ucfg.present_size, interpolation=cv2.INTER_LINEAR,
            )
            self._last_committed_active = self.mask_active.copy()
            return True
        return False

    def set_brush_thickness(self, thickness: int):
        self.brush_thickness = self.cfg.ui.clamp_brush_thickness(thickness)
        self._update_brush_geometry()

    def _update_brush_geometry(self):
        """Keep brush point and stroke widths in sync with the user-selected size."""
        self.brush_stroke_thickness = self.cfg.ui.brush_stroke_thickness_for(
            self.brush_thickness,
        )
        self.brush_point_radius = self.cfg.ui.brush_point_radius_for(self.brush_thickness)

    # -- image patching -------------------------------------------------------

    def patch_image(self, region, result_pil: Image.Image, feather_px: int):
        """Feathered paste-back of *result_pil* into self.image at *region* (x1,y1,x2,y2)."""
        cx1, cy1, cx2, cy2 = region
        cw, ch = cx2 - cx1, cy2 - cy1
        result_np = np.array(result_pil)
        img_np = np.array(self.image)
        feather_alpha = _make_feather_mask(cw, ch, min(feather_px, 32))
        a3 = np.stack([feather_alpha] * 3, axis=-1)
        img_np[cy1:cy2, cx1:cx2] = (
            result_np * a3 + img_np[cy1:cy2, cx1:cx2] * (1 - a3)
        ).astype(np.uint8)
        self.image = Image.fromarray(img_np)


def _make_feather_mask(w: int, h: int, feather: int) -> np.ndarray:
    """Center-weighted Gaussian falloff used to soften paste-back seams.

    Returns a float32 ``(h, w)`` mask with peak ~1.0 in the middle that
    falls off toward the edges. Despite the name this is *not* a true
    distance-from-edge ramp — it's a Gaussian blur of an all-ones rectangle
    whose center stays opaque while the borders tail off, which is what the
    inpaint compositor needs to hide hard crop seams.
    """
    mask = np.ones((h, w), dtype=np.float32)
    if feather > 1:
        ksize = feather * 4 + 1
        if ksize % 2 == 0:
            ksize += 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather)
    return mask
