"""Canvas state: stroke masks, display image, brush logic, and feathered paste-back."""

import numpy as np
import cv2
from PIL import Image

from config import AppConfig


class Canvas:
    """Owns all mask/image buffers and exposes drawing, reset, and compositing helpers."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._init_buffers()

        # Brush derived values
        ucfg = cfg.ui
        self.brush_thickness = ucfg.brush_thickness
        self.brush_point_thickness = int(ucfg.brush_thickness - 0.5)
        self.brush_stroke_thickness = max(1, round(ucfg.brush_thickness * ucfg.brush_stroke_multiplier))

        # Drawing state
        self.drawing = False
        self.has_active_strokes = False
        self.prev_x = -1
        self.prev_y = -1

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

    # -- drawing --------------------------------------------------------------

    def begin_stroke(self, x: int, y: int):
        self.drawing = True
        self.has_active_strokes = True
        cv2.circle(self.mask_active, (x, y), self.brush_point_thickness, 150, -1)
        self.prev_x = x
        self.prev_y = y

    def continue_stroke(self, x: int, y: int):
        if not self.drawing:
            return
        cv2.line(self.mask_active, (self.prev_x, self.prev_y), (x, y), 150, self.brush_stroke_thickness)
        self.prev_x = x
        self.prev_y = y

    def end_stroke(self, x: int, y: int):
        self.drawing = False
        self.has_active_strokes = False
        self.prev_x = x
        self.prev_y = y
        self.mask_active = np.zeros(self.cfg.ui.present_size, dtype="uint8")

    def commit_active_to_mask(self):
        """Burn any active strokes into the persistent mask (called each display loop)."""
        if self.has_active_strokes:
            ucfg = self.cfg.ui
            resized = cv2.resize(self.mask_active, ucfg.image_size, interpolation=cv2.INTER_NEAREST)
            self.mask = cv2.add(self.mask, resized)
            self.mask_present = cv2.resize(
                cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR),
                ucfg.present_size, interpolation=cv2.INTER_LINEAR,
            )
            return True
        return False

    def set_brush_thickness(self, thickness: int):
        self.brush_thickness = max(1, min(self.cfg.ui.max_brush_thickness, thickness))
        self.brush_point_thickness = self.brush_thickness
        self.brush_stroke_thickness = int(self.brush_thickness * self.cfg.ui.brush_stroke_multiplier)

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
    """Returns a float32 (h,w) mask, 1.0 in centre fading to 0.0 at edges."""
    mask = np.ones((h, w), dtype=np.float32)
    if feather > 1:
        ksize = feather * 4 + 1
        if ksize % 2 == 0:
            ksize += 1
        mask = cv2.GaussianBlur(mask, (ksize, ksize), feather)
    return mask
