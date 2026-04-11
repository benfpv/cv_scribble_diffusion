"""Toolbar, progress bar, status bar, and compositing for the OpenCV window.

The UIOverlay owns all layout math and rendering; main.py delegates hit-testing
and final frame composition to it.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import UIConfig


@dataclass(frozen=True)
class ButtonDef:
    """Definition of a single toolbar button."""
    action: str
    label: str
    width: int = 40
    color: Tuple[int, int, int] = (100, 100, 100)
    color_active: Tuple[int, int, int] = (180, 180, 180)


@dataclass(frozen=True)
class StatusInfo:
    """Runtime generation metadata displayed in the status bar."""
    quality: int
    quality_min: int
    quality_max: int
    gen_count: int


_DEFAULT_BUTTONS: Tuple[ButtonDef, ...] = (
    ButtonDef("exit", "EXIT", 40, (100, 100, 100), (180, 180, 180)),
    ButtonDef("reset", "RESET", 40, (50, 50, 150), (80, 80, 240)),
    ButtonDef("mask", "MASK", 40, (50, 200, 50), (50, 120, 50)),
    ButtonDef("fps", "FPS", 40, (120, 100, 50), (180, 150, 80)),
)

_BUTTON_GAP = 8
_BUTTON_V_PAD = 4
_PROGRESS_BG = (40, 40, 40)
_PROGRESS_FG = (200, 140, 60)
_CANVAS_BORDER = (72, 72, 72)


class UIOverlay:
    """Layout engine and frame composer for toolbar + canvas + progress + status."""

    def __init__(self, cfg: UIConfig, buttons: Optional[Tuple[ButtonDef, ...]] = None):
        self._cfg = cfg
        self._buttons = buttons or _DEFAULT_BUTTONS

        # Pre-compute button rectangles (pixel coords in the full window)
        self._button_rects: List[Tuple[str, int, int, int, int]] = []
        self._layout_buttons()

    # -- layout ---------------------------------------------------------------

    def _layout_buttons(self):
        """Compute centred button positions in the toolbar strip."""
        cfg = self._cfg
        ww = cfg.present_size[0] + 2 * cfg.canvas_margin
        total_w = sum(b.width for b in self._buttons) + _BUTTON_GAP * (len(self._buttons) - 1)
        x = (ww - total_w) // 2
        btn_h = cfg.toolbar_height - 2 * _BUTTON_V_PAD
        y1 = _BUTTON_V_PAD
        y2 = y1 + btn_h
        rects = []
        for b in self._buttons:
            rects.append((b.action, x, y1, x + b.width, y2))
            x += b.width + _BUTTON_GAP
        self._button_rects = rects

    # -- public properties ----------------------------------------------------

    @property
    def canvas_x_offset(self) -> int:
        """Horizontal pixel offset from left of window to left of canvas."""
        return self._cfg.canvas_margin

    @property
    def canvas_y_offset(self) -> int:
        """Vertical pixel offset from top of window to top of canvas."""
        tb = self._cfg.toolbar_height if self._cfg.show_toolbar else 0
        return tb + self._cfg.canvas_margin

    # -- coordinate translation -----------------------------------------------

    def canvas_coords(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """Translate window (x, y) to canvas-relative coords, or None if outside canvas."""
        cfg = self._cfg
        cx = x - self.canvas_x_offset
        cy = y - self.canvas_y_offset
        if cx < 0 or cy < 0 or cx >= cfg.present_size[0] or cy >= cfg.present_size[1]:
            return None
        return (cx, cy)

    def hit_test(self, x: int, y: int) -> Optional[str]:
        """Return the action name of the button at (x, y), or None."""
        if not self._cfg.show_toolbar:
            return None
        for action, bx1, by1, bx2, by2 in self._button_rects:
            if bx1 <= x < bx2 and by1 <= y < by2:
                return action
        return None

    # -- frame composition ----------------------------------------------------

    def compose_frame(
        self,
        canvas_frame: np.ndarray,
        mask_active: np.ndarray,
        mask_present: np.ndarray,
        show_mask: bool,
        has_active_strokes: bool,
        generation_progress: float,
        is_generating: bool,
        button_states: Dict[str, bool],
        button_labels: Optional[Dict[str, str]] = None,
        status: Optional[StatusInfo] = None,
    ) -> np.ndarray:
        """Build the full window image: toolbar + margins + canvas + progress + status.

        Parameters
        ----------
        canvas_frame : uint8 BGR at present_size
        mask_active  : uint8 gray at present_size
        mask_present : uint8 BGR at present_size
        show_mask    : whether to overlay the stroke mask
        has_active_strokes : whether there are uncommitted strokes
        generation_progress : 0.0-1.0 (used for progress bar fill)
        is_generating : whether the pipeline is currently running
        button_states : mapping action→triggered (colour swap)
        button_labels : optional mapping action→dynamic label override
        status        : optional generation metadata for the status bar
        """
        cfg = self._cfg
        pw, ph = cfg.present_size
        ww, wh = cfg.window_size
        cx = self.canvas_x_offset
        cy = self.canvas_y_offset

        frame = np.zeros((wh, ww, 3), dtype=np.uint8)

        # -- canvas region --
        display = canvas_frame.copy()
        if has_active_strokes and not show_mask:
            display = cv2.add(display, cv2.cvtColor(mask_active, cv2.COLOR_GRAY2BGR))
        if show_mask:
            display = cv2.add(display, mask_present)

        frame[cy:cy + ph, cx:cx + pw] = display
        cv2.rectangle(frame, (cx, cy), (cx + pw - 1, cy + ph - 1), _CANVAS_BORDER, 1)

        # -- toolbar --
        if cfg.show_toolbar:
            self._draw_toolbar(frame, button_states, button_labels)

        # -- progress bar (canvas-aligned) --
        bar_y = cy + ph
        frame[bar_y:bar_y + cfg.progress_bar_height, cx:cx + pw] = _PROGRESS_BG
        if is_generating and generation_progress > 0:
            fill_w = max(1, int(pw * generation_progress))
            frame[bar_y:bar_y + cfg.progress_bar_height, cx:cx + fill_w] = _PROGRESS_FG

        # -- status bar --
        if status is not None:
            self._draw_status(frame, status, cx, bar_y + cfg.progress_bar_height, pw)

        return frame

    # -- internal drawing helpers ---------------------------------------------

    def _draw_toolbar(self, frame: np.ndarray, button_states: Dict[str, bool],
                       button_labels: Optional[Dict[str, str]] = None):
        """Render toolbar buttons into *frame* (mutates in place)."""
        for btn, (action, bx1, by1, bx2, by2) in zip(self._buttons, self._button_rects):
            triggered = button_states.get(action, False)
            color = btn.color_active if triggered else btn.color
            # For the mask button, active state means mask is ON (use normal color),
            # while inactive means OFF (use active color, which is dimmer).
            if action == "mask":
                color = btn.color if triggered else btn.color_active
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
            # Use dynamic label override if provided, else default
            label = (button_labels or {}).get(action, btn.label)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.3
            thickness = 1
            (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
            tx = bx1 + (btn.width - tw) // 2
            ty = by1 + (by2 - by1 + th) // 2
            cv2.putText(frame, label, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_status(self, frame: np.ndarray, status: StatusInfo,
                     x_off: int, y_top: int, width: int):
        """Render status text below the progress bar (mutates in place)."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.32
        thickness = 1
        color = (110, 110, 110)
        text_y = y_top + 12

        # Left: quality details
        left = self._quality_text(status)
        cv2.putText(frame, left, (x_off + 2, text_y),
                    font, scale, color, thickness, cv2.LINE_AA)

        # Right: generation count
        right = self._instances_text(status)
        (tw, _), _ = cv2.getTextSize(right, font, scale, thickness)
        cv2.putText(frame, right, (x_off + width - tw - 2, text_y),
                    font, scale, color, thickness, cv2.LINE_AA)

    def _quality_text(self, status: StatusInfo) -> str:
        """Human-readable quality label for the status bar."""
        if status.quality > 0:
            cur = str(status.quality)
        else:
            cur = "pending"
        return (
            f"Quality (diffusion steps): {cur} "
            f"(min {status.quality_min}, max {status.quality_max})"
        )

    def _instances_text(self, status: StatusInfo) -> str:
        """Human-readable generation counter label for the status bar."""
        return f"Generations: {status.gen_count}"
