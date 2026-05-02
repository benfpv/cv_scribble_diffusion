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
    display_fps: int = 0
    brush_thickness: int = 0
    ui_notice: Optional[str] = None
    thread_error: Optional[str] = None


_DEFAULT_BUTTONS: Tuple[ButtonDef, ...] = (
    ButtonDef("exit", "EXIT", 42, (58, 58, 58), (86, 86, 86)),
    ButtonDef("reset", "CLEAR", 46, (56, 72, 138), (78, 98, 180)),
    ButtonDef("save", "SAVE", 42, (70, 104, 132), (98, 142, 174)),
    ButtonDef("undo", "UNDO", 44, (124, 84, 68), (172, 116, 92)),
    ButtonDef("mask", "MASK", 42, (62, 136, 82), (44, 96, 58)),
    ButtonDef("brush_dec", "THIN", 42, (92, 84, 70), (130, 120, 102)),
    ButtonDef("brush_inc", "THICK", 46, (92, 84, 70), (130, 120, 102)),
    ButtonDef("steps_dec", "MAX-", 42, (92, 84, 120), (130, 118, 164)),
    ButtonDef("steps_inc", "MAX+", 42, (92, 84, 120), (130, 118, 164)),
    ButtonDef("fps", "FPS", 44, (84, 96, 110), (120, 136, 154)),
)

_BUTTON_GAP = 4
_GROUP_GAP = 14
_BUTTON_V_PAD = 4
_WINDOW_BG = (20, 20, 20)
_TOOLBAR_BG = (28, 28, 28)
_STATUS_BG = (24, 24, 24)
_PANEL_BORDER = (54, 54, 54)
_PROGRESS_BG = (46, 46, 46)
_PROGRESS_FG = (86, 164, 236)
_CANVAS_BORDER = (82, 82, 82)
_RAIL_BG = (18, 18, 19)
_RAIL_DIVIDER = (48, 48, 52)
_RAIL_TEXT = (178, 186, 194)
_RAIL_ACCENT = _PROGRESS_FG
_GROUP_BREAK_AFTER = {"undo", "mask", "brush_inc"}


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
        ww = cfg.window_size[0]
        total_w = sum(b.width for b in self._buttons)
        total_w += sum(self._gap_after(btn.action) for btn in self._buttons[:-1])
        x = (ww - total_w) // 2
        btn_h = cfg.toolbar_height - 2 * _BUTTON_V_PAD
        y1 = _BUTTON_V_PAD
        y2 = y1 + btn_h
        rects = []
        for b in self._buttons:
            rects.append((b.action, x, y1, x + b.width, y2))
            x += b.width + self._gap_after(b.action)
        self._button_rects = rects

    @staticmethod
    def _gap_after(action: str) -> int:
        """Use larger gaps between logical toolbar groups."""
        if action in _GROUP_BREAK_AFTER:
            return _GROUP_GAP
        return _BUTTON_GAP

    # -- public properties ----------------------------------------------------

    @property
    def title_rail_width(self) -> int:
        """Width of the left identity rail, or 0 when disabled."""
        if not self._cfg.show_title_rail:
            return 0
        return max(0, self._cfg.title_rail_width)

    @property
    def canvas_x_offset(self) -> int:
        """Horizontal pixel offset from left of window to left of canvas."""
        return self.title_rail_width + self._cfg.canvas_margin

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
        generation_progress : 0.0-1.0 progress bar fill ratio. Reset to 0.0
            by ``Animator.prepare_generation`` at the start of each cycle and
            held at 1.0 by the outro until the next cycle begins, so the bar
            does not flicker between back-to-back generations.
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
        frame[:, :] = _WINDOW_BG

        # -- toolbar background strip --
        if cfg.show_toolbar:
            frame[:cfg.toolbar_height, :] = _TOOLBAR_BG
            cv2.line(frame, (0, cfg.toolbar_height - 1), (ww - 1, cfg.toolbar_height - 1), _PANEL_BORDER, 1)

        # -- left identity rail --
        if self.title_rail_width > 0:
            self._draw_identity_rail(frame)

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
        # Fill is driven by ``generation_progress`` alone. The animator resets
        # this to 0.0 at the start of each cycle and snaps to 1.0 on outro
        # completion, so chained cycles transition full → 0 → full without a
        # blank frame in between.
        bar_y = cy + ph
        frame[bar_y:bar_y + cfg.progress_bar_height, cx:cx + pw] = _PROGRESS_BG
        if generation_progress > 0:
            fill_w = max(1, int(pw * min(generation_progress, 1.0)))
            frame[bar_y:bar_y + cfg.progress_bar_height, cx:cx + fill_w] = _PROGRESS_FG

        # -- status bar --
        if status is not None:
            status_y = bar_y + cfg.progress_bar_height
            frame[status_y:status_y + cfg.status_bar_height, cx:cx + pw] = _STATUS_BG
            self._draw_status(frame, status, cx, bar_y + cfg.progress_bar_height, pw)

        return frame

    # -- internal drawing helpers ---------------------------------------------

    def _draw_identity_rail(self, frame: np.ndarray):
        """Render the left SCR identity rail."""
        rail_w = self.title_rail_width
        if rail_w <= 0:
            return

        frame[:, :rail_w] = _RAIL_BG
        cv2.line(frame, (rail_w - 1, 0), (rail_w - 1, frame.shape[0] - 1), _RAIL_DIVIDER, 1)

        mark = "".join(ch for ch in self._cfg.title_mark.upper() if ch.isalnum())[:3]
        if not mark:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.46
        thickness = 1
        y = self.canvas_y_offset + 10
        for ch in mark:
            (tw, th), _ = cv2.getTextSize(ch, font, scale, thickness)
            x = max(1, (rail_w - tw) // 2)
            baseline = y + th
            cv2.putText(frame, ch, (x, baseline), font, scale, _RAIL_TEXT, thickness, cv2.LINE_AA)
            y = baseline + 9

        accent_x = rail_w // 2
        accent_y1 = y + 2
        accent_y2 = min(accent_y1 + 18, frame.shape[0] - 8)
        if accent_y2 > accent_y1:
            cv2.line(frame, (accent_x, accent_y1), (accent_x, accent_y2), _RAIL_ACCENT, 1)

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
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), _PANEL_BORDER, 1)
            # Use dynamic label override if provided, else default
            label = (button_labels or {}).get(action, btn.label)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.27
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
        color = (152, 152, 152)
        text_y = y_top + 12

        # Thread error takes precedence over normal status. A small filled
        # "ERR" badge is rendered on the left so the error stands out from
        # the regular metric strip even at small font sizes.
        if status.thread_error:
            error_color = (50, 50, 220)  # red in BGR
            badge_w = 30
            badge_h = 12
            badge_x1 = x_off + 2
            badge_y1 = text_y - badge_h + 2
            badge_x2 = badge_x1 + badge_w
            badge_y2 = badge_y1 + badge_h
            cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2),
                          error_color, -1)
            cv2.putText(frame, "ERR", (badge_x1 + 3, badge_y2 - 3),
                        font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            error_text = status.thread_error[:80]
            cv2.putText(frame, error_text, (badge_x2 + 6, text_y),
                        font, scale, error_color, thickness, cv2.LINE_AA)
            return

        if status.ui_notice:
            notice_color = (90, 190, 230)
            notice_text = status.ui_notice[:90]
            cv2.putText(frame, notice_text, (x_off + 2, text_y),
                        font, scale, notice_color, thickness, cv2.LINE_AA)
            return

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
        return (
            f"Generations: {status.gen_count} | FPS: {status.display_fps} | "
            f"Brush: {status.brush_thickness}"
        )
