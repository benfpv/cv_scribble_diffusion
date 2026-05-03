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
    text_color: Tuple[int, int, int] = (255, 255, 255)
    text_color_active: Tuple[int, int, int] = (255, 255, 255)


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


@dataclass(frozen=True)
class PromptInfo:
    """Runtime prompt text displayed in the bottom prompt editor."""
    text: str
    editing: bool = False
    cursor: int = 0
    cursor_visible: bool = False
    max_chars: int = 0
    selection_start: int = 0
    selection_end: int = 0


_DEFAULT_BUTTONS: Tuple[ButtonDef, ...] = (
    ButtonDef(
        "exit", "EXIT", 42,
        (58, 58, 58), (42, 48, 196),
        (230, 230, 230), (255, 255, 255),
    ),
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
_BUTTON_V_PAD = 9
_WINDOW_BG = (20, 20, 20)
_TOOLBAR_BG = (28, 28, 28)
_STATUS_BG = (24, 24, 24)
_PANEL_BORDER = (54, 54, 54)
_PROGRESS_BG = (46, 46, 46)
_ACCENT = (86, 164, 236)
_PROGRESS_FG = (68, 96, 112)
_CANVAS_BORDER = (82, 82, 82)
_BRUSH_PREVIEW_SLOT_WIDTH = 36
_BRUSH_PREVIEW_V_PAD = 5
_BRUSH_PREVIEW_AFTER_ACTION = "brush_dec"
_BRUSH_PREVIEW_FILL = _ACCENT
_BRUSH_PREVIEW_BORDER = (188, 210, 226)
_RAIL_BG = (18, 18, 19)
_RAIL_DIVIDER = (48, 48, 52)
_RAIL_TEXT = (178, 186, 194)
_RAIL_ACCENT = _ACCENT
_PROMPT_BG = (26, 26, 28)
_PROMPT_BORDER = (60, 60, 66)
_PROMPT_ACTIVE_BORDER = _ACCENT
_PROMPT_TEXT = (212, 214, 216)
_PROMPT_MUTED = (126, 132, 138)
_PROMPT_PLACEHOLDER = (88, 92, 98)
_PROMPT_LIMIT = (80, 120, 220)
_PROMPT_SELECTION_BG = (92, 74, 132)
_PROMPT_TEXT_SCALE = 0.36
_PROMPT_META_SCALE = 0.32
_PROMPT_THICKNESS = 1
_CANVAS_NOTICE_BG = (34, 34, 38)
_CANVAS_NOTICE_DANGER_BG = (30, 42, 172)
_CANVAS_NOTICE_DANGER_BORDER = (72, 92, 236)
_CANVAS_NOTICE_DANGER_TEXT = (246, 246, 252)
_CANVAS_NOTICE_DANGER_SHADOW = (16, 16, 36)
_GROUP_BREAK_AFTER = {"undo", "mask", "brush_inc"}


class UIOverlay:
    """Layout engine and frame composer for toolbar + canvas + progress + status."""

    def __init__(self, cfg: UIConfig, buttons: Optional[Tuple[ButtonDef, ...]] = None):
        self._cfg = cfg
        self._buttons = buttons or _DEFAULT_BUTTONS
        self._brush_preview_rect: Optional[Tuple[int, int, int, int]] = None

        # Pre-compute button rectangles (pixel coords in the full window)
        self._button_rects: List[Tuple[str, int, int, int, int]] = []
        self._layout_buttons()

    # -- layout ---------------------------------------------------------------

    def _layout_buttons(self):
        """Compute centred button positions in the toolbar strip."""
        cfg = self._cfg
        ww = cfg.window_size[0]
        self._brush_preview_rect = None
        include_brush_preview = cfg.show_toolbar and any(
            button.action == _BRUSH_PREVIEW_AFTER_ACTION for button in self._buttons
        )
        total_w = sum(b.width for b in self._buttons)
        if include_brush_preview:
            total_w += _BUTTON_GAP + _BRUSH_PREVIEW_SLOT_WIDTH
        total_w += sum(self._gap_after(btn.action) for btn in self._buttons[:-1])
        x = (ww - total_w) // 2
        btn_h = cfg.toolbar_height - 2 * _BUTTON_V_PAD
        y1 = _BUTTON_V_PAD
        y2 = y1 + btn_h
        rects = []
        for b in self._buttons:
            rects.append((b.action, x, y1, x + b.width, y2))
            x += b.width
            if include_brush_preview and b.action == _BRUSH_PREVIEW_AFTER_ACTION:
                x += _BUTTON_GAP
                preview_y1 = _BRUSH_PREVIEW_V_PAD
                preview_y2 = max(preview_y1 + 1, cfg.toolbar_height - _BRUSH_PREVIEW_V_PAD)
                self._brush_preview_rect = (
                    x, preview_y1, x + _BRUSH_PREVIEW_SLOT_WIDTH, preview_y2,
                )
                x += _BRUSH_PREVIEW_SLOT_WIDTH
            x += self._gap_after(b.action)
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

    @property
    def prompt_box_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Prompt textbox rectangle as (x1, y1, x2, y2), or None when disabled."""
        cfg = self._cfg
        if not cfg.show_prompt_box:
            return None
        x1 = self.canvas_x_offset
        y1 = (
            self.canvas_y_offset + cfg.present_size[1] +
            cfg.progress_bar_height + cfg.status_bar_height + cfg.prompt_box_gap
        )
        return (x1, y1, x1 + cfg.present_size[0], y1 + cfg.prompt_box_height)

    @property
    def brush_preview_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Toolbar brush preview rectangle, or None when the toolbar is hidden."""
        return self._brush_preview_rect

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

    def prompt_hit_test(self, x: int, y: int) -> bool:
        """Return True when (x, y) is inside the prompt textbox."""
        rect = self.prompt_box_rect
        if rect is None:
            return False
        x1, y1, x2, y2 = rect
        return x1 <= x < x2 and y1 <= y < y2

    def prompt_cursor_index(self, text: str, x: int, cursor: int = 0,
                            max_chars: Optional[int] = None) -> int:
        """Estimate prompt cursor index from an x coordinate inside the textbox."""
        rect = self.prompt_box_rect
        if rect is None:
            return 0
        limit = self._effective_prompt_max(max_chars)
        text = text[:limit]
        cursor = max(0, min(cursor, len(text)))
        layout = self._prompt_layout(rect, limit)
        visible = self._visible_prompt_text(text, cursor, layout["text_width"])
        rel_x = max(0, x - layout["text_x"])

        prefix = visible["prefix"]
        if rel_x <= self._text_width(prefix, _PROMPT_TEXT_SCALE):
            return visible["start"]

        for idx in range(visible["start"], visible["end"] + 1):
            segment = prefix + text[visible["start"]:idx]
            seg_w = self._text_width(segment, _PROMPT_TEXT_SCALE)
            if rel_x <= seg_w:
                if idx > visible["start"]:
                    prev = prefix + text[visible["start"]:idx - 1]
                    prev_w = self._text_width(prev, _PROMPT_TEXT_SCALE)
                    if rel_x - prev_w < seg_w - rel_x:
                        return idx - 1
                return idx
        return visible["end"]

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
        prompt: Optional[PromptInfo] = None,
        canvas_notice: Optional[str] = None,
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
        prompt        : optional prompt editor metadata for the bottom textbox
        canvas_notice : optional banner message rendered across the drawing area
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

        if canvas_notice:
            self._draw_canvas_notice(frame, canvas_notice)

        # -- toolbar --
        if cfg.show_toolbar:
            brush_thickness = status.brush_thickness if status is not None else 0
            self._draw_toolbar(frame, button_states, button_labels, brush_thickness)

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

        if prompt is not None:
            self._draw_prompt_box(frame, prompt)

        return frame

    # -- internal drawing helpers ---------------------------------------------

    def _draw_identity_rail(self, frame: np.ndarray):
        """Render the left SCR identity rail."""
        rail_w = self.title_rail_width
        if rail_w <= 0:
            return

        y_top = self._cfg.toolbar_height if self._cfg.show_toolbar else 0
        frame[y_top:, :rail_w] = _RAIL_BG
        cv2.line(frame, (rail_w - 1, y_top), (rail_w - 1, frame.shape[0] - 1), _RAIL_DIVIDER, 1)

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
                       button_labels: Optional[Dict[str, str]] = None,
                       brush_thickness: int = 0):
        """Render toolbar buttons into *frame* (mutates in place)."""
        for btn, (action, bx1, by1, bx2, by2) in zip(self._buttons, self._button_rects):
            triggered = button_states.get(action, False)
            color, text_color = self._button_style(btn, triggered)
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
            cv2.putText(frame, label, (tx, ty), font, scale, text_color, thickness, cv2.LINE_AA)
        self._draw_toolbar_brush_preview(frame, brush_thickness)

    def _button_style(self, btn: ButtonDef, triggered: bool) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Return fill/text colours for a toolbar button state."""
        if btn.action == "mask":
            if triggered:
                return btn.color, btn.text_color
            return btn.color_active, btn.text_color_active
        if triggered:
            return btn.color_active, btn.text_color_active
        return btn.color, btn.text_color

    def _draw_canvas_notice(self, frame: np.ndarray, text: str):
        """Render a prominent canvas-level notice banner."""
        message = text[:48]
        if not message:
            return

        cfg = self._cfg
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.64
        thickness = 1
        (tw, th), _ = cv2.getTextSize(message, font, scale, thickness)
        cx1 = self.canvas_x_offset
        cx2 = self.canvas_x_offset + cfg.present_size[0] - 1
        cy = self.canvas_y_offset + cfg.present_size[1] // 2
        banner_h = max(36, th + 24)
        y1 = max(self.canvas_y_offset + 8, cy - banner_h // 2)
        y2 = min(self.canvas_y_offset + cfg.present_size[1] - 8, y1 + banner_h)
        y1 = max(self.canvas_y_offset + 8, y2 - banner_h)

        cv2.rectangle(frame, (cx1, y1), (cx2, y2), _CANVAS_NOTICE_DANGER_BG, -1)
        cv2.rectangle(frame, (cx1, y1), (cx2, y2), _CANVAS_NOTICE_DANGER_BORDER, 1)
        cv2.line(frame, (cx1, y1 + 2), (cx2, y1 + 2), _CANVAS_NOTICE_BG, 1)
        cv2.line(frame, (cx1, y2 - 2), (cx2, y2 - 2), _CANVAS_NOTICE_BG, 1)

        tx = cx1 + max(12, (cfg.present_size[0] - tw) // 2)
        ty = y1 + (y2 - y1 + th) // 2 - 1
        cv2.putText(frame, message, (tx + 1, ty + 1), font, scale,
                    _CANVAS_NOTICE_DANGER_SHADOW, thickness, cv2.LINE_AA)
        cv2.putText(frame, message, (tx, ty), font, scale,
                    _CANVAS_NOTICE_DANGER_TEXT, thickness, cv2.LINE_AA)

    def _draw_prompt_box(self, frame: np.ndarray, prompt: PromptInfo):
        """Render the bottom prompt editor (mutates in place)."""
        rect = self.prompt_box_rect
        if rect is None:
            return
        x1, y1, x2, y2 = rect
        max_chars = self._effective_prompt_max(prompt.max_chars)
        text = prompt.text[:max_chars]
        cursor = max(0, min(prompt.cursor, len(text)))
        layout = self._prompt_layout(rect, max_chars)
        selection = self._prompt_selection_bounds(prompt, len(text))

        frame[y1:y2, x1:x2] = _PROMPT_BG
        border = _PROMPT_ACTIVE_BORDER if prompt.editing else _PROMPT_BORDER
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), border, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        label_color = _PROMPT_ACTIVE_BORDER if prompt.editing else _PROMPT_MUTED
        cv2.putText(frame, "PROMPT", (layout["label_x"], layout["baseline"]),
                    font, _PROMPT_META_SCALE, label_color, _PROMPT_THICKNESS, cv2.LINE_AA)

        counter = f"{len(text)}/{max_chars}"
        counter_color = _PROMPT_LIMIT if len(text) >= max_chars else _PROMPT_MUTED
        cv2.putText(frame, counter, (layout["counter_x"], layout["baseline"]),
                    font, _PROMPT_META_SCALE, counter_color, _PROMPT_THICKNESS, cv2.LINE_AA)

        if text:
            visible = self._visible_prompt_text(text, cursor, layout["text_width"])
            display_text = visible["text"]
            text_color = _PROMPT_TEXT
        else:
            visible = {"start": 0, "end": 0, "prefix": "", "text": ""}
            display_text = "Click to enter prompt"
            text_color = _PROMPT_PLACEHOLDER

        if text and selection is not None:
            self._draw_prompt_selection(frame, text, visible, selection, layout, y1, y2)

        cv2.putText(frame, display_text, (layout["text_x"], layout["baseline"]),
                    font, _PROMPT_TEXT_SCALE, text_color, _PROMPT_THICKNESS, cv2.LINE_AA)

        if prompt.editing and prompt.cursor_visible and selection is None:
            prefix = visible["prefix"]
            cursor_text = prefix + text[visible["start"]:cursor]
            cursor_x = layout["text_x"] + self._text_width(cursor_text, _PROMPT_TEXT_SCALE)
            cursor_x = max(layout["text_x"], min(layout["text_right"], cursor_x))
            cv2.line(frame, (cursor_x, y1 + 7), (cursor_x, y2 - 7), _PROMPT_ACTIVE_BORDER, 1)

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
        text_x = x_off + width - tw - 2
        cv2.putText(frame, right, (text_x, text_y),
                    font, scale, color, thickness, cv2.LINE_AA)

    def _draw_toolbar_brush_preview(self, frame: np.ndarray, brush_thickness: int):
        """Draw the actual current brush point beside the toolbar thickness controls."""
        rect = self.brush_preview_rect
        if rect is None or brush_thickness <= 0:
            return

        x1, _y1, x2, _y2 = rect
        radius = self._brush_preview_radius(brush_thickness)
        cx = (x1 + x2) // 2
        cy = self._cfg.toolbar_height // 2
        cv2.circle(frame, (cx, cy), radius, _BRUSH_PREVIEW_FILL, -1, cv2.LINE_AA)
        if radius > 0:
            cv2.circle(frame, (cx, cy), radius, _BRUSH_PREVIEW_BORDER, 1, cv2.LINE_AA)

    def _brush_preview_radius(self, brush_thickness: int) -> int:
        """Actual point radius used for toolbar brush-size preview."""
        return self._cfg.brush_point_radius_for(brush_thickness)

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

    def _prompt_layout(self, rect: Tuple[int, int, int, int], max_chars: int) -> Dict[str, int]:
        """Compute prompt field text/counter positions inside *rect*."""
        x1, y1, x2, y2 = rect
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_w = self._text_width("PROMPT", _PROMPT_META_SCALE)
        counter_template = f"{max_chars}/{max_chars}"
        counter_w = self._text_width(counter_template, _PROMPT_META_SCALE)
        (_, text_h), _ = cv2.getTextSize("Ag", font, _PROMPT_TEXT_SCALE, _PROMPT_THICKNESS)
        baseline = y1 + ((y2 - y1) + text_h) // 2 - 1
        label_x = x1 + 8
        text_x = label_x + label_w + 12
        counter_x = x2 - counter_w - 8
        text_right = max(text_x + 1, counter_x - 10)
        return {
            "label_x": label_x,
            "text_x": text_x,
            "text_right": text_right,
            "text_width": max(1, text_right - text_x),
            "counter_x": counter_x,
            "baseline": baseline,
        }

    def _visible_prompt_text(self, text: str, cursor: int, max_width: int) -> Dict[str, object]:
        """Return a clipped prompt substring that keeps *cursor* visible."""
        cursor = max(0, min(cursor, len(text)))
        start = 0
        max_width = max(1, max_width - 2)
        while start < cursor:
            prefix = "..." if start > 0 else ""
            if self._text_width(prefix + text[start:cursor], _PROMPT_TEXT_SCALE) <= max_width:
                break
            start += 1

        end = len(text)
        while end > cursor:
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text) else ""
            visible = prefix + text[start:end] + suffix
            if self._text_width(visible, _PROMPT_TEXT_SCALE) <= max_width:
                break
            end -= 1

        while start < cursor:
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(text) else ""
            visible = prefix + text[start:end] + suffix
            if self._text_width(visible, _PROMPT_TEXT_SCALE) <= max_width:
                break
            start += 1

        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(text) else ""
        return {
            "start": start,
            "end": end,
            "prefix": prefix,
            "text": prefix + text[start:end] + suffix,
        }

    def _prompt_selection_bounds(self, prompt: PromptInfo, text_len: int) -> Optional[Tuple[int, int]]:
        """Return a clipped selection range for prompt rendering, or None."""
        start = max(0, min(prompt.selection_start, text_len))
        end = max(0, min(prompt.selection_end, text_len))
        if start == end:
            return None
        return (min(start, end), max(start, end))

    def _draw_prompt_selection(self, frame: np.ndarray, text: str, visible: Dict[str, object],
                               selection: Tuple[int, int], layout: Dict[str, int],
                               y1: int, y2: int):
        """Draw selection highlight behind the visible selected prompt text."""
        visible_start = int(visible["start"])
        visible_end = int(visible["end"])
        selection_start, selection_end = selection
        if selection_end <= visible_start or selection_start >= visible_end:
            return

        highlight_start = max(selection_start, visible_start)
        highlight_end = min(selection_end, visible_end)
        prefix = str(visible["prefix"])
        pre_selection = prefix + text[visible_start:highlight_start]
        through_selection = prefix + text[visible_start:highlight_end]
        hx1 = layout["text_x"] + self._text_width(pre_selection, _PROMPT_TEXT_SCALE)
        hx2 = layout["text_x"] + self._text_width(through_selection, _PROMPT_TEXT_SCALE)
        hx1 = max(layout["text_x"], min(layout["text_right"], hx1))
        hx2 = max(hx1 + 1, min(layout["text_right"], hx2))
        cv2.rectangle(frame, (hx1, y1 + 6), (hx2, y2 - 7), _PROMPT_SELECTION_BG, -1)

    def _effective_prompt_max(self, max_chars: Optional[int] = None) -> int:
        """Return a positive prompt character limit."""
        if max_chars is None or max_chars <= 0:
            max_chars = self._cfg.prompt_max_chars
        return max(1, int(max_chars))

    @staticmethod
    def _text_width(text: str, scale: float, thickness: int = _PROMPT_THICKNESS) -> int:
        """Measure a single-line OpenCV text width in pixels."""
        if not text:
            return 0
        (width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        return width
