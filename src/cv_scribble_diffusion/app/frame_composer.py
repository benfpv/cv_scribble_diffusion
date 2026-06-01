"""Window-frame assembly for the application UI.

Builds the toolbar button state/label maps and the status payload, then renders
them through the :class:`~cv_scribble_diffusion.ui.overlay.UIOverlay`. Keeping
this assembly separate from :class:`~cv_scribble_diffusion.app.app.App` makes the
display loop a thin coordinator and the rendering inputs explicit.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from cv_scribble_diffusion.ui.overlay import StatusInfo


@dataclass
class FrameState:
    """Explicit snapshot of everything the composer needs to render one frame."""

    is_generating: bool
    is_resetting: bool
    exit_active: bool
    mask_visibility: bool
    current_inference_steps: int
    gen_count: int
    max_inference_steps: int
    generation_progress: float
    display_fps: int
    brush_thickness: int
    mask_active: np.ndarray
    mask_present: np.ndarray
    has_active_strokes: bool
    prompt_info: object
    ui_notice: Optional[str]
    thread_error: Optional[str]
    canvas_notice: Optional[str] = None


class FrameComposer:
    """Assembles toolbar/status data and renders a complete window frame."""

    def __init__(self, cfg, ui):
        self.cfg = cfg
        self.ui = ui

    def compose(self, display_frame: np.ndarray, state: FrameState) -> np.ndarray:
        """Build a complete UI frame for normal display or shutdown notice."""
        icfg = self.cfg.inference
        button_states = {
            "exit": state.exit_active,
            "reset": state.is_resetting,
            "save": False,
            "mask": state.mask_visibility,
            "undo": False,
            "brush_dec": False,
            "brush_inc": False,
            "steps_dec": False,
            "steps_inc": False,
            "fps": False,
        }
        button_labels = {
            "exit": "QUIT?" if state.exit_active else "EXIT",
            "undo": "UNDO",
            "steps_dec": "MAX-",
            "steps_inc": "MAX+",
            "fps": "FPS",
            "brush_dec": "THIN",
            "brush_inc": "THICK",
        }
        status = StatusInfo(
            quality=state.current_inference_steps if (
                state.is_generating or state.gen_count > 0) else 0,
            quality_min=icfg.min_inference_steps,
            quality_max=state.max_inference_steps,
            gen_count=state.gen_count,
            display_fps=state.display_fps,
            brush_thickness=state.brush_thickness,
            ui_notice=state.ui_notice,
            thread_error=state.thread_error,
        )
        return self.ui.compose_frame(
            display_frame,
            state.mask_active,
            state.mask_present,
            state.mask_visibility,
            state.has_active_strokes,
            state.generation_progress,
            button_states,
            button_labels,
            status=status,
            prompt=state.prompt_info,
            canvas_notice=state.canvas_notice,
        )
