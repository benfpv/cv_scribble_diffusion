"""Tests for UIOverlay layout, hit-testing, and frame composition."""

import numpy as np
import pytest

from config import AppConfig, UIConfig
from ui import UIOverlay, ButtonDef, StatusInfo


def make_ui(cfg: UIConfig = None) -> UIOverlay:
    return UIOverlay(cfg or UIConfig())


# -- canvas_y_offset ----------------------------------------------------------

def test_canvas_y_offset_with_toolbar():
    ui = make_ui(UIConfig(toolbar_height=28, show_toolbar=True))
    assert ui.canvas_y_offset == 40  # toolbar(28) + margin(12)


def test_canvas_y_offset_without_toolbar():
    ui = make_ui(UIConfig(show_toolbar=False))
    assert ui.canvas_y_offset == 12  # margin only


# -- canvas_coords -----------------------------------------------------------

def test_canvas_coords_in_canvas():
    ui = make_ui(UIConfig(toolbar_height=28, present_size=(512, 512)))
    result = ui.canvas_coords(100, 50)
    assert result == (88, 10)  # x: 100-12, y: 50-40


def test_canvas_coords_in_toolbar():
    ui = make_ui(UIConfig(toolbar_height=28, present_size=(512, 512)))
    result = ui.canvas_coords(100, 10)
    assert result is None  # y=10 is in toolbar


def test_canvas_coords_below_canvas():
    ui = make_ui(UIConfig(toolbar_height=28, present_size=(512, 512)))
    result = ui.canvas_coords(100, 40 + 512 + 5)  # below canvas
    assert result is None


def test_canvas_coords_in_margin():
    ui = make_ui(UIConfig(toolbar_height=28, present_size=(512, 512)))
    # x=5 is in the left margin (canvas starts at x=12)
    assert ui.canvas_coords(5, 50) is None


# -- hit_test -----------------------------------------------------------------

def test_hit_test_on_button():
    ui = make_ui(UIConfig(toolbar_height=28, present_size=(512, 512)))
    # Buttons are centred; find the exit button by checking the first rect
    rects = ui._button_rects
    action, bx1, by1, bx2, by2 = rects[0]
    assert action == "exit"
    result = ui.hit_test((bx1 + bx2) // 2, (by1 + by2) // 2)
    assert result == "exit"


def test_hit_test_miss():
    ui = make_ui(UIConfig(toolbar_height=28, present_size=(512, 512)))
    result = ui.hit_test(0, 0)
    assert result is None


def test_hit_test_disabled_toolbar():
    ui = make_ui(UIConfig(show_toolbar=False))
    result = ui.hit_test(256, 5)
    assert result is None


def test_hit_test_each_button():
    ui = make_ui()
    for action, bx1, by1, bx2, by2 in ui._button_rects:
        result = ui.hit_test((bx1 + bx2) // 2, (by1 + by2) // 2)
        assert result == action


# -- compose_frame output shape ----------------------------------------------

def test_compose_frame_shape():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=True, has_active_strokes=False,
        generation_progress=0.0, is_generating=False,
        button_states={},
    )
    # h = 28 + 12 + 512 + 12 + 6 + 16 = 586, w = 12 + 512 + 12 = 536
    assert result.shape == (586, 536, 3)


def test_compose_frame_no_toolbar():
    cfg = UIConfig(present_size=(512, 512), show_toolbar=False, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0, is_generating=False,
        button_states={},
    )
    # h = 0 + 12 + 512 + 12 + 6 + 16 = 558, w = 536
    assert result.shape == (558, 536, 3)


# -- progress bar fill -------------------------------------------------------

def test_progress_bar_fills_during_generation():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.5, is_generating=True,
        button_states={},
    )
    margin = cfg.canvas_margin  # 12
    bar_row = 40 + 512  # toolbar(28) + margin(12) + canvas(512)
    # Left side (inside bar) should have accent colour
    left_pixel = result[bar_row, margin + 10]
    # Right side (inside bar, past 50%) should be dark background
    right_pixel = result[bar_row, margin + 400]
    assert int(left_pixel.sum()) > int(right_pixel.sum())


def test_progress_bar_empty_when_not_generating():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.5, is_generating=False,
        button_states={},
    )
    margin = cfg.canvas_margin
    bar_y = 40 + 512
    bar_region = result[bar_y:bar_y + 6, margin:margin + 512]
    # Should be uniform (all dark grey background)
    assert np.all(bar_region == bar_region[0, 0])


# -- mask overlay in compose --------------------------------------------------

def test_compose_overlays_mask_when_show_mask_true():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_present[100, 100] = (0, 150, 0)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=True, has_active_strokes=False,
        generation_progress=0.0, is_generating=False,
        button_states={},
    )
    # Canvas offset: (12, 40). Pixel at canvas (100,100) → window (112, 140)
    assert result[40 + 100, 12 + 100, 1] == 150


# -- fps button ---------------------------------------------------------------

def test_fps_button_in_default_layout():
    ui = make_ui()
    actions = [action for action, *_ in ui._button_rects]
    assert "fps" in actions


def test_hit_test_fps_button():
    ui = make_ui()
    for action, bx1, by1, bx2, by2 in ui._button_rects:
        if action == "fps":
            result = ui.hit_test((bx1 + bx2) // 2, (by1 + by2) // 2)
            assert result == "fps"
            break
    else:
        pytest.fail("fps button not found in layout")


def test_compose_frame_with_button_labels():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0, is_generating=False,
        button_states={},
        button_labels={"fps": "60"},
    )
    assert result.shape == (586, 536, 3)


def test_compose_frame_shape_with_four_buttons():
    """Window shape must remain correct with the extra fps button."""
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0, is_generating=False,
        button_states={},
    )
    assert result.shape == (586, 536, 3)


# -- status bar ---------------------------------------------------------------

def test_compose_frame_with_status_info():
    """StatusInfo is accepted and does not alter the frame shape."""
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)
    status = StatusInfo(quality=6, quality_min=2, quality_max=18, gen_count=3)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.5, is_generating=True,
        button_states={},
        status=status,
    )
    assert result.shape == (586, 536, 3)
    # Status bar region should have non-zero pixels (rendered text)
    status_y = 40 + 512 + 6  # canvas_y_offset + canvas + progress_bar
    status_region = result[status_y:status_y + 16, 12:524]
    assert status_region.sum() > 0


def test_canvas_x_offset():
    ui = make_ui(UIConfig(canvas_margin=12))
    assert ui.canvas_x_offset == 12


def test_quality_text_is_clear_for_active_generation():
    ui = make_ui()
    status = StatusInfo(quality=6, quality_min=2, quality_max=18, gen_count=3)
    text = ui._quality_text(status)
    assert text == "Quality (diffusion steps): 6 (min 2, max 18)"


def test_quality_text_is_clear_when_pending():
    ui = make_ui()
    status = StatusInfo(quality=0, quality_min=2, quality_max=18, gen_count=0)
    text = ui._quality_text(status)
    assert text == "Quality (diffusion steps): pending (min 2, max 18)"


def test_instances_text_is_clear():
    ui = make_ui()
    status = StatusInfo(quality=0, quality_min=2, quality_max=18, gen_count=7)
    text = ui._instances_text(status)
    assert text == "Generations: 7"