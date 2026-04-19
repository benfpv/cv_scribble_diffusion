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
    assert "save" in actions
    assert "undo" in actions
    assert "brush_dec" in actions
    assert "brush_inc" in actions
    assert "steps_dec" in actions
    assert "steps_inc" in actions


def test_hit_test_fps_button():
    ui = make_ui()
    for action, bx1, by1, bx2, by2 in ui._button_rects:
        if action == "fps":
            result = ui.hit_test((bx1 + bx2) // 2, (by1 + by2) // 2)
            assert result == "fps"
            break
    else:
        pytest.fail("fps button not found in layout")


def test_hit_test_save_button():
    ui = make_ui()
    for action, bx1, by1, bx2, by2 in ui._button_rects:
        if action == "save":
            result = ui.hit_test((bx1 + bx2) // 2, (by1 + by2) // 2)
            assert result == "save"
            break
    else:
        pytest.fail("save button not found in layout")


def test_hit_test_undo_button():
    ui = make_ui()
    for action, bx1, by1, bx2, by2 in ui._button_rects:
        if action == "undo":
            result = ui.hit_test((bx1 + bx2) // 2, (by1 + by2) // 2)
            assert result == "undo"
            break
    else:
        pytest.fail("undo button not found in layout")


def test_hit_test_brush_buttons():
    ui = make_ui()
    seen = set()
    for action, bx1, by1, bx2, by2 in ui._button_rects:
        if action in {"brush_dec", "brush_inc"}:
            result = ui.hit_test((bx1 + bx2) // 2, (by1 + by2) // 2)
            assert result == action
            seen.add(action)
    assert seen == {"brush_dec", "brush_inc"}


def test_hit_test_step_buttons():
    ui = make_ui()
    seen = set()
    for action, bx1, by1, bx2, by2 in ui._button_rects:
        if action in {"steps_dec", "steps_inc"}:
            result = ui.hit_test((bx1 + bx2) // 2, (by1 + by2) // 2)
            assert result == action
            seen.add(action)
    assert seen == {"steps_dec", "steps_inc"}


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


def test_compose_frame_shape_with_toolbar_buttons():
    """Window shape must remain correct with the expanded toolbar."""
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


def test_toolbar_group_spacing_is_larger_between_clusters():
    ui = make_ui()
    rects = {action: (x1, x2) for action, x1, _y1, x2, _y2 in ui._button_rects}

    normal_gap = rects["save"][0] - rects["reset"][1]
    group_gap_one = rects["mask"][0] - rects["undo"][1]
    group_gap_two = rects["steps_dec"][0] - rects["brush_inc"][1]

    assert group_gap_one > normal_gap
    assert group_gap_two > normal_gap


# -- status bar ---------------------------------------------------------------

def test_compose_frame_with_status_info():
    """StatusInfo is accepted and does not alter the frame shape."""
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)
    status = StatusInfo(quality=6, quality_min=2, quality_max=18, gen_count=3, display_fps=60)

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
    status = StatusInfo(quality=6, quality_min=2, quality_max=18, gen_count=3, display_fps=60)
    text = ui._quality_text(status)
    assert text == "Quality (diffusion steps): 6 (min 2, max 18)"


def test_quality_text_is_clear_when_pending():
    ui = make_ui()
    status = StatusInfo(quality=0, quality_min=2, quality_max=18, gen_count=0, display_fps=30)
    text = ui._quality_text(status)
    assert text == "Quality (diffusion steps): pending (min 2, max 18)"


def test_instances_text_is_clear():
    ui = make_ui()
    status = StatusInfo(quality=0, quality_min=2, quality_max=18, gen_count=7, display_fps=60, brush_thickness=3)
    text = ui._instances_text(status)
    assert text == "Generations: 7 | FPS: 60 | Brush: 3"


def test_compose_frame_with_ui_notice():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)
    status = StatusInfo(
        quality=0,
        quality_min=2,
        quality_max=18,
        gen_count=0,
        display_fps=60,
        brush_thickness=3,
        ui_notice="Saved snapshot: C:/tmp/saved_image_2.png",
    )

    result = ui.compose_frame(
        canvas_frame,
        mask_active,
        mask_present,
        show_mask=False,
        has_active_strokes=False,
        generation_progress=0.0,
        is_generating=False,
        button_states={},
        status=status,
    )
    status_y = 40 + 512 + 6
    status_region = result[status_y:status_y + 16, 12:524]
    assert status_region.sum() > 0


def test_compose_frame_with_thread_error_shows_red_badge():
    """Thread error status should render a red ERR badge in the status bar."""
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)
    status = StatusInfo(
        quality=0,
        quality_min=2,
        quality_max=18,
        gen_count=0,
        display_fps=60,
        brush_thickness=3,
        thread_error="Simulated pipeline failure",
    )

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0, is_generating=False,
        button_states={},
        status=status,
    )
    status_y = 40 + 512 + 6
    # The ERR badge area (left side of status bar) should contain red pixels (BGR: 50,50,220)
    badge_region = result[status_y:status_y + 14, 14:44]
    # Red channel (BGR index 2) should dominate in the badge region
    assert badge_region[:, :, 2].max() >= 200, "ERR badge should contain red fill"


def test_thread_error_takes_precedence_over_ui_notice():
    """When both thread_error and ui_notice are set, error should be shown."""
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)
    status_with_both = StatusInfo(
        quality=0, quality_min=2, quality_max=18, gen_count=0,
        display_fps=60, brush_thickness=3,
        thread_error="error msg",
        ui_notice="notice msg",
    )
    status_notice_only = StatusInfo(
        quality=0, quality_min=2, quality_max=18, gen_count=0,
        display_fps=60, brush_thickness=3,
        ui_notice="notice msg",
    )
    result_both = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0, is_generating=False,
        button_states={}, status=status_with_both,
    )
    result_notice = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0, is_generating=False,
        button_states={}, status=status_notice_only,
    )
    status_y = 40 + 512 + 6
    badge_area_both = result_both[status_y:status_y + 14, 14:44]
    badge_area_notice = result_notice[status_y:status_y + 14, 14:44]
    # With both set, error badge should have more red than notice-only
    red_both = int(badge_area_both[:, :, 2].sum())
    red_notice = int(badge_area_notice[:, :, 2].sum())
    assert red_both > red_notice