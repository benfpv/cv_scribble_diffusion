"""Tests for UIOverlay layout, hit-testing, and frame composition."""

import numpy as np
import pytest

from config import AppConfig, UIConfig
from ui import UIOverlay, ButtonDef, StatusInfo, PromptInfo


def make_ui(cfg: UIConfig = None) -> UIOverlay:
    return UIOverlay(cfg or UIConfig())


def button_def(ui: UIOverlay, action_name: str) -> ButtonDef:
    for button in ui._buttons:
        if button.action == action_name:
            return button
    raise AssertionError(f"missing button: {action_name}")


# -- canvas_x_offset ----------------------------------------------------------

def test_canvas_x_offset_with_title_rail():
    ui = make_ui(UIConfig(canvas_margin=12, title_rail_width=32))
    assert ui.canvas_x_offset == 44  # rail(32) + margin(12)


def test_canvas_x_offset_without_title_rail():
    ui = make_ui(UIConfig(canvas_margin=12, show_title_rail=False))
    assert ui.canvas_x_offset == 12


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
    assert result == (56, 10)  # x: 100-(rail 32 + margin 12), y: 50-40


def test_canvas_coords_in_toolbar():
    ui = make_ui(UIConfig(toolbar_height=28, present_size=(512, 512)))
    result = ui.canvas_coords(100, 10)
    assert result is None  # y=10 is in toolbar


def test_canvas_coords_below_canvas():
    ui = make_ui(UIConfig(toolbar_height=28, present_size=(512, 512)))
    result = ui.canvas_coords(100, ui.canvas_y_offset + 512 + 5)  # below canvas
    assert result is None


def test_canvas_coords_in_margin():
    ui = make_ui(UIConfig(toolbar_height=28, present_size=(512, 512)))
    # x=36 is in the left margin after the rail (canvas starts at x=44)
    assert ui.canvas_coords(36, 50) is None


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


def test_default_toolbar_buttons_have_extra_top_padding():
    ui = make_ui()
    action, bx1, by1, bx2, by2 = ui._button_rects[0]
    assert action == "exit"
    assert ui._cfg.toolbar_height == 40
    assert by1 == 9
    assert ui._cfg.toolbar_height - by2 == 9
    assert by2 - by1 == 22
    assert ui.hit_test((bx1 + bx2) // 2, by1 - 1) is None
    assert ui.hit_test((bx1 + bx2) // 2, by2 + 1) is None


def test_exit_confirm_button_style_is_danger_and_high_contrast():
    ui = make_ui()
    exit_button = button_def(ui, "exit")

    normal_fill, normal_text = ui._button_style(exit_button, triggered=False)
    confirm_fill, confirm_text = ui._button_style(exit_button, triggered=True)

    assert confirm_fill != normal_fill
    assert confirm_fill[2] > confirm_fill[0]
    assert confirm_fill[2] > confirm_fill[1]
    assert sum(confirm_text) > sum(normal_text)


def test_exit_confirm_button_renders_danger_fill():
    ui = make_ui()
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={"exit": True},
        button_labels={"exit": "QUIT?"},
    )
    action, bx1, by1, _bx2, _by2 = ui._button_rects[0]
    assert action == "exit"
    fill_pixel = result[by1 + 2, bx1 + 2]
    assert int(fill_pixel[2]) > int(fill_pixel[0])
    assert int(fill_pixel[2]) > int(fill_pixel[1])


def test_compose_frame_canvas_notice_renders_danger_banner():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    plain = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
    )
    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
        canvas_notice="Closing safely...",
    )
    center_y = ui.canvas_y_offset + cfg.present_size[1] // 2
    banner_region = result[
        center_y - 18:center_y + 19,
        ui.canvas_x_offset:ui.canvas_x_offset + cfg.present_size[0],
    ]
    plain_region = plain[
        center_y - 18:center_y + 19,
        ui.canvas_x_offset:ui.canvas_x_offset + cfg.present_size[0],
    ]
    assert result.shape == plain.shape
    assert int(banner_region.sum()) > int(plain_region.sum())
    assert int(banner_region[:, :, 2].max()) > 160
    assert int(banner_region[:, :, 2].max()) > int(banner_region[:, :, 0].max())


def test_canvas_notice_banner_spans_canvas_width():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
        canvas_notice="Closing safely...",
    )
    center_y = ui.canvas_y_offset + cfg.present_size[1] // 2
    center_x = ui.canvas_x_offset + cfg.present_size[0] // 2
    left_pixel = result[center_y, ui.canvas_x_offset + 1]
    right_pixel = result[center_y, ui.canvas_x_offset + cfg.present_size[0] - 2]
    text_region = result[center_y - 12:center_y + 13, center_x - 120:center_x + 120]

    assert int(left_pixel[2]) > int(left_pixel[0])
    assert int(right_pixel[2]) > int(right_pixel[0])
    assert int(text_region.reshape(-1, 3).sum(axis=1).max()) > 650


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
        generation_progress=0.0,
        button_states={},
    )
    # h = 28 + 12 + 512 + 12 + 6 + 16 + 8 + 32 = 626, w = rail(32) + 12 + 512 + 12 = 568
    assert result.shape == (626, 568, 3)


def test_compose_frame_default_shape_uses_roomier_toolbar():
    ui = make_ui()
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=True, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
    )
    assert result.shape == (638, 568, 3)


def test_compose_frame_no_toolbar():
    cfg = UIConfig(present_size=(512, 512), show_toolbar=False, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
    )
    # h = 0 + 12 + 512 + 12 + 6 + 16 + 8 + 32 = 598, w = rail(32) + 12 + 512 + 12 = 568
    assert result.shape == (598, 568, 3)


def test_compose_frame_title_rail_disabled_uses_old_width():
    cfg = UIConfig(
        present_size=(512, 512), toolbar_height=28,
        progress_bar_height=6, show_title_rail=False,
    )
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
    )
    assert result.shape == (626, 536, 3)


def test_title_rail_renders_identity_mark():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, title_rail_width=32)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
    )
    rail_region = result[ui.canvas_y_offset:ui.canvas_y_offset + 90, :ui.title_rail_width - 1]
    assert len(np.unique(rail_region.reshape(-1, 3), axis=0)) > 1
    divider_pixel = result[ui.canvas_y_offset + 5, ui.title_rail_width - 1]
    rail_pixel = result[ui.canvas_y_offset + 5, 1]
    assert not np.array_equal(divider_pixel, rail_pixel)


def test_title_rail_divider_starts_below_toolbar_rule():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=40, title_rail_width=32)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
    )

    divider_x = ui.title_rail_width - 1
    toolbar_rule_y = cfg.toolbar_height - 1
    first_rail_y = cfg.toolbar_height
    rail_divider_pixel = result[first_rail_y, divider_x]

    toolbar_column = result[:toolbar_rule_y, divider_x]
    assert not np.any(np.all(toolbar_column == rail_divider_pixel, axis=1))
    assert np.array_equal(result[toolbar_rule_y, divider_x], result[toolbar_rule_y, divider_x + 1])
    assert not np.array_equal(result[first_rail_y, divider_x], result[first_rail_y, divider_x + 1])


# -- prompt box ---------------------------------------------------------------

def test_prompt_box_rect_default_layout():
    ui = make_ui(UIConfig(present_size=(512, 512), toolbar_height=28))
    assert ui.prompt_box_rect == (44, 582, 556, 614)


def test_prompt_box_rect_disabled():
    ui = make_ui(UIConfig(show_prompt_box=False))
    assert ui.prompt_box_rect is None


def test_prompt_hit_test():
    ui = make_ui(UIConfig(present_size=(512, 512), toolbar_height=28))
    x1, y1, x2, y2 = ui.prompt_box_rect
    assert ui.prompt_hit_test(x1 + 10, y1 + 10) is True
    assert ui.prompt_hit_test(x2 + 1, y1 + 10) is False


def test_prompt_cursor_index_from_click():
    ui = make_ui(UIConfig(present_size=(512, 512), toolbar_height=28))
    x1, y1, _x2, _y2 = ui.prompt_box_rect
    cursor = ui.prompt_cursor_index("abcdef", x1 + 95, cursor=6, max_chars=160)
    assert 0 <= cursor <= 6


def test_compose_frame_with_prompt_info_renders_box():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
        prompt=PromptInfo("colourful beautiful anime manga", max_chars=160),
    )
    x1, y1, x2, y2 = ui.prompt_box_rect
    region = result[y1:y2, x1:x2]
    assert len(np.unique(region.reshape(-1, 3), axis=0)) > 1


def test_compose_frame_prompt_editing_draws_active_border_and_cursor():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
        prompt=PromptInfo("neon city", editing=True, cursor=4, cursor_visible=True, max_chars=160),
    )
    x1, y1, _x2, _y2 = ui.prompt_box_rect
    assert int(result[y1, x1 + 4].sum()) > 300


def test_compose_frame_prompt_selection_draws_highlight():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)
    text = "neon city"

    plain = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
        prompt=PromptInfo(text, editing=True, cursor=4, cursor_visible=True, max_chars=160),
    )
    selected = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
        prompt=PromptInfo(
            text, editing=True, cursor=4, cursor_visible=True, max_chars=160,
            selection_start=0, selection_end=4,
        ),
    )
    x1, y1, _x2, y2 = ui.prompt_box_rect
    layout = ui._prompt_layout(ui.prompt_box_rect, 160)
    text_x = layout["text_x"]
    selected_region = selected[y1 + 6:y2 - 6, text_x:text_x + 48]
    plain_region = plain[y1 + 6:y2 - 6, text_x:text_x + 48]
    assert np.any(selected_region != plain_region)
    assert int(selected_region.sum()) > int(plain_region.sum())


def test_compose_frame_prompt_long_text_does_not_bleed_into_margin():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)
    long_prompt = "x" * 240

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
        prompt=PromptInfo(long_prompt, editing=True, cursor=160, cursor_visible=True, max_chars=160),
    )
    _x1, y1, x2, y2 = ui.prompt_box_rect
    right_margin = result[y1:y2, x2:cfg.window_size[0]]
    assert np.all(right_margin == right_margin[0, 0])


# -- progress bar fill -------------------------------------------------------

def test_progress_bar_fills_with_progress_value():
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.5,
        button_states={},
    )
    bar_row = ui.canvas_y_offset + cfg.present_size[1]
    # Left side (inside bar) should have accent colour
    left_pixel = result[bar_row, ui.canvas_x_offset + 10]
    # Right side (inside bar, past 50%) should be dark background
    right_pixel = result[bar_row, ui.canvas_x_offset + 400]
    assert int(left_pixel.sum()) > int(right_pixel.sum())
    assert int(left_pixel.sum()) < 330


def test_progress_bar_empty_when_progress_zero():
    """Bar is empty only when progress is 0 (truly idle, before first cycle)."""
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
    )
    bar_y = ui.canvas_y_offset + cfg.present_size[1]
    bar_region = result[bar_y:bar_y + cfg.progress_bar_height,
                        ui.canvas_x_offset:ui.canvas_x_offset + cfg.present_size[0]]
    # Should be uniform (all dark grey background)
    assert np.all(bar_region == bar_region[0, 0])


def test_progress_bar_stays_full_between_cycles():
    """After a cycle completes (progress=1.0), the bar must remain full until
    the next cycle starts. This prevents the one-frame flicker that occurs
    when gating fill on a transient ``is_generating`` flag."""
    cfg = UIConfig(present_size=(512, 512), toolbar_height=28, progress_bar_height=6)
    ui = make_ui(cfg)
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=1.0,
        button_states={},
    )
    bar_y = ui.canvas_y_offset + cfg.present_size[1]
    # Both ends of the bar should be filled with the accent colour
    left_pixel = result[bar_y + 2, ui.canvas_x_offset + 5]
    right_pixel = result[bar_y + 2, ui.canvas_x_offset + 500]
    assert int(left_pixel.sum()) > 100
    assert int(right_pixel.sum()) > 100
    assert int(left_pixel.sum()) < 330
    assert int(right_pixel.sum()) < 330


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
        generation_progress=0.0,
        button_states={},
    )
    assert result[ui.canvas_y_offset + 100, ui.canvas_x_offset + 100, 1] == 150


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
        generation_progress=0.0,
        button_states={},
        button_labels={"fps": "60"},
    )
    assert result.shape == (626, 568, 3)


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
        generation_progress=0.0,
        button_states={},
    )
    assert result.shape == (626, 568, 3)


def test_toolbar_group_spacing_is_larger_between_clusters():
    ui = make_ui()
    rects = {action: (x1, x2) for action, x1, _y1, x2, _y2 in ui._button_rects}

    normal_gap = rects["save"][0] - rects["reset"][1]
    group_gap_one = rects["mask"][0] - rects["undo"][1]
    group_gap_two = rects["steps_dec"][0] - rects["brush_inc"][1]

    assert group_gap_one > normal_gap
    assert group_gap_two > normal_gap


def test_toolbar_brush_preview_sits_between_thickness_buttons():
    ui = make_ui()
    rects = {action: (x1, y1, x2, y2) for action, x1, y1, x2, y2 in ui._button_rects}
    px1, py1, px2, py2 = ui.brush_preview_rect

    assert rects["brush_dec"][2] < px1 < px2 < rects["brush_inc"][0]
    assert py1 < ui._cfg.toolbar_height // 2 < py2
    assert py1 >= 5
    assert ui._cfg.toolbar_height - py2 >= 5
    assert ui.hit_test((px1 + px2) // 2, (py1 + py2) // 2) is None


def test_toolbar_brush_preview_dot_fits_inside_roomier_slot():
    ui = make_ui()
    x1, y1, x2, y2 = ui.brush_preview_rect
    radius = ui._brush_preview_radius(ui._cfg.max_brush_thickness)
    cx = (x1 + x2) // 2
    cy = ui._cfg.toolbar_height // 2

    assert y1 < cy - radius
    assert cy + radius < y2
    assert x1 < cx - radius
    assert cx + radius < x2


def test_toolbar_brush_preview_is_hidden_without_toolbar():
    ui = make_ui(UIConfig(show_toolbar=False))
    assert ui.brush_preview_rect is None


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
        generation_progress=0.5,
        button_states={},
        status=status,
    )
    assert result.shape == (626, 568, 3)
    # Status bar region should have non-zero pixels (rendered text)
    status_y = ui.canvas_y_offset + cfg.present_size[1] + cfg.progress_bar_height
    status_region = result[status_y:status_y + cfg.status_bar_height,
                           ui.canvas_x_offset:ui.canvas_x_offset + cfg.present_size[0]]
    assert status_region.sum() > 0


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


def test_brush_preview_radius_uses_actual_point_radius():
    ui = make_ui()
    assert ui._brush_preview_radius(0) == 0
    assert ui._brush_preview_radius(1) == ui._cfg.brush_point_radius_for(1)
    assert ui._brush_preview_radius(10) == ui._cfg.brush_point_radius_for(10)
    assert ui._brush_preview_radius(ui._cfg.max_brush_thickness) > ui._brush_preview_radius(10)


def test_compose_frame_draws_toolbar_brush_preview_actual_point():
    ui = make_ui()
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)
    status = StatusInfo(
        quality=0, quality_min=2, quality_max=18, gen_count=0,
        display_fps=60, brush_thickness=10,
    )
    radius = ui._brush_preview_radius(status.brush_thickness)

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
        status=status,
    )

    x1, y1, x2, y2 = ui.brush_preview_rect
    cx = (x1 + x2) // 2
    cy = ui._cfg.toolbar_height // 2
    slot_corner = result[y1 + 2, x1 + 2]
    toolbar_bg = result[1, x1 + 2]
    assert radius == 7
    assert np.array_equal(slot_corner, toolbar_bg)
    assert int(result[cy, cx].sum()) > int(slot_corner.sum())
    assert int(result[cy, cx + radius].sum()) > int(slot_corner.sum())


def test_toolbar_brush_preview_has_no_tall_box_backdrop():
    ui = make_ui()
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)
    status = StatusInfo(
        quality=0, quality_min=2, quality_max=18, gen_count=0,
        display_fps=60, brush_thickness=10,
    )

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
        status=status,
    )

    x1, y1, x2, y2 = ui.brush_preview_rect
    cx = (x1 + x2) // 2
    cy = ui._cfg.toolbar_height // 2
    preview_region = result[y1:y2, x1:x2]
    toolbar_pixel = result[1, x1 + 2]
    dot_pixel = result[cy, cx]

    assert np.array_equal(result[y1 + 1, x1 + 1], toolbar_pixel)
    assert np.array_equal(result[y2 - 2, x2 - 2], toolbar_pixel)
    assert np.any(np.all(preview_region == dot_pixel, axis=2))


def test_compose_frame_keeps_brush_preview_out_of_status_bar():
    ui = make_ui()
    canvas_frame = np.zeros((512, 512, 3), dtype=np.uint8)
    mask_active = np.zeros((512, 512), dtype=np.uint8)
    mask_present = np.zeros((512, 512, 3), dtype=np.uint8)
    status = StatusInfo(
        quality=0, quality_min=2, quality_max=18, gen_count=0,
        display_fps=60, brush_thickness=ui._cfg.max_brush_thickness,
    )

    result = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={},
        status=status,
    )

    status_y = ui.canvas_y_offset + ui._cfg.present_size[1] + ui._cfg.progress_bar_height
    status_region = result[
        status_y:status_y + ui._cfg.status_bar_height,
        ui.canvas_x_offset:ui.canvas_x_offset + ui._cfg.present_size[0],
    ]
    blue_pixels = np.count_nonzero(
        (status_region[:, :, 0] == 86) &
        (status_region[:, :, 1] == 164) &
        (status_region[:, :, 2] == 236)
    )
    assert blue_pixels == 0


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
        button_states={},
        status=status,
    )
    status_y = ui.canvas_y_offset + cfg.present_size[1] + cfg.progress_bar_height
    status_region = result[status_y:status_y + cfg.status_bar_height,
                           ui.canvas_x_offset:ui.canvas_x_offset + cfg.present_size[0]]
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
        generation_progress=0.0,
        button_states={},
        status=status,
    )
    status_y = ui.canvas_y_offset + cfg.present_size[1] + cfg.progress_bar_height
    # The ERR badge area (left side of status bar) should contain red pixels (BGR: 50,50,220)
    badge_region = result[status_y:status_y + 14, ui.canvas_x_offset + 2:ui.canvas_x_offset + 32]
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
        generation_progress=0.0,
        button_states={}, status=status_with_both,
    )
    result_notice = ui.compose_frame(
        canvas_frame, mask_active, mask_present,
        show_mask=False, has_active_strokes=False,
        generation_progress=0.0,
        button_states={}, status=status_notice_only,
    )
    status_y = ui.canvas_y_offset + cfg.present_size[1] + cfg.progress_bar_height
    badge_area_both = result_both[status_y:status_y + 14, ui.canvas_x_offset + 2:ui.canvas_x_offset + 32]
    badge_area_notice = result_notice[status_y:status_y + 14, ui.canvas_x_offset + 2:ui.canvas_x_offset + 32]
    # With both set, error badge should have more red than notice-only
    red_both = int(badge_area_both[:, :, 2].sum())
    red_notice = int(badge_area_notice[:, :, 2].sum())
    assert red_both > red_notice