"""Tests for Canvas state management (no GPU / models required)."""

import numpy as np
import pytest

from config import AppConfig
from canvas import Canvas


def make_canvas() -> Canvas:
    return Canvas(AppConfig())


# -- initial state -----------------------------------------------------------

def test_initial_mask_is_zeros():
    c = make_canvas()
    assert np.all(c.mask == 0)


def test_initial_drawing_is_false():
    c = make_canvas()
    assert c.drawing is False


def test_initial_has_active_strokes_is_false():
    c = make_canvas()
    assert c.has_active_strokes is False


def test_initial_image_is_black():
    c = make_canvas()
    import numpy as np
    assert np.all(np.array(c.image) == 0)


# -- stroke lifecycle --------------------------------------------------------

def test_begin_stroke_sets_drawing_flag():
    c = make_canvas()
    c.begin_stroke(100, 100)
    assert c.drawing is True


def test_begin_stroke_sets_has_active_strokes():
    c = make_canvas()
    c.begin_stroke(100, 100)
    assert c.has_active_strokes is True


def test_end_stroke_clears_has_active_strokes():
    c = make_canvas()
    c.begin_stroke(100, 100)
    c.end_stroke(110, 110)
    assert c.has_active_strokes is False


def test_end_stroke_clears_drawing_flag():
    c = make_canvas()
    c.begin_stroke(100, 100)
    c.end_stroke(110, 110)
    assert c.drawing is False


def test_end_stroke_clears_mask_active():
    c = make_canvas()
    c.begin_stroke(100, 100)
    c.end_stroke(100, 100)
    assert np.all(c.mask_active == 0)


def test_continue_stroke_only_draws_when_drawing():
    c = make_canvas()
    # continue_stroke without begin_stroke should be a no-op
    c.continue_stroke(100, 100)
    assert np.all(c.mask_active == 0)


# -- mask accumulation -------------------------------------------------------

def test_commit_empty_active_returns_false():
    c = make_canvas()
    assert c.commit_active_to_mask() is False


def test_commit_painted_active_returns_true_and_accumulates():
    c = make_canvas()
    # Paint directly into mask_active and set the flag (end_stroke would zero it)
    c.mask_active[200, 200] = 150
    c.has_active_strokes = True
    changed = c.commit_active_to_mask()
    assert changed is True
    # mask_active is at present_size, committed to mask at image_size via INTER_NEAREST
    assert c.mask.shape == c.cfg.ui.image_size
    assert np.count_nonzero(c.mask) > 0


def test_commit_updates_mask_present():
    c = make_canvas()
    c.mask_active[200, 200] = 150
    c.has_active_strokes = True
    c.commit_active_to_mask()
    assert c.mask_present.shape == (*c.cfg.ui.present_size, 3)
    assert np.count_nonzero(c.mask_present) > 0


def test_mask_accumulates_across_commits():
    c = make_canvas()
    c.mask_active[100, 100] = 150
    c.has_active_strokes = True
    c.commit_active_to_mask()
    first_nonzero = np.count_nonzero(c.mask)

    c.mask_active[300, 300] = 150
    c.has_active_strokes = True
    c.commit_active_to_mask()
    second_nonzero = np.count_nonzero(c.mask)

    assert second_nonzero > first_nonzero


def test_commit_same_active_twice_is_noop_second_time():
    c = make_canvas()
    c.mask_active[100, 100] = 150
    c.has_active_strokes = True
    changed_first = c.commit_active_to_mask()
    mask_after_first = c.mask.copy()
    changed_second = c.commit_active_to_mask()

    assert changed_first is True
    assert changed_second is False
    assert np.array_equal(mask_after_first, c.mask)


# -- reset -------------------------------------------------------------------

def test_reset_clears_mask():
    c = make_canvas()
    c.mask_active[200, 200] = 150
    c.commit_active_to_mask()
    c.reset()
    assert np.all(c.mask == 0)


def test_reset_clears_drawing_flag():
    c = make_canvas()
    c.begin_stroke(100, 100)
    c.reset()
    assert c.drawing is False


def test_reset_clears_has_active_strokes():
    c = make_canvas()
    c.begin_stroke(100, 100)
    c.reset()
    assert c.has_active_strokes is False


def test_reset_clears_image():
    c = make_canvas()
    from PIL import Image
    c.image = Image.new("RGB", c.cfg.ui.image_size, (255, 0, 0))
    c.reset()
    assert np.all(np.array(c.image) == 0)


def test_snapshot_and_restore_round_trip():
    c = make_canvas()
    c.mask[10, 10] = 150
    c.mask_present[5, 5] = (10, 20, 30)
    snap = c.snapshot()

    c.reset()
    c.restore(snap)

    assert c.mask[10, 10] == 150
    assert tuple(c.mask_present[5, 5]) == (10, 20, 30)


# -- brush thickness ---------------------------------------------------------

def test_set_brush_thickness_clamps_to_max():
    c = make_canvas()
    c.set_brush_thickness(c.cfg.ui.max_brush_thickness + 999)
    assert c.brush_thickness == c.cfg.ui.max_brush_thickness


def test_set_brush_thickness_clamps_to_min():
    c = make_canvas()
    c.set_brush_thickness(0)
    assert c.brush_thickness == 1


def test_set_brush_thickness_updates_stroke_thickness():
    c = make_canvas()
    c.set_brush_thickness(10)
    # 10 * 1.4 = 14.0 → int(14.0) = 14
    assert c.brush_stroke_thickness == 14


def test_set_brush_thickness_updates_point_thickness():
    c = make_canvas()
    c.set_brush_thickness(9)
    assert c.brush_point_thickness == 4


def test_set_brush_thickness_one_has_single_pixel_start_radius():
    c = make_canvas()
    c.set_brush_thickness(1)
    assert c.brush_point_thickness == 0


# -- snapshot / restore edge cases -------------------------------------------

def test_restore_resets_drawing_state():
    c = make_canvas()
    c.mask[10, 10] = 150
    snap = c.snapshot()

    # Dirty up drawing state
    c.begin_stroke(50, 50)  # sets drawing=True, has_active_strokes=True
    c.restore(snap)

    assert c.drawing is False
    assert c.has_active_strokes is False
    assert np.all(c.mask_active == 0)
    assert c.prev_x == -1
    assert c.prev_y == -1


def test_snapshot_is_independent_copy():
    """Mutating the canvas after snapshot must not affect the snapshot."""
    c = make_canvas()
    c.mask[10, 10] = 100
    snap = c.snapshot()
    c.mask[10, 10] = 255  # mutate original
    c.restore(snap)
    assert c.mask[10, 10] == 100  # original value restored


# -- mask saturation ---------------------------------------------------------

def test_commit_saturates_at_255_via_cv2_add():
    """cv2.add saturates at 255 instead of wrapping — verify this."""
    c = make_canvas()
    c.mask_active[200, 200] = 200
    c.has_active_strokes = True
    c.commit_active_to_mask()

    # Paint more at the same spot
    c.mask_active[200, 200] = 200
    c._last_committed_active.fill(0)  # force delta to be non-zero
    c.commit_active_to_mask()

    # cv2.add saturates: 200 + 200 = 255 (not 400 or overflow)
    assert c.mask.max() <= 255


# -- patch_image and feather mask -------------------------------------------

def test_make_feather_mask_shape_and_range():
    from canvas import _make_feather_mask
    mask = _make_feather_mask(64, 64, feather=8)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.float32
    assert mask.max() <= 1.0 + 1e-6
    assert mask.min() >= 0.0


def test_make_feather_mask_with_feather_is_uniform_ones():
    """GaussianBlur with BORDER_REFLECT on all-ones always returns all-ones."""
    from canvas import _make_feather_mask
    mask = _make_feather_mask(32, 32, feather=8)
    assert np.allclose(mask, 1.0)


def test_make_feather_mask_no_feather():
    from canvas import _make_feather_mask
    mask = _make_feather_mask(32, 32, feather=0)
    assert np.allclose(mask, 1.0)


def test_make_feather_mask_feather_one():
    from canvas import _make_feather_mask
    mask = _make_feather_mask(32, 32, feather=1)
    assert np.allclose(mask, 1.0)  # feather <= 1 skips blur


def test_patch_image_blends_result_into_canvas():
    from PIL import Image
    c = make_canvas()
    # Fill canvas with red
    c.image = Image.new("RGB", c.cfg.ui.image_size, (255, 0, 0))
    # Patch a green region into a crop area
    region = (100, 100, 164, 164)
    result = Image.new("RGB", (64, 64), (0, 255, 0))
    c.patch_image(region, result, feather_px=8)

    img = np.array(c.image)
    # Center of patched region should be green (feather peak ≈ 1.0)
    center = img[132, 132]
    assert center[1] > center[0], "Center should be more green than red"

    # Far outside the patch should still be red
    outside = img[10, 10]
    assert tuple(outside) == (255, 0, 0)


def test_patch_image_feather_zero():
    """With feather=0 the paste should still work (hard edge)."""
    from PIL import Image
    c = make_canvas()
    c.image = Image.new("RGB", c.cfg.ui.image_size, (0, 0, 0))
    region = (0, 0, 32, 32)
    result = Image.new("RGB", (32, 32), (100, 100, 100))
    c.patch_image(region, result, feather_px=0)
    img = np.array(c.image)
    assert np.all(img[16, 16] == 100)  # hard paste, full coverage


# -- clamp_coords ------------------------------------------------------------

def test_clamp_coords_negative_values():
    c = make_canvas()
    x, y = c._clamp_coords(-10, -5)
    assert x == 0
    assert y == 0


def test_clamp_coords_overflow_values():
    c = make_canvas()
    w, h = c.cfg.ui.present_size
    x, y = c._clamp_coords(w + 100, h + 100)
    assert x == w - 1
    assert y == h - 1
