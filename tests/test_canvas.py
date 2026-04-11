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
    assert np.any(c.mask > 0)


def test_commit_updates_mask_present():
    c = make_canvas()
    c.mask_active[200, 200] = 150
    c.has_active_strokes = True
    c.commit_active_to_mask()
    assert np.any(c.mask_present > 0)


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

    assert second_nonzero >= first_nonzero

    assert second_nonzero >= first_nonzero


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
    assert c.brush_stroke_thickness >= 10  # multiplier makes it >= raw value
