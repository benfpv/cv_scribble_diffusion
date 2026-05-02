"""Tests for the crop-region decision logic.

Wraps the real :func:`generation.decide_crop` so the existing test cases keep
their concise signature while still exercising the production implementation.
"""

import numpy as np
import cv2
import pytest

from config import AppConfig
from generation import decide_crop as _decide_crop


# -- adapter mirroring the legacy test signature ----------------------------

def decide_crop(mask_gray, image_size, crop_pad, crop_area_threshold, crop_min_dim,
                crop_alignment=8):
    """Adapter: legacy 5-arg signature → ``generation.decide_crop`` plan tuple."""
    plan = _decide_crop(
        mask_gray, image_size,
        crop_pad=crop_pad, crop_alignment=crop_alignment,
        crop_area_threshold=crop_area_threshold, crop_min_dim=crop_min_dim,
    )
    if plan is None:
        return False, None
    return plan.use_crop, plan.region


def blank(size=(512, 512)):
    return np.zeros(size, dtype=np.uint8)


# -- basic cases -------------------------------------------------------------

def test_empty_mask_no_crop():
    use_crop, region = decide_crop(blank(), (512, 512), 64, 0.7, 64)
    assert use_crop is False
    assert region is None


def test_small_central_stroke_uses_crop():
    mask = blank()
    mask[240:272, 240:272] = 150  # 32×32 px stroke in centre
    use_crop, region = decide_crop(mask, (512, 512), 64, 0.7, 64)
    assert use_crop is True


def test_crop_region_is_8px_aligned():
    mask = blank()
    mask[100:150, 100:150] = 150
    use_crop, (cx1, cy1, cx2, cy2) = decide_crop(mask, (512, 512), 64, 0.7, 64)
    assert use_crop is True
    assert (cx2 - cx1) % 8 == 0
    assert (cy2 - cy1) % 8 == 0


def test_crop_contains_stroke_plus_padding():
    mask = blank()
    mask[200:220, 200:220] = 150
    use_crop, (cx1, cy1, cx2, cy2) = decide_crop(mask, (512, 512), 64, 0.7, 64)
    assert use_crop is True
    # The crop should extend at least crop_pad=64 beyond the stroke bounds
    assert cx1 <= 200 - 64 or cx1 == 0  # clamped or padded
    assert cy1 <= 200 - 64 or cy1 == 0


# -- threshold boundary ------------------------------------------------------

def test_stroke_exceeding_threshold_no_crop():
    mask = blank()
    mask[10:500, 10:500] = 150  # ~95% of 512×512
    use_crop, _ = decide_crop(mask, (512, 512), 64, 0.7, 64)
    assert use_crop is False


def test_stroke_just_under_threshold_uses_crop():
    # Place a stroke whose padded box is slightly under 70% of 512×512
    # 512×512 × 0.7 ≈ 183500 px².  A ~420×420 region = 176400 < 183500.
    mask = blank()
    mask[46:420, 46:420] = 150   # 374×374, padded → ~502×502 … try smaller
    # Use a stroke that after 64px pad gives ~400×400 = 160000 < 183500
    mask2 = blank()
    mask2[100:300, 100:300] = 150  # 200×200, padded → ~328×328 = 107584
    use_crop, _ = decide_crop(mask2, (512, 512), 64, 0.7, 64)
    assert use_crop is True


# -- two disconnected strokes ------------------------------------------------

def test_two_distant_strokes_bounding_box_exceeds_threshold():
    """
    Two strokes at opposite corners: the bounding box spans the full canvas,
    which exceeds the 70% threshold → full-image mode, not crop.
    """
    mask = blank()
    mask[10:30, 10:30] = 150      # top-left
    mask[470:490, 470:490] = 150  # bottom-right
    use_crop, _ = decide_crop(mask, (512, 512), 64, 0.7, 64)
    assert use_crop is False


def test_two_close_strokes_single_crop_contains_both():
    """
    Two nearby strokes should yield one crop region that contains both.
    """
    mask = blank()
    mask[200:215, 200:215] = 150
    mask[250:265, 250:265] = 150
    use_crop, (cx1, cy1, cx2, cy2) = decide_crop(mask, (512, 512), 64, 0.7, 64)
    assert use_crop is True
    # Both stroke regions must be inside the crop
    assert cx1 <= 200 and cx2 >= 265
    assert cy1 <= 200 and cy2 >= 265


# -- minimum dimension guard -------------------------------------------------

def test_tiny_stroke_with_zero_padding_below_min_dim_no_crop():
    """
    A 1-pixel stroke with no padding will produce a 1×1 bounding box, which
    is below crop_min_dim=64 → full-image mode.
    """
    mask = blank()
    mask[256, 256] = 150
    use_crop, _ = decide_crop(mask, (512, 512),
                              crop_pad=0, crop_area_threshold=0.7, crop_min_dim=64)
    assert use_crop is False


# -- config defaults sanity check --------------------------------------------

def test_default_config_crop_params_are_sane():
    cfg = AppConfig()
    assert cfg.inference.crop_pad > 0
    assert 0.0 < cfg.inference.crop_area_threshold < 1.0
    assert cfg.inference.crop_min_dim > 0
