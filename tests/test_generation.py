"""Tests for the pure helpers in generation.py."""

import numpy as np
import pytest

from cv_scribble_diffusion.config import AppConfig, UIConfig, InferenceConfig
from cv_scribble_diffusion.generation.inputs import (
    decide_crop, make_dilation_kernel, make_control_image,
    compute_dist_map_inputs, build_inpaint_inputs, present_bounds,
    CropPlan,
)


def _blank(size=(64, 64)):
    return np.zeros(size, dtype=np.uint8)


def _ucfg():
    return UIConfig(image_size=(64, 64), present_size=(128, 128))


def _icfg():
    return InferenceConfig(
        crop_pad=4, crop_alignment=8, crop_area_threshold=0.7, crop_min_dim=16,
        mask_dilate=3, image_sizes_ramp=((64, 64),),
    )


# -- decide_crop ------------------------------------------------------------

def test_decide_crop_empty_mask_returns_none():
    assert decide_crop(_blank(), (64, 64), 4, 8, 0.7, 16) is None


def test_decide_crop_small_stroke_uses_crop():
    mask = _blank()
    mask[20:28, 20:28] = 150
    plan = decide_crop(mask, (64, 64), 4, 8, 0.7, 16)
    assert plan is not None
    assert plan.use_crop is True
    # Region must contain the stroke
    cx1, cy1, cx2, cy2 = plan.region
    assert cx1 <= 20 and cx2 >= 28
    assert cy1 <= 20 and cy2 >= 28


def test_decide_crop_large_stroke_falls_back_to_full():
    mask = _blank()
    mask[2:62, 2:62] = 150
    plan = decide_crop(mask, (64, 64), 4, 8, 0.7, 16)
    assert plan is not None
    assert plan.use_crop is False
    assert plan.region == (0, 0, 64, 64)


# -- make_control_image -----------------------------------------------------

def test_make_control_image_is_rgb_binary():
    mask = _blank()
    mask[10:20, 10:20] = 80
    img = make_control_image(mask)
    arr = np.array(img)
    assert arr.shape == (64, 64, 3)
    assert set(np.unique(arr).tolist()) <= {0, 255}


# -- make_dilation_kernel ---------------------------------------------------

def test_make_dilation_kernel_shape_is_odd():
    k = make_dilation_kernel(3)
    assert k.shape == (7, 7)


# -- present_bounds ---------------------------------------------------------

def test_present_bounds_scales_by_display_scale():
    ui = _ucfg()  # 64 -> 128 = 2x
    assert present_bounds((0, 0, 32, 16), ui) == (0, 0, 64, 32)


# -- compute_dist_map_inputs ------------------------------------------------

def test_compute_dist_map_inputs_returns_none_when_no_delta():
    mask = _blank()
    mask[20:28, 20:28] = 150
    plan = decide_crop(mask, (64, 64), 4, 8, 0.7, 16)
    kernel = make_dilation_kernel(3)
    # prev == current -> no delta
    inputs = compute_dist_map_inputs(mask, mask.copy(), plan, _ucfg(), kernel)
    assert inputs is None


def test_compute_dist_map_inputs_returns_inputs_for_delta_crop_branch():
    mask = _blank()
    mask[20:28, 20:28] = 150
    plan = decide_crop(mask, (64, 64), 4, 8, 0.7, 16)
    assert plan is not None
    assert plan.use_crop is True
    kernel = make_dilation_kernel(3)
    inputs = compute_dist_map_inputs(mask, _blank(), plan, _ucfg(), kernel)
    assert inputs is not None
    # out_size matches present-bounds size for the crop region
    px1, py1, px2, py2 = present_bounds(plan.region, _ucfg())
    assert inputs.out_size == (px2 - px1, py2 - py1)
    assert inputs.delta_mask.shape == inputs.delta_dilated.shape
    # delta_mask should have non-zero pixels (new strokes detected)
    assert np.count_nonzero(inputs.delta_mask) > 0


def test_compute_dist_map_inputs_full_branch_uses_present_size():
    mask = _blank()
    mask[2:62, 2:62] = 150
    plan = decide_crop(mask, (64, 64), 4, 8, 0.7, 16)
    assert plan is not None
    assert plan.use_crop is False
    kernel = make_dilation_kernel(3)
    inputs = compute_dist_map_inputs(mask, _blank(), plan, _ucfg(), kernel)
    assert inputs is not None
    assert inputs.out_size == _ucfg().present_size
    assert inputs.cx == 32  # w_f // 2
    assert inputs.cy == 32  # h_f // 2


# -- build_inpaint_inputs ---------------------------------------------------

def test_build_inpaint_inputs_crop_branch_sizes():
    from PIL import Image
    canvas = Image.new("RGB", (64, 64), (10, 20, 30))
    mask = _blank()
    mask[20:28, 20:28] = 150
    plan = decide_crop(mask, (64, 64), 4, 8, 0.7, 16)
    kernel = make_dilation_kernel(3)
    control = make_control_image(mask)
    out = build_inpaint_inputs(canvas, mask, control, plan, kernel)
    cx1, cy1, cx2, cy2 = plan.region
    assert (out.width, out.height) == (cx2 - cx1, cy2 - cy1)
    assert out.init_image.size == (out.width, out.height)
    assert out.inpaint_mask.size == (out.width, out.height)


def test_build_inpaint_inputs_full_branch_requires_ramp_size():
    from PIL import Image
    canvas = Image.new("RGB", (64, 64), (10, 20, 30))
    mask = _blank()
    mask[2:62, 2:62] = 150
    plan = decide_crop(mask, (64, 64), 4, 8, 0.7, 16)
    kernel = make_dilation_kernel(3)
    control = make_control_image(mask)
    with pytest.raises(ValueError):
        build_inpaint_inputs(canvas, mask, control, plan, kernel)
    out = build_inpaint_inputs(
        canvas, mask, control, plan, kernel, image_sizes_ramp_size=(48, 48),
    )
    assert (out.width, out.height) == (48, 48)
    assert out.init_image.size == (48, 48)


# -- edge cases --------------------------------------------------------------

def test_decide_crop_stroke_at_origin():
    """Stroke at (0,0) should not produce negative crop coordinates."""
    mask = _blank()
    mask[0:4, 0:4] = 150
    plan = decide_crop(mask, (64, 64), 4, 8, 0.7, 16)
    assert plan is not None
    cx1, cy1, cx2, cy2 = plan.region
    assert cx1 >= 0
    assert cy1 >= 0


def test_decide_crop_stroke_at_bottom_right():
    """Stroke at the far corner should be clamped within image bounds."""
    mask = _blank()
    mask[60:64, 60:64] = 150
    plan = decide_crop(mask, (64, 64), 4, 8, 0.7, 16)
    assert plan is not None
    cx1, cy1, cx2, cy2 = plan.region
    assert cx2 <= 64
    assert cy2 <= 64


def test_decide_crop_region_dimensions_are_aligned():
    mask = _blank()
    mask[10:20, 10:20] = 150
    plan = decide_crop(mask, (64, 64), 4, 8, 0.7, 16)
    assert plan is not None
    cx1, cy1, cx2, cy2 = plan.region
    assert (cx2 - cx1) % 8 == 0
    assert (cy2 - cy1) % 8 == 0


def test_make_dilation_kernel_dilate_zero():
    """mask_dilate=0 should produce a 1×1 kernel."""
    k = make_dilation_kernel(0)
    assert k.shape == (1, 1)


def test_make_control_image_all_zero_mask():
    mask = _blank()
    img = make_control_image(mask)
    arr = np.array(img)
    assert np.all(arr == 0)


def test_make_control_image_all_nonzero_mask():
    mask = np.full((64, 64), 1, dtype=np.uint8)
    img = make_control_image(mask)
    arr = np.array(img)
    assert np.all(arr == 255)


def test_build_inpaint_inputs_crop_preserves_mask_pixels():
    """The inpaint mask should have white pixels where strokes were dilated."""
    from PIL import Image
    canvas = Image.new("RGB", (64, 64), (0, 0, 0))
    mask = _blank()
    mask[20:28, 20:28] = 150
    plan = decide_crop(mask, (64, 64), 4, 8, 0.7, 16)
    kernel = make_dilation_kernel(3)
    control = make_control_image(mask)
    out = build_inpaint_inputs(canvas, mask, control, plan, kernel)
    mask_arr = np.array(out.inpaint_mask)
    assert set(np.unique(mask_arr).tolist()) <= {0, 255}
