"""Tests for reveal.py pure functions (no GPU / models required)."""

import numpy as np
import cv2
import pytest

from reveal import (
    compute_reveal,
    build_dist_map,
    ease_progress,
    compute_outro_duration,
    make_fractal_noise,
    make_cellular_noise,
    make_shard_noise,
)


# -- compute_reveal ----------------------------------------------------------

def test_compute_reveal_on_stroke_is_fully_revealed():
    # dist=0 means we are right on the stroke line.
    # (0.5 - 0) / 0.12 + 0.5 >> 1.0 → clamped to 1.0
    dist = np.zeros((16, 16), dtype=np.float32)
    result = compute_reveal(dist, alpha=0.5, edge=0.12)
    assert result.shape == (16, 16)
    assert np.all(result == 1.0)


def test_compute_reveal_at_boundary_midpoint():
    # dist=1.0, alpha=1.0 → (1.0-1.0)/edge + 0.5 = 0.5
    dist = np.ones((8, 8), dtype=np.float32)
    result = compute_reveal(dist, alpha=1.0, edge=0.12)
    assert np.allclose(result, 0.5)


def test_compute_reveal_outside_is_zero():
    # dist=2.0 (outside mask) with alpha=0.0 → well below 0 → clamped to 0
    dist = np.full((8, 8), 2.0, dtype=np.float32)
    result = compute_reveal(dist, alpha=0.0, edge=0.12)
    assert np.all(result == 0.0)


def test_compute_reveal_output_in_unit_range():
    rng = np.random.default_rng(0)
    dist = rng.uniform(0, 2, (32, 32)).astype(np.float32)
    result = compute_reveal(dist, alpha=0.5, edge=0.12)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


# -- noise generators --------------------------------------------------------

def test_fractal_noise_shape_and_dtype():
    noise = make_fractal_noise(64, 64)
    assert noise.shape == (64, 64)
    assert noise.dtype == np.float32


def test_fractal_noise_is_not_constant():
    noise = make_fractal_noise(64, 64)
    assert noise.std() > 0.0


def test_cellular_noise_shape_and_range():
    noise = make_cellular_noise(64, 64)
    assert noise.shape == (64, 64)
    assert noise.min() >= -1.0
    assert noise.max() <= 1.0


def test_shard_noise_shape():
    noise = make_shard_noise(64, 64, cx=32, cy=32)
    assert noise.shape == (64, 64)


def test_shard_noise_different_seeds():
    # Not seeded, so two calls should almost certainly differ
    n1 = make_shard_noise(32, 32, 16, 16)
    n2 = make_shard_noise(32, 32, 16, 16)
    # Very unlikely to be identical
    assert not np.array_equal(n1, n2)


# -- build_dist_map ----------------------------------------------------------

def _make_stroke_and_dilated(size=64, stroke_slice=(28, 36, 28, 36), ksize=5):
    stroke = np.zeros((size, size), dtype=np.uint8)
    r0, r1, c0, c1 = stroke_slice
    stroke[r0:r1, c0:c1] = 150
    kernel = np.ones((ksize, ksize), np.uint8)
    dilated = cv2.dilate(stroke, kernel)
    return stroke, dilated


def test_build_dist_map_stroke_pixels_near_zero():
    stroke, dilated = _make_stroke_and_dilated()
    dist = build_dist_map(stroke, dilated, (64, 64), 32, 32,
                          reveal_mode=1, stochastic_noise_strength=0.0)
    # Pixels inside the original stroke should be close to 0
    assert dist[32, 32] < 0.2


def test_build_dist_map_outside_pixels_are_two():
    stroke, dilated = _make_stroke_and_dilated()
    dist = build_dist_map(stroke, dilated, (64, 64), 32, 32,
                          reveal_mode=1, stochastic_noise_strength=0.0)
    # Top-left corner is well outside the dilated mask
    assert dist[0, 0] == pytest.approx(2.0)


def test_build_dist_map_output_size():
    stroke, dilated = _make_stroke_and_dilated()
    out_w, out_h = 128, 96
    dist = build_dist_map(stroke, dilated, (out_w, out_h), 32, 32,
                          reveal_mode=1, stochastic_noise_strength=0.0)
    assert dist.shape == (out_h, out_w)


def test_build_dist_map_stochastic_mode_changes_values():
    stroke, dilated = _make_stroke_and_dilated()
    d_smooth = build_dist_map(stroke, dilated, (64, 64), 32, 32,
                              reveal_mode=1, stochastic_noise_strength=0.0)
    d_noisy  = build_dist_map(stroke, dilated, (64, 64), 32, 32,
                              reveal_mode=2, stochastic_noise_strength=0.5)
    # Noisy mode should produce different values inside the mask
    assert not np.allclose(d_smooth, d_noisy)


# -- ease_progress -----------------------------------------------------------

def test_ease_progress_identity_at_zero():
    assert ease_progress(0.0, 2.5) == pytest.approx(0.0)


def test_ease_progress_identity_at_one():
    assert ease_progress(1.0, 2.5) == pytest.approx(1.0)


def test_ease_progress_linear_when_power_is_one():
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        assert ease_progress(t, 1.0) == pytest.approx(t)


def test_ease_progress_front_loads_with_high_power():
    # At t=0.3 with power=2.5, eased should be > 0.3 (front-loaded)
    eased = ease_progress(0.3, 2.5)
    assert eased > 0.3


def test_ease_progress_monotonically_increasing():
    power = 2.5
    prev = 0.0
    for i in range(1, 101):
        t = i / 100
        val = ease_progress(t, power)
        assert val >= prev
        prev = val


def test_ease_progress_clamps_out_of_range():
    assert ease_progress(-0.5, 2.0) == pytest.approx(0.0)
    # Values above 1.0 pass through unchanged (needed for full boundary reveal)
    assert ease_progress(1.5, 2.0) == pytest.approx(1.5)


# -- compute_outro_duration --------------------------------------------------

def test_outro_duration_respects_ratio():
    # 4s generation × 0.5 ratio = 2s (within min/max)
    assert compute_outro_duration(4.0, 0.5, 1.0, 3.0) == pytest.approx(2.0)


def test_outro_duration_clamps_to_min():
    assert compute_outro_duration(0.5, 0.5, 1.0, 3.0) == pytest.approx(1.0)


def test_outro_duration_clamps_to_max():
    assert compute_outro_duration(20.0, 0.5, 1.0, 3.0) == pytest.approx(3.0)


def test_outro_duration_zero_generation_returns_min():
    assert compute_outro_duration(0.0, 0.5, 1.0, 3.0) == pytest.approx(1.0)
