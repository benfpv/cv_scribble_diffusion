"""Tests for colorspace conversion helpers."""

import numpy as np

from colorspace import rgb_to_bgr, gray_to_rgb, gray_to_bgr


def test_rgb_to_bgr_swaps_channels():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[:, :] = (100, 0, 200)  # R=100, G=0, B=200
    out = rgb_to_bgr(img)
    assert out.shape == (4, 4, 3)
    assert tuple(out[0, 0]) == (200, 0, 100)  # BGR


def test_rgb_to_bgr_roundtrip():
    img = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
    assert np.array_equal(rgb_to_bgr(rgb_to_bgr(img)), img)


def test_gray_to_rgb_expands_channels():
    gray = np.full((4, 4), 128, dtype=np.uint8)
    out = gray_to_rgb(gray)
    assert out.shape == (4, 4, 3)
    assert tuple(out[0, 0]) == (128, 128, 128)


def test_gray_to_rgb_preserves_zero():
    gray = np.zeros((4, 4), dtype=np.uint8)
    out = gray_to_rgb(gray)
    assert np.all(out == 0)


def test_gray_to_bgr_expands_channels():
    gray = np.full((4, 4), 200, dtype=np.uint8)
    out = gray_to_bgr(gray)
    assert out.shape == (4, 4, 3)
    assert tuple(out[0, 0]) == (200, 200, 200)


def test_gray_to_bgr_dtype():
    gray = np.zeros((2, 2), dtype=np.uint8)
    out = gray_to_bgr(gray)
    assert out.dtype == np.uint8
