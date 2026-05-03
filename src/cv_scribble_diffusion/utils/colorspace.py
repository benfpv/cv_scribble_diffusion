"""Centralized RGB/BGR color-space conversions.

Every conversion in the codebase should go through these helpers so the
implicit color-layout assumption is visible at every call site.
"""

import cv2
import numpy as np


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 array to BGR (for OpenCV display/write)."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def gray_to_rgb(image: np.ndarray) -> np.ndarray:
    """Expand a single-channel grayscale array to 3-channel RGB."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def gray_to_bgr(image: np.ndarray) -> np.ndarray:
    """Expand a single-channel grayscale array to 3-channel BGR."""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
