"""Pure-function noise generators and reveal-wavefront math.

Every function here is stateless — no side effects, no thread state,
easy to test and swap.
"""

import numpy as np
import cv2


# -- reveal wavefront --------------------------------------------------------

def compute_reveal(dist_map: np.ndarray, alpha: float, edge: float) -> np.ndarray:
    """Per-pixel blend weight in [0,1] for the current denoising progress."""
    return np.clip((alpha - dist_map) / edge + 0.5, 0.0, 1.0)


def build_dist_map(
    stroke_mask: np.ndarray,
    dilated_mask: np.ndarray,
    out_size: tuple,
    cx: int,
    cy: int,
    reveal_mode: int,
    stochastic_noise_strength: float,
) -> np.ndarray:
    """Build a normalised distance map: 0=stroke, 1=inpaint boundary, 2=outside.

    Parameters
    ----------
    stroke_mask : uint8 (H, W) — raw stroke pixels (>0 = stroked)
    dilated_mask : uint8 (H, W) — dilated version (>0 = inpaint region)
    out_size : (W, H) output pixel size for cv2.resize
    cx, cy : centre for stochastic noise generation
    reveal_mode : noise style selector (1=smooth, 2=fractal, 3=cellular, 4=shards)
    stochastic_noise_strength : amplitude of perturbation
    """
    h, w = stroke_mask.shape[:2]
    dist_input = np.where(stroke_mask > 0, np.uint8(0), np.uint8(255))
    dist_raw = cv2.distanceTransform(dist_input, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dmax = float(dist_raw[dilated_mask > 0].max()) if np.any(dilated_mask > 0) else 1.0
    dist_norm = np.where(dilated_mask > 0, dist_raw / max(dmax, 1e-6), 2.0).astype(np.float32)
    if reveal_mode >= 2:
        noise = make_stochastic_noise(h, w, cx, cy, reveal_mode)
        dist_norm = np.where(
            dilated_mask > 0,
            np.clip(dist_norm + noise * stochastic_noise_strength, 0.0, 1.5),
            2.0,
        ).astype(np.float32)
    return cv2.resize(dist_norm, out_size, interpolation=cv2.INTER_LINEAR)


# -- noise generators --------------------------------------------------------

def make_stochastic_noise(h: int, w: int, cx: int, cy: int, reveal_mode: int) -> np.ndarray:
    """Dispatch to the appropriate noise generator based on reveal_mode."""
    if reveal_mode == 2:
        return make_fractal_noise(h, w)
    elif reveal_mode == 3:
        return make_cellular_noise(h, w)
    elif reveal_mode == 4:
        return make_shard_noise(h, w, cx, cy)
    return np.zeros((h, w), dtype=np.float32)


def make_cellular_noise(h: int, w: int, n_cells: int = 25) -> np.ndarray:
    """Worley/Voronoi noise in [-1, 1]."""
    pts_x = (np.random.rand(n_cells) * w).astype(np.float32)
    pts_y = (np.random.rand(n_cells) * h).astype(np.float32)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    min_dist = np.full((h, w), np.inf, dtype=np.float32)
    for i in range(n_cells):
        d = np.hypot(gx - pts_x[i], gy - pts_y[i])
        np.minimum(min_dist, d, out=min_dist)
    min_dist /= (min_dist.max() + 1e-6)
    return min_dist * 2.0 - 1.0


def make_shard_noise(h: int, w: int, cx: int, cy: int) -> np.ndarray:
    """Angular spoke noise with crisp starburst boundaries."""
    n_spokes = np.random.randint(6, 18)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    angles = np.arctan2(gy - cy, gx - cx)
    band_width = 2.0 * np.pi / n_spokes
    band_idx = ((angles + np.pi) / band_width).astype(np.int32) % n_spokes
    spoke_offsets = (np.random.rand(n_spokes).astype(np.float32) - 0.5)
    return spoke_offsets[band_idx]


def make_fractal_noise(h: int, w: int) -> np.ndarray:
    """Multi-octave fractal noise in [-1, 1]."""
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    for freq in [3, 6, 12, 24, 48]:
        fh, fw = max(2, min(h, freq)), max(2, min(w, freq))
        layer = np.random.rand(fh, fw).astype(np.float32)
        layer = cv2.resize(layer, (w, h), interpolation=cv2.INTER_LINEAR)
        noise += layer * amplitude
        amplitude *= 0.5
    noise -= noise.mean()
    std = noise.std()
    if std > 1e-6:
        noise /= std
    return noise
