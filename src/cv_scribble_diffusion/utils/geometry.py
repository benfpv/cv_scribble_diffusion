"""Pure geometry helpers shared across UI and generation layers."""

from typing import Tuple


Region = Tuple[int, int, int, int]
ScaledBounds = Tuple[int, int, int, int]


def present_bounds(region: Region, display_scale: Tuple[float, float]) -> ScaledBounds:
    """Map an image-space rect to present-space pixel bounds.

    ``display_scale`` is the ``(sx, sy)`` factor from image-space to
    present-space (see :pyattr:`UIConfig.display_scale`).
    """
    cx1, cy1, cx2, cy2 = region
    sx, sy = display_scale
    return int(cx1 * sx), int(cy1 * sy), int(cx2 * sx), int(cy2 * sy)
