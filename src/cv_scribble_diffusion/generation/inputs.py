"""Pure-function helpers for the diffusion generation cycle.

Extracted from ``App.async_diffusion`` to keep the gen thread thin and to make
the crop / mask / dist-map decisions independently testable. No threading,
no model loading, no global state.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from cv_scribble_diffusion.utils.colorspace import gray_to_rgb
from cv_scribble_diffusion.config import InferenceConfig, UIConfig


CropRegion = Tuple[int, int, int, int]
PresentBounds = Tuple[int, int, int, int]


@dataclass(frozen=True)
class CropPlan:
    """Outcome of the crop decision for one generation cycle.

    ``use_crop`` is False when the bounding box is too large or too small;
    in that case ``region`` is the full-image rect and the caller should run
    the resized full-frame branch instead of the crop branch.
    """

    use_crop: bool
    region: CropRegion  # (x1, y1, x2, y2) in image-space pixels


def decide_crop(
    mask_gray: np.ndarray,
    image_size: Tuple[int, int],
    crop_pad: int,
    crop_alignment: int,
    crop_area_threshold: float,
    crop_min_dim: int,
) -> Optional[CropPlan]:
    """Decide whether to inpaint a crop region or the full frame.

    Returns ``None`` when the mask is entirely empty (no work to do).
    Returns a ``CropPlan`` otherwise; ``use_crop`` indicates whether the
    crop is small enough and large enough to be worth running.
    """
    nonzero = cv2.findNonZero(mask_gray)
    if nonzero is None:
        return None

    bx, by, bw, bh = cv2.boundingRect(nonzero)
    rx1 = max(0, bx - crop_pad)
    ry1 = max(0, by - crop_pad)
    rx2 = min(image_size[0], bx + bw + crop_pad)
    ry2 = min(image_size[1], by + bh + crop_pad)
    rw = (rx2 - rx1) // crop_alignment * crop_alignment
    rh = (ry2 - ry1) // crop_alignment * crop_alignment
    rx2 = rx1 + rw
    ry2 = ry1 + rh

    full_area = image_size[0] * image_size[1]
    use_crop = (
        rw * rh < int(crop_area_threshold * full_area)
        and rw >= crop_min_dim
        and rh >= crop_min_dim
    )
    if use_crop:
        return CropPlan(use_crop=True, region=(rx1, ry1, rx2, ry2))
    return CropPlan(use_crop=False, region=(0, 0, image_size[0], image_size[1]))


def make_dilation_kernel(mask_dilate: int) -> np.ndarray:
    """Elliptical structuring element used to dilate the stroke mask."""
    dil = mask_dilate * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))


def make_control_image(mask_gray: np.ndarray) -> Image.Image:
    """Build the binary scribble control image consumed by ControlNet."""
    control_np = np.where(mask_gray > 0, np.uint8(255), np.uint8(0))
    return Image.fromarray(gray_to_rgb(control_np))


def present_bounds(region: CropRegion, ui: UIConfig) -> PresentBounds:
    """Map an image-space crop to present-space pixel bounds."""
    cx1, cy1, cx2, cy2 = region
    sx, sy = ui.display_scale
    return int(cx1 * sx), int(cy1 * sy), int(cx2 * sx), int(cy2 * sy)


@dataclass(frozen=True)
class DistMapInputs:
    """Inputs needed to build a per-cycle reveal distance map.

    ``None`` from :func:`compute_dist_map_inputs` means there are no newly added
    strokes vs the previous generation, so the caller should skip the dist map
    and use a global crossfade reveal instead.
    """

    delta_mask: np.ndarray         # uint8 (H, W) — newly added stroke pixels
    delta_dilated: np.ndarray      # uint8 (H, W) — dilated delta mask
    out_size: Tuple[int, int]      # (W, H) target size for the dist map
    cx: int
    cy: int


def compute_dist_map_inputs(
    mask_gray: np.ndarray,
    prev_gen_mask: np.ndarray,
    plan: CropPlan,
    ui: UIConfig,
    kernel: np.ndarray,
) -> Optional[DistMapInputs]:
    """Compute the inputs for ``Animator.make_dist_map`` for this cycle.

    Returns ``None`` when there is no delta vs the previous generation, in
    which case the caller should treat this as a refinement pass.
    """
    if plan.use_crop:
        cx1, cy1, cx2, cy2 = plan.region
        cw, ch = cx2 - cx1, cy2 - cy1
        stroke_crop = mask_gray[cy1:cy2, cx1:cx2]
        delta = cv2.subtract(stroke_crop, prev_gen_mask[cy1:cy2, cx1:cx2])
        if int(cv2.countNonZero(delta)) == 0:
            return None
        px1, py1, px2, py2 = present_bounds(plan.region, ui)
        return DistMapInputs(
            delta_mask=delta,
            delta_dilated=cv2.dilate(delta, kernel),
            out_size=(px2 - px1, py2 - py1),
            cx=cw // 2,
            cy=ch // 2,
        )

    delta = cv2.subtract(mask_gray, prev_gen_mask)
    if int(cv2.countNonZero(delta)) == 0:
        return None
    h_f, w_f = mask_gray.shape
    return DistMapInputs(
        delta_mask=delta,
        delta_dilated=cv2.dilate(delta, kernel),
        out_size=ui.present_size,
        cx=w_f // 2,
        cy=h_f // 2,
    )


@dataclass
class InpaintInputs:
    """Tensor-free inputs handed to :meth:`DiffusionPipeline.run_inpaint`."""

    init_image: Image.Image
    inpaint_mask: Image.Image
    control_image: Image.Image
    width: int
    height: int


def build_inpaint_inputs(
    canvas_image: Image.Image,
    mask_gray: np.ndarray,
    control_pil: Image.Image,
    plan: CropPlan,
    kernel: np.ndarray,
    image_sizes_ramp_size: Optional[Tuple[int, int]] = None,
) -> InpaintInputs:
    """Build the per-branch (crop vs full) inpaint inputs.

    For ``plan.use_crop`` the crop is taken from canvas/control directly and
    the dilated stroke mask is preserved at native resolution.
    For the full-frame branch ``image_sizes_ramp_size`` is required: the
    canvas, control, and dilated mask are all resized to that target size.
    """
    if plan.use_crop:
        cx1, cy1, cx2, cy2 = plan.region
        cw, ch = cx2 - cx1, cy2 - cy1
        stroke_crop = mask_gray[cy1:cy2, cx1:cx2]
        dil_crop = cv2.dilate(stroke_crop, kernel)
        inpaint_mask = Image.fromarray(np.where(dil_crop > 0, np.uint8(255), np.uint8(0)))
        return InpaintInputs(
            init_image=canvas_image.crop(plan.region),
            inpaint_mask=inpaint_mask,
            control_image=control_pil.crop(plan.region),
            width=cw,
            height=ch,
        )

    if image_sizes_ramp_size is None:
        raise ValueError("image_sizes_ramp_size is required for full-frame inpaint inputs")
    dilated_full = cv2.dilate(mask_gray, kernel)
    inpaint_mask = Image.fromarray(cv2.resize(
        np.where(dilated_full > 0, np.uint8(255), np.uint8(0)),
        image_sizes_ramp_size, interpolation=cv2.INTER_NEAREST,
    ))
    return InpaintInputs(
        init_image=canvas_image.resize(image_sizes_ramp_size),
        inpaint_mask=inpaint_mask,
        control_image=control_pil.resize(image_sizes_ramp_size, Image.NEAREST),
        width=image_sizes_ramp_size[0],
        height=image_sizes_ramp_size[1],
    )
