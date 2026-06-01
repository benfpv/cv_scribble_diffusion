"""Unit tests for LatentDecoder (TAESD decode + resize) and RevealCompositor."""

import numpy as np
import torch

from cv_scribble_diffusion.ui.latent_decoder import LatentDecoder
from cv_scribble_diffusion.ui.reveal_compositor import RevealCompositor
from cv_scribble_diffusion.config import AppConfig, RevealConfig, UIConfig
from conftest import DummyTAESD


def _cfg() -> AppConfig:
    return AppConfig(
        ui=UIConfig(image_size=(16, 16), present_size=(16, 16), interp_fps=30),
        reveal=RevealConfig(
            reveal_mode=1,
            reveal_start_threshold=0.0,
            reveal_ease_power=2.0,
            reveal_edge_width=0.16,
        ),
    )


# -- LatentDecoder -----------------------------------------------------------

def test_decoder_full_frame_returns_present_size_bgr_f32():
    decoder = LatentDecoder(_cfg(), DummyTAESD(), torch.device("cpu"))
    latents = torch.full((1, 3, 8, 8), 0.5, dtype=torch.float32)

    frame = decoder.decode_to_frame_f32(latents, crop_region=None)

    assert frame.shape == (16, 16, 3)
    assert frame.dtype == np.float32
    # DummyTAESD passes latents through (0.5 -> 127.5 after *255).
    assert 120.0 <= float(frame.mean()) <= 135.0


def test_decoder_crop_region_returns_crop_sized_frame():
    decoder = LatentDecoder(_cfg(), DummyTAESD(), torch.device("cpu"))
    latents = torch.full((1, 3, 8, 8), 0.5, dtype=torch.float32)

    frame = decoder.decode_to_frame_f32(latents, crop_region=(4, 4, 12, 12))

    # display_scale is 1.0 for equal image/present size, so crop maps 1:1.
    assert frame.shape == (8, 8, 3)


def test_decoded_to_frame_f32_resizes_uint8_rgb_to_present_size():
    decoder = LatentDecoder(_cfg(), DummyTAESD(), torch.device("cpu"))
    decoded = np.full((8, 8, 3), 100, dtype=np.uint8)

    frame = decoder.decoded_to_frame_f32(decoded, crop_region=None)

    assert frame.shape == (16, 16, 3)
    assert frame.dtype == np.float32


# -- RevealCompositor --------------------------------------------------------

def test_compositor_no_dist_map_alpha_zero_keeps_previous_frame():
    comp = RevealCompositor(_cfg())
    prev = np.zeros((16, 16, 3), dtype=np.uint8)
    source = np.full((16, 16, 3), 100, dtype=np.float32)

    out = comp.compose_reveal(source, alpha=0.0, dist_map=None,
                              crop_region=None, prev_img=prev)

    assert np.array_equal(out, prev)


def test_compositor_no_dist_map_with_crop_blends_only_crop_region():
    comp = RevealCompositor(_cfg())
    prev = np.zeros((16, 16, 3), dtype=np.uint8)
    source_crop = np.full((8, 8, 3), 120, dtype=np.float32)

    out = comp.compose_reveal(source_crop, alpha=0.5, dist_map=None,
                              crop_region=(4, 4, 12, 12), prev_img=prev)

    assert int(out[1, 1].mean()) == 0
    assert int(out[6, 6].mean()) > 0


def test_compositor_blend_preview_interpolates_and_handles_shape_mismatch():
    comp = RevealCompositor(_cfg())
    start = np.zeros((4, 4, 3), dtype=np.float32)
    end = np.full((4, 4, 3), 10.0, dtype=np.float32)

    mid = comp.blend_preview(start, end, 0.5)
    assert np.allclose(mid, 5.0)

    # Shape mismatch falls back to the end frame.
    assert comp.blend_preview(np.zeros((2, 2, 3), dtype=np.float32), end, 0.5) is end


def test_compositor_present_crop_bounds_maps_with_display_scale():
    comp = RevealCompositor(_cfg())
    assert comp.present_crop_bounds((4, 4, 12, 12)) == (4, 4, 12, 12)
