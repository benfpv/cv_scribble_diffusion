"""Tests for Animator sequencing and reveal state progression."""

import numpy as np
import torch

import animator as animator_module
from animator import Animator
from config import AppConfig, RevealConfig, UIConfig
from conftest import DummyTAESD


class FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def time(self):
        return self.now


def make_animator() -> Animator:
    cfg = AppConfig(
        ui=UIConfig(image_size=(16, 16), present_size=(16, 16), interp_fps=30),
        reveal=RevealConfig(
            reveal_mode=1,
            reveal_start_threshold=0.0,
            reveal_outro_min_duration=0.2,
            reveal_outro_max_duration=0.2,
            reveal_outro_duration_ratio=0.2,
            reveal_ease_power=2.0,
            reveal_edge_width=0.16,
        ),
    )
    return Animator(cfg, DummyTAESD(), torch.device("cpu"))


def make_latents(value: float) -> torch.Tensor:
    return torch.full((1, 3, 8, 8), value, dtype=torch.float32)


def test_on_step_uses_current_phase_progress_for_alpha_continuity(monkeypatch):
    clock = FakeClock(1.0)
    monkeypatch.setattr(animator_module.time, "time", clock.time)

    animator = make_animator()
    dist_map = np.zeros((16, 16), dtype=np.float32)
    animator.prepare_generation(None, dist_map)

    animator.on_step(step=0, total_steps=4, latents_tensor=make_latents(0.2))
    first_phase = animator._phase
    assert first_phase is not None
    assert first_phase.alpha_start < first_phase.alpha_end

    clock.now = 1.5
    animator.on_step(step=1, total_steps=4, latents_tensor=make_latents(0.4))
    second_phase = animator._phase
    assert second_phase is not None
    assert second_phase.alpha_start > first_phase.alpha_start
    assert second_phase.alpha_start < second_phase.alpha_end


def test_outro_completes_to_exact_final_frame_and_signals_done(monkeypatch):
    clock = FakeClock(2.0)
    monkeypatch.setattr(animator_module.time, "time", clock.time)

    animator = make_animator()
    dist_map = np.zeros((16, 16), dtype=np.float32)
    animator.prepare_generation(None, dist_map)
    animator.on_step(step=1, total_steps=2, latents_tensor=make_latents(0.5))

    final_frame = np.full((16, 16, 3), 180, dtype=np.uint8)
    animator.start_outro(final_frame, generation_duration=1.0)
    assert animator.wait_for_outro(timeout=0.0) is False

    assert animator._phase is not None
    clock.now = animator._phase.start_time + animator._phase.duration + 0.01
    rendered = animator.get_display_frame()

    assert np.array_equal(rendered, final_frame)
    assert animator.wait_for_outro(timeout=0.0) is True
    assert np.array_equal(animator.prev_image_present, final_frame)


def test_start_outro_without_dist_map_uses_global_crossfade(monkeypatch):
    clock = FakeClock(5.0)
    monkeypatch.setattr(animator_module.time, "time", clock.time)

    animator = make_animator()
    final_frame = np.full((16, 16, 3), 90, dtype=np.uint8)
    animator.start_outro(final_frame, generation_duration=1.0)

    # Outro should now animate even when dist_map is None.
    assert animator.wait_for_outro(timeout=0.0) is False
    assert animator._phase is not None

    clock.now = animator._phase.start_time + animator._phase.duration + 0.01
    rendered = animator.get_display_frame()
    assert np.array_equal(rendered, final_frame)
    assert animator.wait_for_outro(timeout=0.0) is True


def test_on_step_without_dist_map_does_not_stage_phase(monkeypatch):
    clock = FakeClock(10.0)
    monkeypatch.setattr(animator_module.time, "time", clock.time)

    animator = make_animator()
    animator.prepare_generation(None, None)
    animator.on_step(step=0, total_steps=4, latents_tensor=make_latents(0.3))
    assert animator._phase is None


def test_compose_reveal_no_dist_map_has_no_white_preflash():
    animator = make_animator()
    prev_img = np.zeros((16, 16, 3), dtype=np.uint8)
    source = np.full((16, 16, 3), 100, dtype=np.float32)

    out = animator._compose_reveal(source, alpha=0.0, dist_map=None,
                                   crop_region=None, prev_img=prev_img)
    # No white preflash in global crossfade mode, so alpha=0 keeps previous frame.
    assert np.array_equal(out, prev_img)


def test_compose_reveal_no_dist_map_with_crop_blends_only_crop_region():
    animator = make_animator()
    prev_img = np.zeros((16, 16, 3), dtype=np.uint8)
    source_crop = np.full((8, 8, 3), 120, dtype=np.float32)
    crop_region = (4, 4, 12, 12)

    out = animator._compose_reveal(
        source_crop,
        alpha=0.5,
        dist_map=None,
        crop_region=crop_region,
        prev_img=prev_img,
    )

    # Outside the crop should remain unchanged.
    assert int(out[1, 1].mean()) == 0
    # Inside the crop should be blended (non-zero).
    assert int(out[6, 6].mean()) > 0


def test_update_prev_image_keeps_outside_pixels_unchanged():
    animator = make_animator()
    # Previous frame has dark background; new final frame is bright.
    animator.prev_image_present = np.zeros((16, 16, 3), dtype=np.uint8)
    final_frame = np.full((16, 16, 3), 200, dtype=np.uint8)

    # Center region is reveal-covered (dist=0), outside remains dist=2.
    dist_map = np.full((16, 16), 2.0, dtype=np.float32)
    dist_map[6:10, 6:10] = 0.0

    merged = animator._update_prev_image(final_frame, dist_map=dist_map, crop_region=None)
    # Center should be updated strongly to new frame.
    assert int(merged[7, 7].mean()) > 180
    # Outside should remain previous frame (near zero).
    assert int(merged[0, 0].mean()) < 5