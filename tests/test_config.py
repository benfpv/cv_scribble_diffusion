"""Tests for centralised configuration dataclasses."""

from cv_scribble_diffusion.config import (
    AppConfig, ModelConfig, InferenceConfig, RevealConfig,
    UIConfig, DebugConfig, LoggingConfig,
)


# -- AppConfig post_init defaults -------------------------------------------

def test_app_config_defaults_all_sub_configs():
    cfg = AppConfig()
    assert isinstance(cfg.model, ModelConfig)
    assert isinstance(cfg.inference, InferenceConfig)
    assert isinstance(cfg.reveal, RevealConfig)
    assert isinstance(cfg.ui, UIConfig)
    assert isinstance(cfg.debug, DebugConfig)
    assert isinstance(cfg.logging, LoggingConfig)


def test_app_config_preserves_explicit_sub_configs():
    custom_ui = UIConfig(image_size=(256, 256))
    cfg = AppConfig(ui=custom_ui)
    assert cfg.ui.image_size == (256, 256)
    # Other sub-configs still get defaults
    assert isinstance(cfg.inference, InferenceConfig)


# -- RevealConfig.reveal_edge -----------------------------------------------

def test_reveal_edge_mode_1():
    cfg = RevealConfig(reveal_mode=1, reveal_edge_width=0.16)
    assert cfg.reveal_edge == 0.08  # hard-coded for mode 1


def test_reveal_edge_mode_other():
    cfg = RevealConfig(reveal_mode=3, reveal_edge_width=0.20)
    assert cfg.reveal_edge == 0.20  # uses reveal_edge_width


# -- UIConfig computed properties -------------------------------------------

def test_display_scale_identity():
    cfg = UIConfig(image_size=(512, 512), present_size=(512, 512))
    sx, sy = cfg.display_scale
    assert sx == 1.0
    assert sy == 1.0


def test_display_scale_2x():
    cfg = UIConfig(image_size=(64, 64), present_size=(128, 128))
    sx, sy = cfg.display_scale
    assert sx == 2.0
    assert sy == 2.0


def test_display_scale_non_uniform():
    cfg = UIConfig(image_size=(100, 200), present_size=(200, 400))
    sx, sy = cfg.display_scale
    assert sx == 2.0
    assert sy == 2.0


def test_window_size_with_toolbar():
    cfg = UIConfig(
        present_size=(512, 512), show_toolbar=True,
        toolbar_height=28, canvas_margin=12,
        progress_bar_height=6, status_bar_height=16,
    )
    w, h = cfg.window_size
    assert w == 32 + 12 + 512 + 12  # 568
    assert h == 28 + 12 + 512 + 12 + 6 + 16 + 8 + 32  # 626


def test_window_size_without_toolbar():
    cfg = UIConfig(
        present_size=(512, 512), show_toolbar=False,
        toolbar_height=28, canvas_margin=12,
        progress_bar_height=6, status_bar_height=16,
    )
    w, h = cfg.window_size
    assert w == 568
    assert h == 0 + 12 + 512 + 12 + 6 + 16 + 8 + 32  # 598


def test_window_size_with_title_rail_disabled():
    cfg = UIConfig(
        present_size=(512, 512), show_toolbar=True, show_title_rail=False,
        toolbar_height=28, canvas_margin=12,
        progress_bar_height=6, status_bar_height=16,
    )
    w, h = cfg.window_size
    assert w == 12 + 512 + 12  # 536
    assert h == 28 + 12 + 512 + 12 + 6 + 16 + 8 + 32  # 626


def test_window_size_with_prompt_box_disabled():
    cfg = UIConfig(
        present_size=(512, 512), show_toolbar=True, show_prompt_box=False,
        toolbar_height=28, canvas_margin=12,
        progress_bar_height=6, status_bar_height=16,
    )
    w, h = cfg.window_size
    assert w == 32 + 12 + 512 + 12  # 568
    assert h == 28 + 12 + 512 + 12 + 6 + 16  # 586


def test_ui_config_title_defaults():
    cfg = UIConfig()
    assert cfg.app_title == "Scribble Diffusion"
    assert cfg.title_mark == "SCR"
    assert cfg.show_title_rail is True
    assert cfg.title_rail_width == 32
    assert cfg.toolbar_height == 40
    assert cfg.window_name == "Scribble Diffusion"
    assert cfg.show_prompt_box is True
    assert cfg.prompt_box_height == 32
    assert cfg.prompt_box_gap == 8
    assert cfg.prompt_max_chars == 160
    assert cfg.min_brush_thickness == 1
    assert cfg.max_brush_thickness == 20
    assert cfg.borderless_window is True
    assert cfg.window_size == (568, 638)


def test_ui_config_clamps_brush_thickness_to_configured_bounds():
    cfg = UIConfig(min_brush_thickness=3, max_brush_thickness=12)
    assert cfg.clamp_brush_thickness(-20) == 3
    assert cfg.clamp_brush_thickness(8) == 8
    assert cfg.clamp_brush_thickness(99) == 12


def test_ui_config_brush_clamp_handles_inverted_bounds():
    cfg = UIConfig(min_brush_thickness=14, max_brush_thickness=6)
    assert cfg.clamp_brush_thickness(1) == 14
    assert cfg.clamp_brush_thickness(99) == 14


def test_ui_config_brush_geometry_helpers_match_effective_point_size():
    cfg = UIConfig(brush_stroke_multiplier=1.4)
    assert cfg.brush_stroke_thickness_for(1) == 1
    assert cfg.brush_point_radius_for(1) == 0
    assert cfg.brush_stroke_thickness_for(10) == 14
    assert cfg.brush_point_radius_for(10) == 7
    assert cfg.brush_point_radius_for(0) == 0


def test_ui_config_app_title_sets_default_window_name():
    cfg = UIConfig(app_title="Sketch Lab")
    assert cfg.window_name == "Sketch Lab"


def test_ui_config_window_name_can_override_app_title():
    cfg = UIConfig(app_title="Sketch Lab", window_name="Custom Window")
    assert cfg.window_name == "Custom Window"


# -- InferenceConfig defaults sanity ----------------------------------------

def test_inference_config_steps_ordering():
    cfg = InferenceConfig()
    assert cfg.min_inference_steps < cfg.max_inference_steps
    assert cfg.max_runtime_inference_steps == 86
    assert cfg.runtime_step_cap == 86
    assert cfg.rate_inference_steps_change > 0


def test_inference_runtime_step_cap_never_below_starting_max():
    cfg = InferenceConfig(max_inference_steps=32, max_runtime_inference_steps=12)
    assert cfg.runtime_step_cap == 32


def test_inference_config_crop_params():
    cfg = InferenceConfig()
    assert cfg.crop_pad > 0
    assert 0.0 < cfg.crop_area_threshold < 1.0
    assert cfg.crop_min_dim > 0
    assert cfg.crop_alignment > 0


def test_inference_config_image_sizes_ramp_ascending():
    cfg = InferenceConfig()
    for i in range(1, len(cfg.image_sizes_ramp)):
        prev_w, prev_h = cfg.image_sizes_ramp[i - 1]
        cur_w, cur_h = cfg.image_sizes_ramp[i]
        assert cur_w >= prev_w and cur_h >= prev_h
