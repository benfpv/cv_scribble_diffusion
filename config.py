"""Centralised application configuration."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ModelConfig:
    """Paths and identifiers for diffusion models."""
    pipe_path: str = "stable-diffusion-v1-5"
    scribble_path: str = "sd-controlnet-scribble"
    taesd_id: str = "madebyollin/taesd"
    use_gpu: bool = True


@dataclass
class InferenceConfig:
    """Controls that affect how the generation adheres to drawings."""
    prompt: str = "colourful beautiful anime manga"
    guidance_scale: float = 3.5
    controlnet_conditioning_scale: float = 1.0
    mask_dilate: int = 48
    crop_pad: int = 64
    crop_alignment: int = 8
    crop_feather_px: int = 32
    crop_area_threshold: float = 0.7
    crop_min_dim: int = 64
    min_inference_steps: int = 2
    max_inference_steps: int = 18
    rate_inference_steps_change: int = 2
    image_sizes_ramp: Tuple[Tuple[int, int], ...] = (
        (240, 240), (280, 280), (320, 320), (360, 360),
        (400, 400), (440, 440), (480, 480), (512, 512),
    )


@dataclass
class RevealConfig:
    """Controls for the wavefront reveal animation and noise."""
    reveal_mode: int = 3
    stochastic_noise_strength: float = 0.35
    reveal_start_threshold: float = 0.3
    reveal_white_steps: float = 0.25
    reveal_outro_alpha: float = 0.35
    reveal_outro_duration_ratio: float = 0.5
    reveal_outro_min_duration: float = 1.0
    reveal_outro_max_duration: float = 3.0
    reveal_ease_power: float = 2.0
    reveal_edge_width: float = 0.16
    latent_interp_smooth: float = 0.55
    step_phase_min_duration: float = 0.08
    step_phase_max_duration: float = 0.75
    refinement_crossfade_min_duration: float = 0.9
    refinement_crossfade_max_duration: float = 1.6

    @property
    def reveal_edge(self) -> float:
        """Edge width for the reveal wavefront (varies by mode)."""
        if self.reveal_mode == 1:
            return 0.08
        return self.reveal_edge_width


@dataclass
class UIConfig:
    """Brush, window, and layout settings."""
    image_size: Tuple[int, int] = (512, 512)
    present_size: Tuple[int, int] = (512, 512)
    brush_thickness: int = 2
    brush_stroke_multiplier: float = 1.4
    max_brush_thickness: int = 20
    show_toolbar: bool = True
    toolbar_height: int = 28
    progress_bar_height: int = 6
    canvas_margin: int = 12
    status_bar_height: int = 16
    display_fps_options: Tuple[int, ...] = (
        15, 24, 30, 60, 75, 90, 100, 120, 144, 165, 180, 200, 240,
    )
    display_fps_default: int = 60
    interp_fps: int = 30
    image_store_limit_count: int = 9
    window_name: str = "ai_paint_diffusion"

    @property
    def display_scale(self) -> Tuple[float, float]:
        """(sx, sy) scale from image_size to present_size."""
        return (
            self.present_size[0] / self.image_size[0],
            self.present_size[1] / self.image_size[1],
        )

    @property
    def window_size(self) -> Tuple[int, int]:
        """(w, h) of the full OpenCV window: toolbar + margins + canvas + footer."""
        tb = self.toolbar_height if self.show_toolbar else 0
        footer = self.progress_bar_height + self.status_bar_height
        w = self.canvas_margin * 2 + self.present_size[0]
        h = tb + self.canvas_margin + self.present_size[1] + self.canvas_margin + footer
        return (w, h)


@dataclass
class DebugConfig:
    """Controls optional debug image output."""
    enabled: bool = False
    dir: str = "debug_output"


@dataclass
class LoggingConfig:
    """Controls structured runtime logging output."""
    enabled: bool = True
    dir: str = "logs"
    file_name: str = "app.log"
    level: str = "DEBUG"
    console_level: str = "INFO"
    max_bytes: int = 1_048_576
    backup_count: int = 5


@dataclass
class AppConfig:
    """Top-level configuration composed of grouped sub-configs."""
    model: ModelConfig = None      # type: ignore[assignment]
    inference: InferenceConfig = None  # type: ignore[assignment]
    reveal: RevealConfig = None    # type: ignore[assignment]
    ui: UIConfig = None            # type: ignore[assignment]
    debug: DebugConfig = None      # type: ignore[assignment]
    logging: LoggingConfig = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.reveal is None:
            self.reveal = RevealConfig()
        if self.ui is None:
            self.ui = UIConfig()
        if self.debug is None:
            self.debug = DebugConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
