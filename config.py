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
    reveal_white_steps: float = 0.25
    reveal_outro_alpha: float = 0.35
    reveal_outro_duration: float = 1.5
    latent_interp_smooth: float = 0.55
    interp_fps: int = 30

    @property
    def reveal_edge(self) -> float:
        """Edge width for the reveal wavefront (varies by mode)."""
        return 0.08 if self.reveal_mode == 1 else 0.12


@dataclass
class UIConfig:
    """Brush, window, and pen-control settings."""
    image_size: Tuple[int, int] = (512, 512)
    present_size: Tuple[int, int] = (512, 512)
    brush_thickness: int = 2
    brush_stroke_multiplier: float = 1.4
    max_brush_thickness: int = 20
    pen_controls_active: bool = True
    image_store_limit_count: int = 9
    window_name: str = "ai_paint_diffusion"

    @property
    def display_scale(self) -> Tuple[float, float]:
        """(sx, sy) scale from image_size to present_size."""
        return (
            self.present_size[0] / self.image_size[0],
            self.present_size[1] / self.image_size[1],
        )


@dataclass
class AppConfig:
    """Top-level configuration composed of grouped sub-configs."""
    model: ModelConfig = None      # type: ignore[assignment]
    inference: InferenceConfig = None  # type: ignore[assignment]
    reveal: RevealConfig = None    # type: ignore[assignment]
    ui: UIConfig = None            # type: ignore[assignment]

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.inference is None:
            self.inference = InferenceConfig()
        if self.reveal is None:
            self.reveal = RevealConfig()
        if self.ui is None:
            self.ui = UIConfig()
