"""Model loading and inference for ControlNet-guided Stable Diffusion inpainting."""

import numpy as np
import torch
from PIL import Image
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    AutoencoderTiny,
)

from config import AppConfig


class DiffusionPipeline:
    """Loads SD v1.5 + scribble ControlNet + TAESD and exposes a simple inference API."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        # Detect GPU
        if torch.cuda.is_available() and cfg.model.use_gpu:
            print("CUDA is available.")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
            self._use_gpu = True
        else:
            if not torch.cuda.is_available():
                print("CUDA is not available. Running on CPU.")
            self._use_gpu = False
        print(f"Use GPU?: {self._use_gpu}")

        # ControlNet
        controlnet = ControlNetModel.from_pretrained(
            cfg.model.scribble_path, torch_dtype=torch.float16
        )

        # SD inpainting pipeline
        self._pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            cfg.model.pipe_path, controlnet=controlnet, torch_dtype=torch.float16
        )
        if self._use_gpu:
            self._pipe.enable_sequential_cpu_offload()

        self._pipe.safety_checker = None
        self._pipe.scheduler = UniPCMultistepScheduler.from_config(
            self._pipe.scheduler.config
        )
        self._pipe.enable_vae_slicing()
        self._pipe.enable_attention_slicing()

        # TAESD for fast preview decoding (~10× faster than full VAE)
        self._taesd = AutoencoderTiny.from_pretrained(
            cfg.model.taesd_id, torch_dtype=torch.float16
        )
        if self._use_gpu:
            self._taesd.to("cuda")
        self._taesd.eval()

    # -- public API -----------------------------------------------------------

    @property
    def taesd(self):
        """The AutoencoderTiny model (for external consumers like Animator)."""
        return self._taesd

    @property
    def taesd_device(self):
        """Device the TAESD model lives on."""
        return next(self._taesd.parameters()).device

    def run_inpaint(
        self,
        init_image: Image.Image,
        mask_image: Image.Image,
        control_image: Image.Image,
        prompt: str,
        num_inference_steps: int,
        width: int,
        height: int,
        step_callback=None,
    ) -> Image.Image:
        """Run ControlNet-guided inpainting and return the generated PIL image."""
        return self._pipe(
            prompt,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=self.cfg.inference.guidance_scale,
            controlnet_conditioning_scale=self.cfg.inference.controlnet_conditioning_scale,
            width=width,
            height=height,
            callback=step_callback,
            callback_steps=1,
        ).images[0]
