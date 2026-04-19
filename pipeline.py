"""Model loading and inference for ControlNet-guided Stable Diffusion inpainting."""

import time
import torch
from PIL import Image
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
    AutoencoderTiny,
)

from config import AppConfig
from runtime_logging import get_logger


logger = get_logger(__name__)


class DiffusionPipeline:
    """Loads SD v1.5 + scribble ControlNet + TAESD and exposes a simple inference API."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        # Detect GPU
        if torch.cuda.is_available() and cfg.model.use_gpu:
            logger.info("CUDA is available")
            logger.info("GPU Name: %s", torch.cuda.get_device_name(0))
            torch.cuda.empty_cache()
            self._use_gpu = True
        else:
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available; running on CPU")
            self._use_gpu = False
        logger.info("Use GPU: %s", self._use_gpu)

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

        # The bundled SD safety checker is disabled for this local-use creative
        # tool: latency matters and false positives during interactive scribble
        # sessions interrupt the experience. Reinstate it before any deployment
        # to a public-facing surface.
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
        logger.info("Diffusion pipeline initialized")

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
        start_t = time.time()
        logger.info(
            "Starting inpaint: steps=%s size=%sx%s",
            num_inference_steps,
            width,
            height,
        )

        def on_step_end(pipe, step_index, timestep, callback_kwargs):
            latents = callback_kwargs.get("latents")
            if step_callback is not None and latents is not None:
                step_callback(step_index, num_inference_steps, latents)
            return callback_kwargs

        kwargs = dict(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=self.cfg.inference.guidance_scale,
            controlnet_conditioning_scale=self.cfg.inference.controlnet_conditioning_scale,
            width=width,
            height=height,
        )

        if step_callback is not None:
            kwargs["callback_on_step_end"] = on_step_end
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        try:
            result = self._pipe(**kwargs).images[0]
        except TypeError as exc:
            if step_callback is None:
                raise
            logger.warning(
                "callback_on_step_end unsupported (%s); falling back to deprecated callback API",
                exc,
            )
            kwargs.pop("callback_on_step_end", None)
            kwargs.pop("callback_on_step_end_tensor_inputs", None)
            kwargs["callback"] = step_callback
            kwargs["callback_steps"] = 1
            result = self._pipe(**kwargs).images[0]
        logger.info("Finished inpaint in %.3fs", time.time() - start_t)
        return result
