"""Pipeline call wrappers for Stable Diffusion ControlNet-guided generation."""


class Functions_Diffusion_Image():
    """Static helpers that invoke a diffusers pipeline and return the generated PIL image."""

    def run_cnet_pipe_scribble(pipe, image, prompt, num_inference_steps=8, guidance_scale=3.5, image_size=[512, 512], step_callback=None):
        """Run a ControlNet scribble-to-image pass (no inpainting mask).

        Parameters
        ----------
        pipe : StableDiffusionControlNetPipeline
            Loaded pipeline.
        image : PIL.Image
            Scribble control image (grayscale or RGB).
        prompt : str
            Text prompt.
        num_inference_steps : int
            Number of denoising steps.
        guidance_scale : float
            Classifier-free guidance scale.
        image_size : list[int, int]
            Output [width, height] in pixels.
        step_callback : callable or None
            Called after every denoising step: callback(step, timestep, latents).

        Returns
        -------
        PIL.Image
            Generated image.
        """
        image = pipe(prompt, image=image, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=image_size[0], height=image_size[1], callback=step_callback, callback_steps=1).images[0]
        return image

    def run_cnet_pipe_inpaint(pipe, init_image, mask_image, control_image, prompt, num_inference_steps=8, guidance_scale=3.5, width=512, height=512, step_callback=None):
        """Run a ControlNet inpainting pass guided by a scribble control image.

        Parameters
        ----------
        pipe : StableDiffusionControlNetInpaintPipeline
            Loaded pipeline.
        init_image : PIL.Image
            Base image; pixels outside the mask are preserved.
        mask_image : PIL.Image
            Binary inpaint mask (white=repaint, black=keep).
        control_image : PIL.Image
            Scribble control image (must be binarised to 0/255 for reliable guidance).
        prompt : str
            Text prompt.
        num_inference_steps : int
            Number of denoising steps.
        guidance_scale : float
            Classifier-free guidance scale.
        width, height : int
            Output dimensions in pixels (must be multiples of 8).
        step_callback : callable or None
            Called after every denoising step: callback(step, timestep, latents).

        Returns
        -------
        PIL.Image
            Generated image.
        """
        image = pipe(prompt, image=init_image, mask_image=mask_image, control_image=control_image, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=width, height=height, callback=step_callback, callback_steps=1).images[0]
        return image