
class Functions_Diffusion_Image():
    def run_cnet_pipe_scribble(pipe, image, prompt, num_inference_steps=8, guidance_scale=3.5, image_size=[512, 512]):
        image = pipe(prompt, image=image, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, width=image_size[0], height=image_size[1]).images[0]
        return image