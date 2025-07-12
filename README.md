# cv_scribble_diffusion
cv_scribble_diffusion opens a drawing window using OpenCV(2), allowing you to sketch with your mouse. Your scribble serves as a mask input for image generation (e.g., SD1.5), which runs asynchronously to generate an image. The result appears in real time as a background behind your drawing, ideally creating a continuous loop of visual inspiration.

![cv_scribble_diffusion_demo](https://github.com/user-attachments/assets/a32a683a-5139-4949-90e5-e7ec20cef680)

This is partly inspired by already existing similar projects, for example (not necessarily all inclusive):
- krita-ai-diffusion (https://github.com/Acly/krita-ai-diffusion/releases/tag/v1.9.0)

Models used:
- stable-diffusion-v1-5
- sd-controlnet-scribble

# Repository Contents
1. main.py
2. functions_diffusion_image_s.py
3. LICENSE
4. README.md

# Requirements
1. Python 3.12.10 (may also work on other versions)
2. Python packages: numpy, cv2 (opencv-python), pillow, diffusers, torch
3. stable-diffusion-v1-5 (this is an image generation model)
4. sd-controlnet-scribble (this is a 'scribble->image' controlnet for the stable-diffusion-v1-5 image generation model)
- Note: If you wish to use a different configuration (e.g., stable-diffusion-v1-5 only, other image generation models with/without other controlnets), you will likely need to update 'main.py' to use the 'diffusers' library appropriately for your specific use case.

# Instructions to Run
1. In 'main.py', set 'local_pipe_path' to the location of your 'stable-diffusion-v1-5'.
2. In 'main.py', set 'local_scribble_path' to the location of your 'sd-controlnet-scribble'.
3. (Optional) In 'main.py', set parameters (e.g., prompt) to your preferences, under 'Main.__init__'.
4. Run 'main.py' with Python (e.g., 'py main.py' in command prompt/terminal).
5. Expected outcome: May take a few seconds/minutes to load the image generation model, especially the first time. Then a window is launched. Finally, you may draw on the window and thereafter, images will be continuously generated and presented on the background.
6. Exit by pressing 'Esc' on the window, or possibly 'Ctrl+C' a few times in the command prompt/terminal.

# Keyboard Inputs
- Spacebar: Reset the canvas to blank
- Enter: Save the current image (without paint-mask visible) to current working directory
- Tab: Toggle paint-mask visibility
- Left: Reduce brush size
- Right: Increase brush size
- Esc: Exit app

# Known Issues
- Seems like some frames occasionally fail to present/update after running the pipe(). Need to investigate.

# Known Limitations & Future Directions
- Reduce power draw: Consider using smaller model? Reduce frequency of image generation? Option to use NPU?
- For fun: Consider camera integration (camera takes photo via cv2 -> edge filtering -> edges guide drawing alongside the users painting).
- Consider adding tracing (import the edges of an existing image, and thereafter, allow user to edit it or draw on top of it).
- Either more brushes etc., or consider implementing into/working instead on an existing drawing app like Krita if there is enough demand for this.
