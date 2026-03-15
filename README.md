# cv_scribble_diffusion
cv_scribble_diffusion opens a drawing window using OpenCV, allowing you to sketch with your mouse. Your scribble guides a ControlNet-driven Stable Diffusion inpainting pipeline that runs asynchronously and reveals generated imagery with a smooth animated wavefront expanding outward from your drawn strokes — creating a continuous loop of visual inspiration.

![cv_scribble_diffusion_demo](https://github.com/user-attachments/assets/4e8307c4-4dd4-437d-9142-435f09a02ce7)

This is partly inspired by already existing similar projects, for example (not necessarily all inclusive):
- krita-ai-diffusion (https://github.com/Acly/krita-ai-diffusion/releases/tag/v1.9.0)

# Repository Contents
1. `main.py` — application entry point; all drawing, diffusion, and animation logic
2. `functions_diffusion_image_s.py` — thin wrappers around the diffusers pipeline calls
3. `LICENSE`
4. `README.md`

# Models
**The model weights are NOT included in this repository** (files are too large).
You must acquire and place them in the project root yourself before running.

### Required models (place in project root)

| Directory | Source |
|---|---|
| `stable-diffusion-v1-5/` | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| `sd-controlnet-scribble/` | [lllyasviel/sd-controlnet-scribble](https://huggingface.co/lllyasviel/sd-controlnet-scribble) |

The expected project layout after placing the models:
```
cv_scribble_diffusion/
    main.py
    functions_diffusion_image_s.py
    stable-diffusion-v1-5/      ← download separately
    sd-controlnet-scribble/     ← download separately
```

### Auto-downloaded model
`madebyollin/taesd` (AutoencoderTiny) is downloaded automatically on the first run via HuggingFace and cached locally. No manual step needed.

> **Note:** If you want to use a different base model or ControlNet, update `local_pipe_path` / `local_scribble_path` in `main.py` and adjust the pipeline class in `functions_diffusion_image_s.py` as needed.

# Requirements
1. Python 3.12+ (may also work on other 3.x versions)
2. Python packages: `numpy`, `opencv-python`, `pillow`, `diffusers`, `torch`
3. A CUDA-capable GPU is strongly recommended; CPU inference is supported but very slow

# Instructions to Run
1. Download the required models listed above and place them in the project root (paths must match `local_pipe_path` and `local_scribble_path` at the top of `main.py`).
2. *(Optional)* Edit parameters such as `prompt`, `reveal_mode`, or `brush_thickness` in `Main.__init__` inside `main.py`.
3. Run `main.py`:
   ```
   py main.py
   ```
4. The first run may take a minute or two to load the models. Once the window appears, draw on it and images will be generated continuously in the background.
5. Exit by pressing **Esc** in the window, or **Ctrl+C** in the terminal.

# Keyboard Inputs
| Key | Action |
|---|---|
| Draw (left mouse button) | Add strokes to guide generation |
| **Spacebar** | Reset canvas to blank |
| **Enter** | Save current image to working directory (no mask overlay) |
| **Tab** | Toggle stroke-mask visibility |
| **Left arrow** | Decrease brush size |
| **Right arrow** | Increase brush size |
| **Esc** | Exit app |

# Known Limitations & Future Directions
- **Power draw:** Consider a smaller model, lower generation frequency, or NPU offload.
- **Camera integration:** Capture a photo via cv2, apply edge filtering, and use the edges alongside the user's strokes as ControlNet input.
- **Image tracing:** Import an existing image's edges as a starting-point overlay for the user to draw on top of.
- **Richer tooling:** More brushes, layers, or deeper integration with an existing drawing app like Krita.
