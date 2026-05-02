# cv_scribble_diffusion

Real-time scribble-to-image generation with OpenCV, ControlNet scribble guidance, and Stable Diffusion inpainting.

Draw on the canvas and the app continuously generates imagery around your strokes. New results are blended in with a reveal animation so the output feels live and iterative.

The app window includes a compact **SCR** identity rail on the left; the full window title and rail mark can be configured in `UIConfig` inside `config.py`.

![ezgif-6ada7aacde7deecd](https://github.com/user-attachments/assets/701740fa-ab1a-492e-b688-8f53b629e586)

Inspired in part by projects such as [krita-ai-diffusion](https://github.com/Acly/krita-ai-diffusion/releases/tag/v1.9.0).

## Repository Overview

- `main.py`: app lifecycle, toolbar input, keyboard shortcuts, generation loop
- `pipeline.py`: model loading and inpaint execution
- `canvas.py`: stroke masks, brush behavior, image compositing helpers
- `animator.py`: reveal/interpolation animation pipeline
- `reveal.py`: reveal map and blend math
- `config.py`: centralized app settings (`ModelConfig`, `InferenceConfig`, `RevealConfig`, `UIConfig`)
- `generation.py`: crop planning, mask dilation, and inpaint assembly helpers
- `ui.py`: window composition and toolbar rendering
- `tests/`: unit and integration coverage for layout, generation helpers, undo, logging, and pipeline wiring
- `requirements.txt`: Python dependencies

## Models

Model weights are not included in this repository.

Download and place these folders in the project root:

| Directory | Source |
|---|---|
| `stable-diffusion-v1-5/` | [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| `sd-controlnet-scribble/` | [lllyasviel/sd-controlnet-scribble](https://huggingface.co/lllyasviel/sd-controlnet-scribble) |

`madebyollin/taesd` is downloaded automatically on first run.

> **Note:** To use a different base model or ControlNet, update `ModelConfig.pipe_path` / `ModelConfig.scribble_path` in `config.py` and adjust `pipeline.py` as needed.

## Requirements

- Python 3.10+
- GPU strongly recommended (CPU is supported but much slower)
- Packages listed in `requirements.txt`:
    - `numpy`
    - `opencv-python`
    - `pillow`
    - `diffusers`
    - `torch`
    - `accelerate`

## Quick Start

1. Open a terminal in the repository root.
2. Create and activate a virtual environment.
3. Install dependencies:

     ```bash
     pip install -r requirements.txt
     ```

4. Ensure required model directories are present.
5. Run the app:

     ```bash
     py main.py
     ```

First launch may take a while due to model initialization.

## Controls

### Mouse

- Left click + drag: draw strokes
- Toolbar buttons (top strip):
    - `EXIT`: exit (multi-click confirmation)
    - `RESET`: clear canvas
    - `SAVE`: save current image to `saved_image_N.png`
    - `MASK`: toggle mask visibility
    - `UNDO`: undo previous stroke
    - `THIN` / `THICK`: adjust brush size
    - `MAX-` / `MAX+`: adjust runtime max diffusion steps
    - `FPS`: cycle display FPS presets

### Keyboard

| Key | Action |
|---|---|
| `Space` | Reset canvas |
| `Enter` | Save current image |
| `Tab` | Toggle mask visibility |
| `Ctrl+Z`, `Z`, `U` | Undo last stroke |
| `Left Arrow` | Decrease brush thickness |
| `Right Arrow` | Increase brush thickness |
| `Esc` | Arm/confirm exit (press twice to exit) |

## Configuration

Edit values in `config.py`:

- `ModelConfig`: local model paths and GPU usage
- `InferenceConfig`: prompt, guidance, crop behavior, step schedule
- `RevealConfig`: reveal mode, interpolation, noise/outro tuning
- `UIConfig`: image/present size, brush defaults, window settings (including SCR identity rail)

## Troubleshooting

- Startup fails loading models:
    - Verify `stable-diffusion-v1-5/` and `sd-controlnet-scribble/` exist in the repository root.
- Very slow generation:
    - Confirm CUDA is available and `use_gpu` is enabled in config.
- No visible effect after drawing:
    - Draw a clearer stroke, then wait for the next generation pass.

## Future Directions

- Add a Gradio demo for browser-based interaction and easier sharing.
- Add camera/photo import as a source for initial edge guidance.
- Add image tracing overlays and richer brush tooling.
- Explore lower-power runtime modes (smaller models or reduced generation frequency).
