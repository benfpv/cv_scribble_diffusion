# cv_scribble_diffusion

Real-time scribble-to-image generation with OpenCV, ControlNet scribble guidance, and Stable Diffusion inpainting.

Draw on the canvas and the app continuously generates imagery around your strokes. New results are blended in with a reveal animation so the output feels live and iterative.

The app uses a borderless OpenCV window with a compact **SCR** identity rail on the left; the full window title, rail mark, and borderless mode can be configured in `UIConfig` inside `config.py`.

The bottom prompt field lets you edit the generation prompt at runtime. `InferenceConfig.prompt` remains the startup default.

https://github.com/user-attachments/assets/0743eba9-6edc-4657-a160-ae95a96438b4

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

First launch may take a while due to model initialization; the terminal shows a startup progress indicator while models and UI are loading.

## Controls

### Mouse

- Left click + drag: draw strokes; single click places a brush-sized point
- Bottom prompt field: click to edit, drag to select prompt text, or double-click to select all
- Clicking the canvas while editing the prompt applies the prompt and starts drawing immediately
- Toolbar buttons (top strip):
    - `EXIT`: exit (multi-click confirmation)
    - `RESET`: clear canvas
    - `SAVE`: save current image to a timestamped `saved_image_*.png`
    - `MASK`: toggle mask visibility
    - `UNDO`: undo previous stroke
    - `THIN` / `THICK`: adjust brush size; the dot between them previews the current point size
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

When the prompt field is active, text input edits the prompt instead of triggering shortcuts. `Enter` applies the prompt, `Esc` cancels the edit, `Ctrl+A` selects all prompt text, `Backspace` / `Delete` edit or clear selected text, and `Left` / `Right` move the prompt cursor. Prompt input is single-line and capped by `UIConfig.prompt_max_chars`.

The status bar shows the current brush thickness as text, while the toolbar dot between `THIN` and `THICK` previews the actual current point size.
Brush size is clamped by `UIConfig.min_brush_thickness` / `UIConfig.max_brush_thickness`, and `MAX+` can raise runtime diffusion steps up to `InferenceConfig.max_runtime_inference_steps`.

## Configuration

Edit values in `config.py`:

- `ModelConfig`: local model paths and GPU usage
- `InferenceConfig`: startup prompt, guidance, crop behavior, step schedule, runtime max-step ceiling
- `RevealConfig`: reveal mode, interpolation, noise/outro tuning
- `UIConfig`: image/present size, brush defaults/limits, window settings including borderless mode, SCR identity rail, and prompt field sizing/limits

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
