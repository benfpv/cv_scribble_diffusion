"""cv_scribble_diffusion — real-time scribble-to-image OpenCV application.

Draw on the OpenCV window with your mouse; a ControlNet-guided Stable Diffusion
inpainting pipeline continuously generates imagery around your strokes and reveals
it with an animated wavefront that expands outward from the drawn lines.

Architecture
------------
- StableDiffusionControlNetInpaintPipeline (SD v1.5 + scribble ControlNet)
- UniPCMultistepScheduler for fast multi-step sampling
- AutoencoderTiny (TAESD) for low-latency per-step preview decoding
- Per-generation background threads:
    _interp_thread_latent_lerp  — lerps CPU latents between steps, decodes with TAESD
    _interp_thread              — outro wavefront animation after generation completes

Model setup
-----------
SD v1.5 and the scribble ControlNet weights are NOT included in this repository.
Place the model directories in the project root before running (see README.md).
TAESD (madebyollin/taesd) is downloaded automatically on first run via HuggingFace.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import time
import cv2
from PIL import Image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler, AutoencoderTiny
import torch
import threading

from functions_diffusion_image_s import *

# Path to the local model directory
local_pipe_path = r"stable-diffusion-v1-5"

local_scribble_path = r"sd-controlnet-scribble"

# Keyboard Inputs
# - Spacebar: Reset the canvas to blank
# - Enter: Save the current image (without paint-mask visible) to current working directory
# - Tab: Toggle paint-mask visibility
# - Left: Reduce brush size
# - Right: Increase brush size
# - Esc: Exit app

class Main():
    """Application entry point: owns all state, threads, and the OpenCV event loop."""

    def __init__(self):
        """Initialise all parameters, state, models, and the OpenCV window."""
        ### Parameters (all x sizes must be identical to y sizes for now) ###
        self.image_sizes_minmax = [[240,240], [280,280], [320,320], [360,360], [400,400], [440,440], [480,480], [512,512]]
        self.image_size = (512,512) 
        self.present_size = (512,512) # window size, image_size is resized to present size before presenting to window.
        self.use_gpu = True # False for CPU (for artists), True for GPU (for showoffs); disable "enable optimizations
        
        self.brush_thickness = 2
        self.prompt = "colourful beautiful anime manga"
        self.crop_pad = 64    # padding (px) around stroke bbox for regional diffusion
        self.mask_dilate = 48  # how far (px) beyond the stroke lines to let the model paint
        # Reveal mode: 1 = smooth radial wavefront
        #              2 = fractal/fbm soft branches
        #              3 = cellular/Voronoi — crystalline patches with sharp seam edges
        #              4 = angular spokes — spiked starburst with hard wedge boundaries
        self.reveal_mode = 3
        self.stochastic_noise_strength = 0.35  # amplitude of stochastic perturbation (0.0-0.5)
        self.reveal_white_steps = 0.25  # fraction of total steps over which revealed area fades from white to decoded
        self.reveal_outro_alpha = 0.35   # extra wavefront alpha animated AFTER generation completes
        self.reveal_outro_duration = 1.5 # seconds the post-generation reveal animation takes
        self.latent_interp_smooth = 0.55 # temporal EMA weight per frame (0=none, closer to 1=slower/smoother)
        
        self.min_inference_steps = 2
        self.max_inference_steps = 18
        self.rate_inference_steps_change = 2
        
        self.pen_controls_active = True
        
        ### Init ###
        if torch.cuda.is_available():
            print("CUDA is available.")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
        else:
            print("CUDA is not available. Running on CPU.")
            self.use_gpu = False
            #exit()
        print(f"Use GPU?: {self.use_gpu}")
        
        self.drawing = False
        self.inhibit_this_mouse_event = False
        self.present_size_half = (int(self.present_size[0]*0.5), int(self.present_size[1]*0.5))
        self.mask = np.zeros(self.image_size, dtype="uint8")
        self.mask_active = np.zeros(self.present_size, dtype="uint8")
        self.mask_present = np.zeros((self.present_size[0],self.present_size[1],3), dtype="uint8")
        self.image = Image.new("RGB", self.image_size, (0, 0, 0))
        self.image_present = np.zeros((self.present_size[0],self.present_size[1],3), dtype="uint8")
        self.prev_image_present = np.zeros((self.present_size[0],self.present_size[1],3), dtype="uint8")
        self.current_inference_steps = 1  # set before each run so callback can compute alpha
        self._crop_region = None  # (x1, y1, x2, y2) in image_size coords; None = full image
        self._dist_map_norm = None  # float32 map at present-size: 0.0=stroke, 1.0=inpaint boundary
        self.interp_fps = 30         # target FPS for smooth expansion animation between steps
        self._frame_lock = threading.Lock()  # serialises all writes to self.image_present
        self._interp_stop = threading.Event()
        self._interp_decoded_f32 = None  # decoded frame (float32) from last completed step
        self._interp_alpha_start = 0.0
        self._interp_alpha_end = 0.0
        self._interp_edge = 0.08
        self._prev_alpha = 0.0           # ideal target alpha after each step
        self._last_display_alpha = 0.0   # actual alpha last written to image_present (updated by thread)
        self._prev_latents = None        # CPU float32 numpy latents from last step, for latent lerping
        self._last_interp_f32 = None     # last EMA frame from latent lerp thread (for outro cross-fade)
        self._step_wall_time = 1.0   # estimated seconds per step, updated each callback
        self._step_wall_ts = 0.0
        
        self.image_sizes_max_index = len(self.image_sizes_minmax)-1
        self.image_size_index = 0
        
        # Brush Params
        self.brush_point_thickness = int(self.brush_thickness-0.5)
        self.brush_stroke_multiplier = 1.4
        self.brush_stroke_thickness = max([1, round(self.brush_thickness * self.brush_stroke_multiplier)])
        
        # Pen Controls
        # Exit
        self.pen_controls_exit_loc_begin = [self.present_size_half[0] -62, 2]
        self.pen_controls_exit_loc_end = [self.present_size_half[0] -22, 12]
        self.pen_controls_exit_color = [100,100,100]
        self.pen_controls_exit_color_triggered = [180,180,180]
        # Reset
        self.pen_controls_reset_loc_begin = [self.present_size_half[0] - 20, 2]
        self.pen_controls_reset_loc_end = [self.present_size_half[0] + 20, 12]
        self.pen_controls_reset_color_static = [50,50,150]
        self.pen_controls_reset_color = [50,50,150]
        self.pen_controls_reset_color_triggered = [80,80,240]
        # Mask Visiblity
        self.pen_controls_visibility_loc_begin = [self.present_size_half[0] + 22, 2]
        self.pen_controls_visibility_loc_end = [self.present_size_half[0] + 62, 12]
        self.pen_controls_visibility_color_on = [50,200,50]
        self.pen_controls_visibility_color = [50,200,50]
        self.pen_controls_visibility_color_off = [50,120,50]
        
        # Other
        self.num_inference_steps = -1
        
        self.prev_x = -1
        self.prev_y = -1
        
        # Default Params
        self.exit_triggered = False
        self.mask_visibility_toggle = True
        self.image_save_count = 1
        self.image_diffused_this_loop = False
        self.skip_next_image = False
        
        self.image_store_limit_count = 9

        # Load ControlNet model
        self.controlnet = ControlNetModel.from_pretrained(
                                                        local_scribble_path, 
                                                        torch_dtype=torch.float16
                                                        )
        # Load Stable Diffusion pipeline
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                                                                    local_pipe_path,
                                                                    controlnet=self.controlnet,
                                                                    torch_dtype=torch.float16
                                                                    )
        if (self.use_gpu):
            self.pipe.to("cuda")
            self.controlnet.to("cuda")
            self.pipe.enable_model_cpu_offload()
            self.pipe.enable_sequential_cpu_offload()
            
        # Tweak the pipeline
        self.pipe.safety_checker = None
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_vae_slicing()
        self.pipe.enable_attention_slicing()

        # Load TAESD for fast preview decoding in step callback (~10x faster than full VAE)
        self._taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16)
        if self.use_gpu:
            self._taesd.to("cuda")
        self._taesd.eval()
        
        cv2.namedWindow("ai_paint_diffusion")
        cv2.setMouseCallback("ai_paint_diffusion", self.mouse_callback)
    
    def mouse_callback(self, event, x, y, flags, param):
        """OpenCV mouse callback: handles pen-control clicks, stroke drawing, and stroke commit."""
        # Drawing
        if (event == cv2.EVENT_LBUTTONDOWN):
            # Pen Controls
            if (self.pen_controls_active == True):
                if (y < 50): # additional filter for efficiency (assume pen controls are always y < 50)
                    if ((self.pen_controls_exit_loc_begin[0] < x < self.pen_controls_exit_loc_end[0]) and (self.pen_controls_exit_loc_begin[1] < y < self.pen_controls_exit_loc_end[1])):
                        print("...[Pen Controls] Exiting...")
                        self.trigger_exit()
                    elif ((self.pen_controls_reset_loc_begin[0] < x < self.pen_controls_reset_loc_end[0]) and (self.pen_controls_reset_loc_begin[1] < y < self.pen_controls_reset_loc_end[1])):
                        if (self.skip_next_image == False):
                            print("...[Pen Controls] Reset Canvas (changes take effect on next present)...")
                            self.reset_canvas()
                            self.pen_controls_reset_color = self.pen_controls_reset_color_triggered
                            self.inhibit_this_mouse_event = True
                    elif ((self.pen_controls_visibility_loc_begin[0] < x < self.pen_controls_visibility_loc_end[0]) and (self.pen_controls_visibility_loc_begin[1] < y < self.pen_controls_visibility_loc_end[1])):
                        if (self.mask_visibility_toggle):
                            print(f"...[Pen Controls] Toggle Mask Visibility [Off] (changes take effect on next present)...")
                            self.mask_visibility_toggle = False
                            self.pen_controls_visibility_color = self.pen_controls_visibility_color_off
                            self.inhibit_this_mouse_event = True
                        else:
                            print(f"...[Pen Controls] Toggle Mask Visibility [On]...")
                            self.mask_visibility_toggle = True
                            self.inhibit_this_mouse_event = True
            if (self.inhibit_this_mouse_event == False):
                self.drawing = True
                cv2.circle(self.mask_active, (x, y), self.brush_point_thickness, 150, -1)
                self.num_inference_steps = self.min_inference_steps
                self.prev_x = x
                self.prev_y = y
            else:
                self.inhibit_this_mouse_event = False
        elif (event == cv2.EVENT_MOUSEMOVE):
            if (self.drawing):
                cv2.line(self.mask_active, (self.prev_x, self.prev_y), (x,y), 150, self.brush_stroke_thickness)
                self.num_inference_steps = self.min_inference_steps
                self.prev_x = x
                self.prev_y = y
        elif (event == cv2.EVENT_LBUTTONUP):
            self.drawing = False
            self.prev_x = x
            self.prev_y = y
            self.mask_active = np.zeros(self.present_size, dtype="uint8")
    
    def reset_canvas(self):
        """Clear all stroke data, generated imagery, and interpolation state for a fresh start."""
        self.drawing = False
        self.mask = np.zeros(self.image_size, dtype="uint8")
        self.mask_active = np.zeros(self.present_size, dtype="uint8")
        self.mask_present = np.zeros((self.present_size[0],self.present_size[1],3), dtype="uint8")
        self.image = Image.new("RGB", self.image_size, (0, 0, 0))
        self.image_present = np.zeros((self.present_size[0],self.present_size[1],3), dtype="uint8")
        self.prev_image_present = np.zeros((self.present_size[0],self.present_size[1],3), dtype="uint8")
        self._crop_region = None
        self._dist_map_norm = None
        self._interp_stop.set()  # kill any running interp thread
        self._interp_decoded_f32 = None
        self._prev_latents = None
        self._last_interp_f32 = None
        self._last_display_alpha = 0.0
        self._step_wall_ts = 0.0
        self.num_inference_steps = -1
        self.skip_next_image = True
        
    def _step_callback(self, step, timestep, latents):
        """Latent interpolation: hand latents to the interp thread so it can lerp+decode at display FPS."""
        if self.skip_next_image:
            return
        # Measure wall time of the previous step so interp thread can pace correctly
        now = time.time()
        if self._step_wall_ts > 0:
            self._step_wall_time = now - self._step_wall_ts
        self._step_wall_ts = now
        # Stop previous inter-step interpolation
        self._interp_stop.set()
        alpha = (step + 1) / max(self.current_inference_steps, 1)
        edge = 0.08 if self.reveal_mode == 1 else 0.12
        # Snapshot latents to CPU numpy — cheap, non-blocking, no GPU stall
        curr_latents_np = latents.float().cpu().numpy()
        prev_latents_np = self._prev_latents if self._prev_latents is not None else curr_latents_np
        if self.interp_fps > 0:
            self._interp_stop = threading.Event()
            snap_dist     = self._dist_map_norm.copy() if self._dist_map_norm is not None else None
            snap_crop     = self._crop_region
            snap_prev     = self.prev_image_present.copy()
            snap_alpha_s  = self._last_display_alpha  # actual displayed alpha, not the ideal target
            snap_alpha_e  = alpha
            snap_duration = max(self._step_wall_time, 0.05)
            threading.Thread(
                target=self._interp_thread_latent_lerp,
                args=(self._interp_stop, prev_latents_np, curr_latents_np,
                      snap_dist, snap_crop, snap_prev,
                      snap_alpha_s, snap_alpha_e, edge, snap_duration),
                daemon=True
            ).start()
        self._prev_latents = curr_latents_np
        self._prev_alpha = alpha  # advance for next step

    def _decoded_to_frame_f32(self, decoded_uint8, edge):
        """Resize decoded uint8 RGB array to the correct display size (crop or full) as float32 BGR."""
        return self._decoded_to_frame_f32_snap(decoded_uint8, self._crop_region)

    def _decoded_to_frame_f32_snap(self, decoded_uint8, crop_region):
        """Thread-safe version: takes crop_region explicitly."""
        if crop_region is not None:
            cx1, cy1, cx2, cy2 = crop_region
            sx = self.present_size[0] / self.image_size[0]
            sy = self.present_size[1] / self.image_size[1]
            pw = int(cx2 * sx) - int(cx1 * sx)
            ph = int(cy2 * sy) - int(cy1 * sy)
            return cv2.resize(cv2.cvtColor(decoded_uint8, cv2.COLOR_RGB2BGR),
                              (pw, ph), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        return cv2.resize(cv2.cvtColor(decoded_uint8, cv2.COLOR_RGB2BGR),
                          self.present_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def _apply_reveal_frame(self, decoded_bgr_f32, alpha, edge):
        """Composite decoded_bgr_f32 into image_present using the dist map at the given alpha."""
        self._apply_reveal_frame_snap(
            decoded_bgr_f32, alpha, edge,
            self._dist_map_norm, self._crop_region, self.prev_image_present)

    def _apply_reveal_frame_snap(self, decoded_bgr_f32, alpha, edge, dist_map, crop_region, prev_img, stop_event=None):
        """Same as _apply_reveal_frame but takes all shared state as explicit args (thread-safe)."""
        # Blend revealed area from white toward decoded over the first reveal_white_steps fraction
        white_w = float(np.clip(1.0 - alpha / max(self.reveal_white_steps, 1e-6), 0.0, 1.0))
        if white_w > 0:
            source = decoded_bgr_f32 * (1.0 - white_w) + 255.0 * white_w
        else:
            source = decoded_bgr_f32
        # --- compute result into a local variable first ---
        if crop_region is not None and dist_map is not None:
            cx1, cy1, cx2, cy2 = crop_region
            sx = self.present_size[0] / self.image_size[0]
            sy = self.present_size[1] / self.image_size[1]
            px1, py1 = int(cx1 * sx), int(cy1 * sy)
            px2, py2 = int(cx2 * sx), int(cy2 * sy)
            reveal = self._compute_reveal(dist_map, alpha, edge)[:, :, np.newaxis]
            prev_crop = prev_img[py1:py2, px1:px2].astype(np.float32)
            composited = (source * reveal + prev_crop * (1.0 - reveal)).astype(np.uint8)
            result = prev_img.copy()
            result[py1:py2, px1:px2] = composited
        elif dist_map is not None:
            reveal = self._compute_reveal(dist_map, alpha, edge)[:, :, np.newaxis]
            result = (
                source * reveal +
                prev_img.astype(np.float32) * (1.0 - reveal)
            ).astype(np.uint8)
        else:
            result = cv2.addWeighted(
                prev_img, 1.0 - alpha,
                source.astype(np.uint8), alpha, 0)
        # Guard: if this interp interval was cancelled during computation, discard the result
        if stop_event is not None and stop_event.is_set():
            return
        with self._frame_lock:
            self.image_present = result

    def _interp_thread_latent_lerp(self, stop_event, prev_latents_np, curr_latents_np,
                                    dist_map, crop_region, prev_img,
                                    alpha_s, alpha_e, edge, step_duration):
        """Sub-step latent interpolation: lerp between consecutive denoising step latents,
        decode each intermediate frame with TAESD, and composite via reveal wavefront.
        This splits one denoising step into ~(interp_fps * step_duration) smooth sub-frames."""
        frame_dt = 1.0 / max(self.interp_fps, 1)
        start_t  = time.time()
        taesd    = self._taesd
        device   = next(taesd.parameters()).device
        diff_np  = curr_latents_np - prev_latents_np  # precompute delta
        smooth   = self.latent_interp_smooth
        last_f32 = self._last_interp_f32  # seed EMA from previous thread's last frame for continuity
        while not stop_event.is_set() and not self.skip_next_image:
            elapsed   = time.time() - start_t
            t         = min(elapsed / step_duration, 1.0)
            sub_alpha = alpha_s + (alpha_e - alpha_s) * t
            # Lerp latents on CPU, decode on device with TAESD
            lerped_np = prev_latents_np + diff_np * t
            lerped_t  = torch.from_numpy(lerped_np).to(device=device, dtype=torch.float16)
            with torch.no_grad():
                decoded = taesd.decode(lerped_t).sample.clamp(0, 1)
            decoded_np    = decoded.cpu().permute(0, 2, 3, 1).float().numpy()[0]
            decoded_uint8 = (decoded_np * 255).astype(np.uint8)
            decoded_bgr_f32 = self._decoded_to_frame_f32_snap(decoded_uint8, crop_region)
            # Temporal EMA: seed from previous thread so there is never a raw-decode first frame
            if last_f32 is None or last_f32.shape != decoded_bgr_f32.shape:
                last_f32 = decoded_bgr_f32
            else:
                decoded_bgr_f32 = decoded_bgr_f32 * (1.0 - smooth) + last_f32 * smooth
                last_f32 = decoded_bgr_f32
            # Always keep _last_interp_f32 and _last_display_alpha current so outro is always up to date
            self._last_interp_f32 = last_f32
            self._last_display_alpha = sub_alpha
            self._apply_reveal_frame_snap(decoded_bgr_f32, sub_alpha, edge,
                                          dist_map, crop_region, prev_img, stop_event=stop_event)
            if t >= 1.0:
                break
            next_tick = start_t + (elapsed // frame_dt + 1) * frame_dt
            sleep_t   = next_tick - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _interp_thread(self, stop_event, decoded_f32, dist_map, crop_region,
                       prev_img, alpha_s, alpha_e, edge, step_duration, start_f32=None):
        """Animate the reveal wavefront at display FPS over a fixed decoded frame.
        If start_f32 is given, cross-fades from start_f32 → decoded_f32 over the duration
        (used by the outro to bridge the TAESD→full-VAE gap)."""
        frame_dt = 1.0 / max(self.interp_fps, 1)
        start_t  = time.time()
        while not stop_event.is_set() and not self.skip_next_image:
            elapsed   = time.time() - start_t
            t         = min(elapsed / step_duration, 1.0)
            sub_alpha = alpha_s + (alpha_e - alpha_s) * t
            if start_f32 is not None:
                frame = start_f32 * (1.0 - t) + decoded_f32 * t
            else:
                frame = decoded_f32
            self._apply_reveal_frame_snap(frame, sub_alpha, edge, dist_map, crop_region, prev_img, stop_event=stop_event)
            if t >= 1.0:
                break
            next_tick = start_t + (elapsed // frame_dt + 1) * frame_dt
            sleep_t   = next_tick - time.time()
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _compute_reveal(self, dist_map, alpha, edge):
        """Return per-pixel blend weight in [0,1] for the current denoising step.
        dist_map: float32 (H,W), 0=stroke, 1=inpaint edge, 2=outside (never revealed)."""
        return np.clip((alpha - dist_map) / edge + 0.5, 0.0, 1.0)

    def _make_stochastic_noise(self, h, w, cx, cy):
        """Dispatch to the appropriate noise generator based on reveal_mode."""
        if self.reveal_mode == 2:
            return self._make_fractal_noise(h, w)
        elif self.reveal_mode == 3:
            return self._make_cellular_noise(h, w)
        elif self.reveal_mode == 4:
            return self._make_shard_noise(h, w, cx, cy)
        return np.zeros((h, w), dtype=np.float32)

    def _make_cellular_noise(self, h, w, n_cells=25):
        """Worley/Voronoi noise: smooth within cells, sharp discontinuities at boundaries.
        Returns float32 in [-1, 1]: low near cell seeds (reveals sooner), high at boundaries."""
        pts_x = (np.random.rand(n_cells) * w).astype(np.float32)
        pts_y = (np.random.rand(n_cells) * h).astype(np.float32)
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
        min_dist = np.full((h, w), np.inf, dtype=np.float32)
        for i in range(n_cells):
            d = np.hypot(gx - pts_x[i], gy - pts_y[i])
            np.minimum(min_dist, d, out=min_dist)
        min_dist /= (min_dist.max() + 1e-6)
        return min_dist * 2.0 - 1.0  # [-1, 1]

    def _make_shard_noise(self, h, w, cx, cy):
        """Angular spoke noise: each wedge around (cx,cy) gets a random hard offset.
        Creates crisp spiked/starburst boundaries with sharp edges between wedges."""
        n_spokes = np.random.randint(6, 18)
        gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
        angles = np.arctan2(gy - cy, gx - cx)  # [-pi, pi]
        band_width = 2.0 * np.pi / n_spokes
        band_idx = ((angles + np.pi) / band_width).astype(np.int32) % n_spokes
        spoke_offsets = (np.random.rand(n_spokes).astype(np.float32) - 0.5)
        return spoke_offsets[band_idx]  # ~[-0.5, 0.5]

    def _make_fractal_noise(self, h, w):
        """Multi-octave fractal noise in [-1, 1]. No external deps, pure numpy+cv2."""
        noise = np.zeros((h, w), dtype=np.float32)
        amplitude = 1.0
        for freq in [3, 6, 12, 24, 48]:
            fh, fw = max(2, min(h, freq)), max(2, min(w, freq))
            layer = np.random.rand(fh, fw).astype(np.float32)
            layer = cv2.resize(layer, (w, h), interpolation=cv2.INTER_LINEAR)
            noise += layer * amplitude
            amplitude *= 0.5
        noise -= noise.mean()
        std = noise.std()
        if std > 1e-6:
            noise /= std
        return noise

    def _make_feather_mask(self, w, h, feather):
        """Returns a float32 (h,w) mask, 1.0 in centre fading to 0.0 at edges."""
        mask = np.ones((h, w), dtype=np.float32)
        if feather > 1:
            ksize = feather * 4 + 1
            if ksize % 2 == 0:
                ksize += 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), feather)
        return mask

    def trigger_exit(self):
        """Signal the application to exit and update the exit-button colour to confirmed state."""
        self.exit_triggered = True
        self.pen_controls_exit_color = self.pen_controls_exit_color_triggered
        cv2.rectangle(self.image_present, self.pen_controls_exit_loc_begin, self.pen_controls_exit_loc_end, self.pen_controls_exit_color, -1)
        # Present
        cv2.imshow("ai_paint_diffusion", self.image_present)
        cv2.moveWindow("ai_paint_diffusion", 300, 0)
    
    def async_diffusion(self):
        """Background thread: runs the diffusion pipeline continuously whenever strokes are present.

        Each iteration:
        1. Computes a tight crop bbox around the current strokes (falls back to full-image).
        2. Binarises the stroke mask into a ControlNet scribble image and an inpaint mask.
        3. Builds a signed-distance-field reveal map, optionally perturbed by stochastic noise.
        4. Runs the pipeline, firing _step_callback on every denoising step.
        5. After completion, launches an outro thread to briefly extend the wavefront animation.
        """
        while 1:
            if (self.num_inference_steps > 1):
                mask_gray = self.mask.copy()
                nonzero = cv2.findNonZero(mask_gray)

                # Decide whether strokes fit in a useful crop
                use_crop = False
                cx1 = cy1 = 0
                cx2, cy2 = self.image_size[0], self.image_size[1]
                if nonzero is not None:
                    bx, by, bw, bh = cv2.boundingRect(nonzero)
                    pad = self.crop_pad
                    rx1 = max(0, bx - pad)
                    ry1 = max(0, by - pad)
                    rx2 = min(self.image_size[0], bx + bw + pad)
                    ry2 = min(self.image_size[1], by + bh + pad)
                    # Snap dimensions DOWN to multiple of 8 (clamp may have broken the round-up)
                    rw = (rx2 - rx1) // 8 * 8
                    rh = (ry2 - ry1) // 8 * 8
                    rx2 = rx1 + rw
                    ry2 = ry1 + rh
                    if rw * rh < int(0.7 * self.image_size[0] * self.image_size[1]) and rw >= 64 and rh >= 64:
                        use_crop = True
                        cx1, cy1, cx2, cy2 = rx1, ry1, rx2, ry2

                init_pil = self.image
                # Binarise to pure white strokes — scribble ControlNet expects 255, not 150
                control_np = np.where(mask_gray > 0, np.uint8(255), np.uint8(0))
                control_pil = Image.fromarray(cv2.cvtColor(control_np, cv2.COLOR_GRAY2RGB))
                inference_steps = self.num_inference_steps  # fix value before diffusion starts
                self.current_inference_steps = inference_steps
                # Start alpha at -edge/2 so reveal=0 even at dist=0 on the very first sub-frame
                _edge0 = 0.08 if self.reveal_mode == 1 else 0.12
                self._prev_alpha = -_edge0 / 2
                self._last_display_alpha = -_edge0 / 2
                self._prev_latents = None   # new generation may have different latent shape
                pre_gen_prev = self.prev_image_present.copy()  # snapshot before pipeline runs

                if use_crop:
                    self._crop_region = (cx1, cy1, cx2, cy2)
                    cw, ch = cx2 - cx1, cy2 - cy1
                    # Dilate the stroke mask to give the model a halo around each line
                    dil = self.mask_dilate * 2 + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
                    stroke_crop = mask_gray[cy1:cy2, cx1:cx2]
                    dil_crop = cv2.dilate(stroke_crop, kernel)
                    # Binarise inpaint mask — pipeline expects 0=keep, 255=repaint
                    inpaint_mask = Image.fromarray(np.where(dil_crop > 0, np.uint8(255), np.uint8(0)))
                    # Build normalised distance map (0=stroke, 1=inpaint boundary) at present-size
                    dist_input = np.where(stroke_crop > 0, np.uint8(0), np.uint8(255))
                    dist_raw = cv2.distanceTransform(dist_input, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                    dmax = float(dist_raw[dil_crop > 0].max()) if np.any(dil_crop > 0) else 1.0
                    dist_norm = np.where(dil_crop > 0, dist_raw / max(dmax, 1e-6), 2.0).astype(np.float32)
                    if self.reveal_mode >= 2:
                        noise = self._make_stochastic_noise(ch, cw, cw // 2, ch // 2)
                        dist_norm = np.where(dil_crop > 0,
                                             np.clip(dist_norm + noise * self.stochastic_noise_strength, 0.0, 1.5),
                                             2.0).astype(np.float32)
                    sx = self.present_size[0] / self.image_size[0]
                    sy = self.present_size[1] / self.image_size[1]
                    px1_d, py1_d = int(cx1 * sx), int(cy1 * sy)
                    px2_d, py2_d = int(cx2 * sx), int(cy2 * sy)
                    self._dist_map_norm = cv2.resize(dist_norm, (px2_d - px1_d, py2_d - py1_d), interpolation=cv2.INTER_LINEAR)
                    result_crop = Functions_Diffusion_Image.run_cnet_pipe_inpaint(
                        self.pipe, init_pil.crop((cx1, cy1, cx2, cy2)), inpaint_mask,
                        control_pil.crop((cx1, cy1, cx2, cy2)),
                        self.prompt, inference_steps, width=cw, height=ch,
                        step_callback=lambda s, ts, l: self._step_callback(s, ts, l)
                    )
                    # Feathered paste-back into full image
                    result_np = np.array(result_crop)
                    img_np = np.array(init_pil)
                    feather_alpha = self._make_feather_mask(cw, ch, min(self.crop_pad // 2, 32))
                    a3 = np.stack([feather_alpha] * 3, axis=-1)
                    img_np[cy1:cy2, cx1:cx2] = (
                        result_np * a3 + img_np[cy1:cy2, cx1:cx2] * (1 - a3)
                    ).astype(np.uint8)
                    self.image = Image.fromarray(img_np)
                else:
                    self._crop_region = None
                    image_size = self.image_sizes_minmax[self.image_size_index]
                    # Dilate full-resolution stroke mask to define paint region
                    dil = self.mask_dilate * 2 + 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil))
                    dilated_full = cv2.dilate(mask_gray, kernel)
                    # Binarise inpaint mask — pipeline expects 0=keep, 255=repaint
                    inpaint_mask = Image.fromarray(cv2.resize(np.where(dilated_full > 0, np.uint8(255), np.uint8(0)), image_size, interpolation=cv2.INTER_NEAREST))
                    # Build normalised distance map (0=stroke, 1=inpaint boundary) at present-size
                    dist_input_f = np.where(mask_gray > 0, np.uint8(0), np.uint8(255))
                    dist_raw_f = cv2.distanceTransform(dist_input_f, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                    dmax_f = float(dist_raw_f[dilated_full > 0].max()) if np.any(dilated_full > 0) else 1.0
                    dist_norm_f = np.where(dilated_full > 0, dist_raw_f / max(dmax_f, 1e-6), 2.0).astype(np.float32)
                    if self.reveal_mode >= 2:
                        h_f, w_f = dist_norm_f.shape
                        noise_f = self._make_stochastic_noise(h_f, w_f, w_f // 2, h_f // 2)
                        dist_norm_f = np.where(dilated_full > 0,
                                               np.clip(dist_norm_f + noise_f * self.stochastic_noise_strength, 0.0, 1.5),
                                               2.0).astype(np.float32)
                    self._dist_map_norm = cv2.resize(dist_norm_f, self.present_size, interpolation=cv2.INTER_LINEAR)
                    result = Functions_Diffusion_Image.run_cnet_pipe_inpaint(
                        self.pipe, init_pil.resize(image_size), inpaint_mask,
                        control_pil.resize(image_size, Image.NEAREST),
                        self.prompt, inference_steps, width=image_size[0], height=image_size[1],
                        step_callback=lambda s, ts, l: self._step_callback(s, ts, l)
                    )
                    self.image = result.resize(self.image_size)
                    self.image_size_index += 1
                    self.image_size_index = np.min([self.image_size_index, self.image_sizes_max_index])

                final_frame = cv2.resize(cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR), self.present_size, interpolation=cv2.INTER_LINEAR)
                # Always update the background used by the next generation
                self.prev_image_present = final_frame.copy()
                # Launch outro: continue wavefront expand using the finished image for a moment longer
                if self.reveal_outro_alpha > 0 and self._dist_map_norm is not None:
                    # Don't write image_present here — outro cross-fade handles the visual transition
                    self._interp_stop.set()
                    self._interp_stop = threading.Event()
                    outro_dist  = self._dist_map_norm.copy()
                    outro_crop  = self._crop_region
                    outro_edge  = 0.08 if self.reveal_mode == 1 else 0.12
                    # decoded_f32 must be crop-sized when a crop region exists, same as step callback
                    if outro_crop is not None:
                        cx1, cy1, cx2, cy2 = outro_crop
                        sx = self.present_size[0] / self.image_size[0]
                        sy = self.present_size[1] / self.image_size[1]
                        px1 = int(cx1 * sx); py1 = int(cy1 * sy)
                        px2 = int(cx2 * sx); py2 = int(cy2 * sy)
                        outro_decoded = final_frame[py1:py2, px1:px2].astype(np.float32)
                        # Seed start_f32 from actual image_present (what's on screen right now)
                        with self._frame_lock:
                            outro_start_f32 = self.image_present[py1:py2, px1:px2].astype(np.float32)
                    else:
                        outro_decoded = final_frame.astype(np.float32)
                        with self._frame_lock:
                            outro_start_f32 = self.image_present.astype(np.float32)
                    # Use actual display alpha so the wavefront continues from where it really is
                    outro_alpha_s = self._last_display_alpha
                    threading.Thread(
                        target=self._interp_thread,
                        args=(self._interp_stop,
                              outro_decoded,
                              outro_dist, outro_crop, pre_gen_prev,
                              outro_alpha_s, outro_alpha_s + self.reveal_outro_alpha,
                              outro_edge, self.reveal_outro_duration,
                              outro_start_f32),  # cross-fade from actual last screen frame
                        daemon=True
                    ).start()
                else:
                    # No outro: write final frame directly
                    with self._frame_lock:
                        self.image_present = final_frame
                self.image_diffused_this_loop = True
                while (self.image_diffused_this_loop):
                    time.sleep(0.1)
                self.num_inference_steps += self.rate_inference_steps_change
                self.num_inference_steps = np.min([self.num_inference_steps, self.max_inference_steps])
                time.sleep(0.1)
            else:
                time.sleep(1)
            
    def run(self):
        """Start the diffusion thread and enter the OpenCV display/event loop."""
        thread = threading.Thread(target=self.async_diffusion, daemon=True)
        thread.start()
        
        while 1:
            ### Present ###
            if (np.any(self.mask_active)):
                self.mask = cv2.add(self.mask, cv2.resize(self.mask_active, self.image_size, interpolation=cv2.INTER_NEAREST))
                self.mask_present = cv2.resize(cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR), self.present_size, interpolation=cv2.INTER_LINEAR)
                self.image_size_index = 0
                self.num_inference_steps = self.min_inference_steps

            if (self.skip_next_image):
                if (self.image_diffused_this_loop):
                    self.image_diffused_this_loop = False
                    self.skip_next_image = False
                    with self._frame_lock:
                        self.image_present = np.zeros((self.present_size[0],self.present_size[1],3), dtype="uint8")
                    self.pen_controls_reset_color = self.pen_controls_reset_color_static
            else:
                self.image_diffused_this_loop = False

            # Build a per-frame display copy — never accumulate onto self.image_present,
            # which is owned by the interp/callback threads.
            with self._frame_lock:
                display_frame = self.image_present.copy()
            if (np.any(self.mask_active) and self.mask_visibility_toggle == False):
                display_frame = cv2.add(display_frame, cv2.cvtColor(self.mask_active, cv2.COLOR_GRAY2BGR))
            if (self.mask_visibility_toggle):
                display_frame = cv2.add(display_frame, self.mask_present)
                self.pen_controls_visibility_color = self.pen_controls_visibility_color_on
            # Draw Pre-Present Statics (Pen Controls)
            if (self.pen_controls_active == True):
                cv2.rectangle(display_frame, self.pen_controls_exit_loc_begin, self.pen_controls_exit_loc_end, self.pen_controls_exit_color, -1)
                cv2.rectangle(display_frame, self.pen_controls_reset_loc_begin, self.pen_controls_reset_loc_end, self.pen_controls_reset_color, -1)
                cv2.rectangle(display_frame, self.pen_controls_visibility_loc_begin, self.pen_controls_visibility_loc_end, self.pen_controls_visibility_color, -1)
            # Present
            cv2.imshow("ai_paint_diffusion", display_frame)
            cv2.moveWindow("ai_paint_diffusion", 300, 0)
            
            key = cv2.waitKey(1) & 0xFF  # Waits 1 ms for a key press
            
            if (key == 27): # Esc
                print("...[Esc] Exiting...")
                self.trigger_exit()
            elif (key == 9): # Tab
                if (self.mask_visibility_toggle):
                    print(f"...[Tab] Toggle Mask Visibility [Off] (changes take effect on next present)...")
                    self.mask_visibility_toggle = False
                else:
                    print(f"...[Tab] Toggle Mask Visibility [On]...")
                    self.mask_visibility_toggle = True
            elif (key == 13): # Enter
                self.image_save_count = 1
                for i in range(self.image_store_limit_count - self.image_save_count + 1):
                    self.image_save_count += 1
                    self.image_save_name = f"saved_image_{self.image_save_count}.png"
                    if (not os.path.isfile(self.image_save_name)):
                        break
                if (self.image_save_count >= (self.image_store_limit_count+1)):
                    print(f"...[Enter] Saving failed - image count ({self.image_save_count}) exceeds limit ({self.image_store_limit_count})...")
                else:
                    print(f"...[Enter] Saving {self.image_save_name}...")
                    image_to_save = self.image.copy()
                    image_to_save = image_to_save.resize(self.image_size, resample=Image.NEAREST)
                    image_to_save = cv2.resize(cv2.cvtColor(np.array(image_to_save), cv2.COLOR_RGB2BGR), self.present_size, interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(self.image_save_name, image_to_save)
            elif (key == 32): # Space Bar
                if (self.skip_next_image == False):
                    print("...[SpaceBar] Reset Canvas (changes take effect on next present)...")
                    self.reset_canvas()
            elif (key == 81): # Left Arrow
                self.brush_thickness -= 1
                self.brush_thickness = np.max([1, self.brush_thickness])
                print(f"...[Left] [-] Brush Thickness to {self.brush_thickness}...")
                self.brush_point_thickness = self.brush_thickness
                self.brush_stroke_thickness = int(self.brush_thickness * self.brush_stroke_multiplier)
            elif (key == 83): # Right Arrow
                self.brush_thickness += 1
                self.brush_thickness = np.min([20, self.brush_thickness])
                print(f"...[Right] [+] Brush Thickness to {self.brush_thickness}...")
                self.brush_point_thickness = self.brush_thickness
                self.brush_stroke_thickness = int(self.brush_thickness * self.brush_stroke_multiplier)
            
            if (self.exit_triggered == True):
                break
            
            time.sleep(0.01)
                
if (__name__ == "__main__"):
    app = Main()
    app.run()