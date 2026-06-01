"""Background diffusion worker.

Owns the daemon thread that drives the Stable Diffusion inpainting pipeline,
the cooperative-shutdown event, and the display-loop handshake events. The
worker reads and mutates shared generation state through a
:class:`~cv_scribble_diffusion.app.state.GenerationState` instance so the
display loop and the worker stay synchronized.
"""
import time
import threading
from typing import Callable, Optional

import cv2
import numpy as np

from cv_scribble_diffusion.generation.inputs import (
    decide_crop, make_dilation_kernel, make_control_image,
    compute_dist_map_inputs, build_inpaint_inputs,
)
from cv_scribble_diffusion.utils.colorspace import rgb_to_bgr
from cv_scribble_diffusion.infra.runtime_logging import get_logger
from cv_scribble_diffusion.app.state import GenState, GenerationState


logger = get_logger(__name__)


class _DiffusionCancelled(Exception):
    """Raised inside the pipeline step callback to abort cleanly on shutdown."""


class GenerationWorker:
    """Drives diffusion cycles on a background thread.

    The worker exposes the ``_stop_event``/``_gen_done``/``_reset_ack`` events
    used to coordinate with the display loop, plus ``start``/``stop`` lifecycle
    helpers. The actual loop body lives in :meth:`run`.
    """

    def __init__(self, cfg, canvas, animator, pipeline, dbg,
                 state: GenerationState,
                 prompt_provider: Callable[[], str],
                 notify: Callable[..., None],
                 max_steps_provider: Callable[[], int],
                 image_sizes_max_index: int):
        self.cfg = cfg
        self.canvas = canvas
        self.animator = animator
        self.pipe = pipeline
        self.dbg = dbg
        self.state = state
        self._prompt_provider = prompt_provider
        self._notify = notify
        self._max_steps_provider = max_steps_provider
        self.image_sizes_max_index = image_sizes_max_index

        # Generation tracking
        self._gen_seq = 0
        self.prev_gen_mask = np.zeros(cfg.ui.image_size, dtype="uint8")

        # Consecutive-failure tracking for exponential backoff.
        # NOTE: Not reset by reset_canvas() — the pipeline's instability is
        # independent of canvas state. Only a successful generation clears it.
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5

        # Cooperative shutdown for the diffusion thread.
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Synchronisation between diffusion thread and display loop.
        self._gen_done = threading.Event()
        self._reset_ack = threading.Event()
        self._reset_ack.set()

    # -- lifecycle ------------------------------------------------------------

    def start(self):
        """Spawn the diffusion daemon thread."""
        self._thread = threading.Thread(
            target=self.run, name="diffusion", daemon=True,
        )
        self._thread.start()

    def stop(self, join_timeout: float = 2.0):
        """Signal the thread to stop and wait briefly for it to exit."""
        self._stop_event.set()
        # Unblock any thread waiting on these events so it can observe stop.
        self._reset_ack.set()
        self._gen_done.set()
        if self._thread is not None:
            logger.info("Waiting for diffusion thread to stop")
            self._thread.join(timeout=join_timeout)
            if self._thread.is_alive():
                logger.warning("Diffusion thread did not stop within timeout")

    def reset_tracking(self):
        """Clear per-generation tracking on canvas reset."""
        self._gen_seq = 0
        self.prev_gen_mask = np.zeros(self.cfg.ui.image_size, dtype="uint8")

    # -- loop -----------------------------------------------------------------

    def run(self):
        """Background thread: runs the pipeline whenever strokes are present."""
        cfg = self.cfg
        icfg = cfg.inference
        ucfg = cfg.ui
        state = self.state
        while not self._stop_event.is_set():
            try:
                if state.gen_state != GenState.READY:
                    time.sleep(1)
                    continue

                state.gen_state = GenState.GENERATING
                # Note: thread_error is intentionally NOT cleared here. It is
                # only cleared on reset/undo so transient pipeline failures
                # remain visible to the user across subsequent successful runs.
                self._gen_seq += 1
                mask_gray = self.canvas.mask.copy()

                plan = decide_crop(
                    mask_gray, ucfg.image_size,
                    icfg.crop_pad, icfg.crop_alignment,
                    icfg.crop_area_threshold, icfg.crop_min_dim,
                )
                if plan is None:
                    logger.debug("Generation skipped: empty mask")
                    state.gen_state = GenState.IDLE
                    time.sleep(0.05)
                    continue
                logger.info(
                    "Generation cycle #%s starting: steps=%s use_crop=%s region=%s",
                    self._gen_seq, state.inference_steps, plan.use_crop, plan.region,
                )

                init_pil = self.canvas.image
                control_pil = make_control_image(mask_gray)
                inference_steps = state.inference_steps
                state.current_inference_steps = inference_steps
                kernel = make_dilation_kernel(icfg.mask_dilate)

                def step_cb(s, total_steps, l):
                    if self._stop_event.is_set():
                        # Best-effort cancellation: raising here aborts the diffusers loop.
                        raise _DiffusionCancelled()
                    if state.gen_state != GenState.RESETTING:
                        logger.debug("Pipeline step callback: step=%s/%s", s + 1, total_steps)
                        self.animator.on_step(s, state.current_inference_steps, l)
                    else:
                        logger.debug("Skipped step callback due to reset request")

                # Build distance map and inpaint inputs from helpers.
                dist_inputs = compute_dist_map_inputs(
                    mask_gray, self.prev_gen_mask, plan, ucfg, kernel,
                )
                if dist_inputs is None:
                    dist_map = None
                    logger.info("No delta strokes; using global crossfade reveal")
                else:
                    dist_map = self.animator.make_dist_map(
                        dist_inputs.delta_mask, dist_inputs.delta_dilated,
                        dist_inputs.out_size, dist_inputs.cx, dist_inputs.cy,
                    )

                ramp_size = (
                    None if plan.use_crop
                    else icfg.image_sizes_ramp[state.image_size_index]
                )
                inpaint = build_inpaint_inputs(
                    init_pil, mask_gray, control_pil, plan, kernel,
                    image_sizes_ramp_size=ramp_size,
                )

                self.animator.prepare_generation(
                    plan.region if plan.use_crop else None, dist_map,
                )

                # Debug artifacts mirror the previous per-branch tags.
                if plan.use_crop:
                    cw, ch = inpaint.width, inpaint.height
                    self.dbg.save_annotated_crop(init_pil, plan.region,
                        f"steps={inference_steps} size={cw}x{ch}")
                    self.dbg.save("crop_init", inpaint.init_image)
                    self.dbg.save("crop_control", inpaint.control_image)
                    self.dbg.save("crop_mask", inpaint.inpaint_mask)
                else:
                    self.dbg.save("full_init", inpaint.init_image)
                    self.dbg.save("full_control", inpaint.control_image)
                    self.dbg.save("full_mask", inpaint.inpaint_mask)

                prompt = self._prompt_provider()
                gen_start_time = time.time()
                result = self.pipe.run_inpaint(
                    inpaint.init_image, inpaint.inpaint_mask, inpaint.control_image,
                    prompt, inference_steps,
                    width=inpaint.width, height=inpaint.height,
                    step_callback=step_cb,
                )

                if state.gen_state == GenState.RESETTING:
                    logger.info("Reset requested during generation; skipping commit")
                    self._reset_ack.clear()
                    self._gen_done.set()
                    self._reset_ack.wait()
                    self._gen_done.clear()
                    time.sleep(0.05)
                    continue

                if plan.use_crop:
                    self.dbg.save("crop_result", result)
                    self.canvas.patch_image(plan.region, result, icfg.crop_feather_px)
                else:
                    self.dbg.save("full_result", result)
                    self.canvas.image = result.resize(ucfg.image_size)
                    state.image_size_index = min(
                        state.image_size_index + 1, self.image_sizes_max_index,
                    )

                final_frame = cv2.resize(
                    rgb_to_bgr(np.array(self.canvas.image)),
                    ucfg.present_size, interpolation=cv2.INTER_LINEAR)
                generation_duration = time.time() - gen_start_time
                logger.info("Generation inference complete in %.3fs; staging outro", generation_duration)
                self.animator.start_outro(final_frame, generation_duration)
                self.animator.wait_for_outro()
                logger.info("Outro completed")
                state.gen_state = GenState.READY
                state.gen_count += 1
                self.prev_gen_mask = mask_gray.copy()

                # Signal the display loop and wait for it to acknowledge
                self._reset_ack.clear()
                self._gen_done.set()
                self._reset_ack.wait()
                self._gen_done.clear()

                state.inference_steps = min(
                    state.inference_steps + icfg.rate_inference_steps_change,
                    self._max_steps_provider())
                self._consecutive_failures = 0
                time.sleep(0.1)
            except _DiffusionCancelled:
                logger.info("Diffusion cancelled by stop event")
                break
            except Exception as exc:
                logger.exception("Unhandled exception in diffusion thread")
                self._consecutive_failures += 1
                state.thread_error = str(exc)
                if self._consecutive_failures >= self._max_consecutive_failures:
                    self._notify(
                        f"Pipeline failing repeatedly ({self._consecutive_failures}x); see logs",
                        duration_s=10.0,
                    )
                if state.gen_state == GenState.GENERATING:
                    state.gen_state = GenState.READY
                if state.gen_state == GenState.RESETTING:
                    self._gen_done.set()
                else:
                    self._gen_done.clear()
                self._reset_ack.set()
                # Exponential backoff capped at 4s.
                backoff = min(0.25 * (2 ** (self._consecutive_failures - 1)), 4.0)
                self._stop_event.wait(timeout=backoff)
