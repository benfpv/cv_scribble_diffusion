"""Tests for DiffusionPipeline wiring with mocked model backends."""

import types

import torch
from PIL import Image

from cv_scribble_diffusion.config import AppConfig, InferenceConfig, ModelConfig
from cv_scribble_diffusion.generation import pipeline as pipeline_module


class FakeControlNetModel:
    last_call = None

    @classmethod
    def from_pretrained(cls, path, torch_dtype=None):
        cls.last_call = (path, torch_dtype)
        return cls()


class FakeScheduler:
    @classmethod
    def from_config(cls, config):
        return {"source": "unipc", "config": config}


class FakePipe:
    def __init__(self):
        self.scheduler = types.SimpleNamespace(config={"x": 1})
        self.safety_checker = "enabled"
        self.enable_seq_called = False
        self.enable_vae_called = False
        self.enable_attn_called = False
        self.fail_new_callback_once = False
        self.call_count = 0
        self.last_kwargs = None

    @classmethod
    def from_pretrained(cls, path, controlnet=None, torch_dtype=None):
        pipe = cls()
        pipe.from_pretrained_args = (path, controlnet, torch_dtype)
        return pipe

    def enable_sequential_cpu_offload(self):
        self.enable_seq_called = True

    def enable_vae_slicing(self):
        self.enable_vae_called = True

    def enable_attention_slicing(self):
        self.enable_attn_called = True

    def __call__(self, **kwargs):
        self.call_count += 1
        self.last_kwargs = kwargs
        if self.fail_new_callback_once and "callback_on_step_end" in kwargs:
            self.fail_new_callback_once = False
            raise TypeError("unsupported callback_on_step_end")

        if "callback_on_step_end" in kwargs:
            cb = kwargs["callback_on_step_end"]
            cb(None, 1, None, {"latents": "L"})

        return types.SimpleNamespace(
            images=[Image.new("RGB", (kwargs["width"], kwargs["height"]), (1, 2, 3))]
        )


class FakeTAESD:
    def __init__(self):
        self.to_device = None
        self.eval_called = False

    @classmethod
    def from_pretrained(cls, _id, torch_dtype=None):
        return cls()

    def to(self, device):
        self.to_device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def parameters(self):
        return iter([torch.tensor([1.0])])


def _cfg(use_gpu=True):
    return AppConfig(
        model=ModelConfig(
            pipe_path="pipe",
            scribble_path="scribble",
            taesd_id="taesd",
            use_gpu=use_gpu,
        ),
        inference=InferenceConfig(),
    )


def _patch_backends(monkeypatch):
    monkeypatch.setattr(pipeline_module, "ControlNetModel", FakeControlNetModel)
    monkeypatch.setattr(pipeline_module, "StableDiffusionControlNetInpaintPipeline", FakePipe)
    monkeypatch.setattr(pipeline_module, "UniPCMultistepScheduler", FakeScheduler)
    monkeypatch.setattr(pipeline_module, "AutoencoderTiny", FakeTAESD)


def test_pipeline_init_cpu_path(monkeypatch):
    _patch_backends(monkeypatch)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    pipe = pipeline_module.DiffusionPipeline(_cfg(use_gpu=True))

    assert pipe._use_gpu is False
    assert pipe._pipe.enable_seq_called is False
    assert pipe._pipe.safety_checker is None
    assert pipe._pipe.enable_vae_called is True
    assert pipe._pipe.enable_attn_called is True
    assert pipe._pipe.scheduler["source"] == "unipc"
    assert pipe.taesd.eval_called is True
    assert str(pipe.taesd_device) == "cpu"


def test_pipeline_init_gpu_path_enables_offload_and_moves_taesd(monkeypatch):
    _patch_backends(monkeypatch)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda _idx: "Fake GPU")
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: None)

    pipe = pipeline_module.DiffusionPipeline(_cfg(use_gpu=True))

    assert pipe._use_gpu is True
    assert pipe._pipe.enable_seq_called is True
    assert pipe.taesd.to_device == "cuda"


def test_run_inpaint_uses_callback_on_step_end_when_supported(monkeypatch):
    _patch_backends(monkeypatch)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    pipe = pipeline_module.DiffusionPipeline(_cfg(use_gpu=False))
    calls = []

    def step_callback(step_index, total_steps, latents):
        calls.append((step_index, total_steps, latents))

    result = pipe.run_inpaint(
        init_image=Image.new("RGB", (16, 16), (0, 0, 0)),
        mask_image=Image.new("L", (16, 16), 0),
        control_image=Image.new("RGB", (16, 16), (0, 0, 0)),
        prompt="p",
        num_inference_steps=5,
        width=16,
        height=16,
        step_callback=step_callback,
    )

    assert result.size == (16, 16)
    assert pipe._pipe.last_kwargs["callback_on_step_end_tensor_inputs"] == ["latents"]
    assert calls == [(1, 5, "L")]


def test_run_inpaint_falls_back_to_deprecated_callback_api(monkeypatch):
    _patch_backends(monkeypatch)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    pipe = pipeline_module.DiffusionPipeline(_cfg(use_gpu=False))
    pipe._pipe.fail_new_callback_once = True

    result = pipe.run_inpaint(
        init_image=Image.new("RGB", (8, 8), (0, 0, 0)),
        mask_image=Image.new("L", (8, 8), 0),
        control_image=Image.new("RGB", (8, 8), (0, 0, 0)),
        prompt="p",
        num_inference_steps=3,
        width=8,
        height=8,
        step_callback=lambda *_: None,
    )

    assert result.size == (8, 8)
    assert pipe._pipe.call_count == 2
    assert "callback" in pipe._pipe.last_kwargs
    assert pipe._pipe.last_kwargs["callback_steps"] == 1
    assert "callback_on_step_end" not in pipe._pipe.last_kwargs
