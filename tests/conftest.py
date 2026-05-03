"""Shared pytest fixtures for tests that don't need GPUs or real models."""

import sys
import os
import time

# Make the project root importable from any test file
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
from PIL import Image


# -- TAESD / pipeline stand-ins ---------------------------------------------

class _DecodeResult:
    def __init__(self, sample):
        self.sample = sample


class DummyTAESD:
    """Minimal TAESD stand-in that returns the input tensor unchanged."""

    def decode(self, latents_tensor):
        return _DecodeResult(latents_tensor.clamp(0, 1))

    def eval(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def parameters(self):
        return iter([torch.tensor([1.0])])


class MockPipeline:
    """Stand-in for DiffusionPipeline that produces dummy images without GPU."""

    def __init__(self, cfg):
        self.cfg = cfg
        self._taesd = DummyTAESD()
        self.prompts = []

    @property
    def taesd(self):
        return self._taesd

    @property
    def taesd_device(self):
        return torch.device("cpu")

    def run_inpaint(self, init_image, mask_image, control_image,
                    prompt, num_inference_steps, width, height,
                    step_callback=None):
        self.prompts.append(prompt)
        for i in range(num_inference_steps):
            if step_callback:
                latents = torch.rand(1, 3, height // 8, width // 8)
                step_callback(i, num_inference_steps, latents)
        return Image.new("RGB", (width, height), (128, 128, 128))


class FailingPipeline(MockPipeline):
    """Pipeline that raises on the first call to run_inpaint, then succeeds."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._fail_count = 1

    def run_inpaint(self, *args, **kwargs):
        if self._fail_count > 0:
            self._fail_count -= 1
            raise RuntimeError("Simulated pipeline failure")
        return super().run_inpaint(*args, **kwargs)


class AlwaysFailingPipeline(MockPipeline):
    """Pipeline that always raises; used to exercise the backoff path."""

    def run_inpaint(self, *args, **kwargs):
        raise RuntimeError("Persistent simulated failure")


class SlowPipeline(MockPipeline):
    """Pipeline that sleeps to make reset-during-generation reproducible."""

    def run_inpaint(self, *args, **kwargs):
        time.sleep(0.2)
        width = kwargs.get("width", 64)
        height = kwargs.get("height", 64)
        return Image.new("RGB", (width, height), (255, 255, 255))


# -- pytest fixtures --------------------------------------------------------

@pytest.fixture
def dummy_taesd():
    return DummyTAESD()


@pytest.fixture
def mock_pipeline_cls():
    return MockPipeline


@pytest.fixture
def failing_pipeline_cls():
    return FailingPipeline


@pytest.fixture
def always_failing_pipeline_cls():
    return AlwaysFailingPipeline


@pytest.fixture
def slow_pipeline_cls():
    return SlowPipeline


@pytest.fixture
def patch_cv_window(monkeypatch):
    """No-op the OpenCV window calls so App() can be constructed headlessly."""
    monkeypatch.setattr("cv2.namedWindow", lambda *a, **kw: None)
    monkeypatch.setattr("cv2.imshow", lambda *a, **kw: None)
    monkeypatch.setattr("cv2.waitKeyEx", lambda *a, **kw: -1)
    monkeypatch.setattr("cv2.moveWindow", lambda *a, **kw: None)
    monkeypatch.setattr("cv2.setMouseCallback", lambda *a, **kw: None)
