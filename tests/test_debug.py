"""Tests for debug artifact writing helpers."""

import os

import numpy as np
from PIL import Image

from config import DebugConfig
from debug import DebugWriter, _write_image


def test_debug_writer_disabled_is_noop(tmp_path):
    writer = DebugWriter(DebugConfig(enabled=False, dir=str(tmp_path)))
    writer.save("disabled", Image.new("RGB", (8, 8), (0, 0, 0)), info="ignored")
    assert os.listdir(tmp_path) == []


def test_debug_writer_save_writes_png_and_info(tmp_path):
    writer = DebugWriter(DebugConfig(enabled=True, dir=str(tmp_path)))
    writer.save("sample", Image.new("RGB", (8, 8), (10, 20, 30)), info="hello")

    files = sorted(os.listdir(tmp_path))
    assert any(name.endswith("_sample.png") for name in files)
    txt_file = next(name for name in files if name.endswith("_sample.txt"))
    with open(tmp_path / txt_file, "r", encoding="utf-8") as fh:
        assert fh.read() == "hello"


def test_write_image_rgb_array_uses_rgb_to_bgr(monkeypatch, tmp_path):
    captured = {}

    def fake_rgb_to_bgr(arr):
        captured["converted"] = True
        return np.full_like(arr, 7)

    def fake_imwrite(path, arr):
        captured["path"] = path
        captured["array"] = arr.copy()
        return True

    monkeypatch.setattr("debug.rgb_to_bgr", fake_rgb_to_bgr)
    monkeypatch.setattr("debug.cv2.imwrite", fake_imwrite)

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    out_path = str(tmp_path / "rgb.png")
    _write_image(out_path, image)

    assert captured["converted"] is True
    assert captured["path"] == out_path
    assert np.all(captured["array"] == 7)


def test_save_annotated_crop_draws_green_outline(tmp_path, monkeypatch):
    writer = DebugWriter(DebugConfig(enabled=True, dir=str(tmp_path)))
    captured = {}

    def fake_save(tag, image, info=""):
        captured["tag"] = tag
        captured["image"] = np.array(image)
        captured["info"] = info

    monkeypatch.setattr(writer, "save", fake_save)

    canvas = Image.new("RGB", (32, 32), (0, 0, 0))
    writer.save_annotated_crop(canvas, (8, 8, 24, 24), info="crop")

    assert captured["tag"] == "crop_box"
    assert captured["info"] == "crop"
    # Top-left outline pixel is green.
    assert tuple(captured["image"][8, 8]) == (0, 255, 0)
    # Interior remains untouched black.
    assert tuple(captured["image"][16, 16]) == (0, 0, 0)
