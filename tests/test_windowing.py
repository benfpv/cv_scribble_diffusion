"""Tests for native OpenCV window management helpers."""

import cv2

from cv_scribble_diffusion.ui import windowing


class _FakeUser32:
    def __init__(self, hwnd=1234):
        self.hwnd = hwnd
        self.styles = []
        self.positions = []
        self.metrics = {0: 1920, 1: 1080}
        self.ex_style = windowing.WS_EX_WINDOWEDGE | windowing.WS_EX_CLIENTEDGE | 0x40

    def FindWindowW(self, _class_name, window_name):
        self.window_name = window_name
        return self.hwnd

    def GetSystemMetrics(self, index):
        return self.metrics[index]

    def GetWindowLongPtrW(self, hwnd, index):
        assert hwnd == self.hwnd
        assert index == windowing.GWL_EXSTYLE
        return self.ex_style

    def SetWindowLongPtrW(self, hwnd, index, value):
        assert hwnd == self.hwnd
        self.styles.append((index, value))

    def SetWindowPos(self, hwnd, insert_after, x, y, width, height, flags):
        assert hwnd == self.hwnd
        self.positions.append((insert_after, x, y, width, height, flags))


class _FakeDwmApi:
    def __init__(self):
        self.calls = []

    def DwmSetWindowAttribute(self, hwnd, attr, _policy, size):
        self.calls.append((hwnd, attr, size))


def test_center_position_never_negative():
    assert windowing.center_position((500, 300), (1920, 1080)) == (710, 390)
    assert windowing.center_position((2000, 1200), (1920, 1080)) == (0, 0)


def test_screen_size_uses_win32_metrics():
    assert windowing.screen_size(_FakeUser32()) == (1920, 1080)


def test_center_window_moves_to_center(monkeypatch):
    moved = {}
    monkeypatch.setattr(windowing, "screen_size", lambda: (1000, 800))
    monkeypatch.setattr(windowing.cv2, "moveWindow", lambda name, x, y: moved.update(name=name, x=x, y=y))

    assert windowing.center_window("Scribble Diffusion", (400, 300)) == (300, 250)
    assert moved == {"name": "Scribble Diffusion", "x": 300, "y": 250}


def test_center_window_handles_backend_move_errors(monkeypatch):
    monkeypatch.setattr(windowing, "screen_size", lambda: (1000, 800))

    def fail_move(*_args):
        raise cv2.error("move failed")

    monkeypatch.setattr(windowing.cv2, "moveWindow", fail_move)
    assert windowing.center_window("Scribble Diffusion", (400, 300)) is None


def test_apply_borderless_window_sets_popup_style_and_size(monkeypatch):
    fake_user32 = _FakeUser32()
    fake_dwmapi = _FakeDwmApi()
    shown = {}
    monkeypatch.setattr(windowing.cv2, "imshow", lambda name, frame: shown.update(name=name, shape=frame.shape))
    monkeypatch.setattr(windowing.cv2, "waitKeyEx", lambda _delay: -1)

    applied = windowing.apply_borderless_window(
        "Scribble Diffusion", (568, 638), user32=fake_user32, dwmapi=fake_dwmapi,
    )

    assert applied is True
    assert shown == {"name": "Scribble Diffusion", "shape": (638, 568, 3)}
    assert (windowing.GWL_STYLE, windowing.WS_POPUP | windowing.WS_VISIBLE) in fake_user32.styles
    assert (windowing.GWL_EXSTYLE, 0x40) in fake_user32.styles
    assert fake_user32.positions == [(
        None, 0, 0, 568, 638,
        windowing.SWP_NOMOVE | windowing.SWP_NOZORDER | windowing.SWP_FRAMECHANGED,
    )]
    assert fake_dwmapi.calls[0][0] == fake_user32.hwnd
    assert fake_dwmapi.calls[0][1] == windowing.DWMWA_NCRENDERING_POLICY


def test_apply_borderless_window_returns_false_without_hwnd(monkeypatch):
    fake_user32 = _FakeUser32(hwnd=0)
    monkeypatch.setattr(windowing.cv2, "imshow", lambda *_args: None)
    monkeypatch.setattr(windowing.cv2, "waitKeyEx", lambda _delay: -1)

    assert windowing.apply_borderless_window("Missing", (568, 638), user32=fake_user32) is False


def test_create_app_window_uses_borderless_then_centers(monkeypatch):
    calls = []
    monkeypatch.setattr(windowing.cv2, "namedWindow", lambda name, flag: calls.append(("named", name, flag)))
    monkeypatch.setattr(
        windowing, "apply_borderless_window",
        lambda name, size: calls.append(("borderless", name, size)) or True,
    )
    monkeypatch.setattr(windowing, "center_window", lambda name, size: calls.append(("center", name, size)))

    assert windowing.create_app_window("Scribble Diffusion", (568, 638), borderless=True) is True
    assert calls == [
        ("named", "Scribble Diffusion", cv2.WINDOW_AUTOSIZE),
        ("borderless", "Scribble Diffusion", (568, 638)),
        ("center", "Scribble Diffusion", (568, 638)),
    ]


def test_create_app_window_can_skip_borderless(monkeypatch):
    calls = []
    monkeypatch.setattr(windowing.cv2, "namedWindow", lambda name, flag: calls.append(("named", name, flag)))
    monkeypatch.setattr(windowing, "apply_borderless_window", lambda *_args: calls.append(("borderless",)))
    monkeypatch.setattr(windowing, "center_window", lambda name, size: calls.append(("center", name, size)))

    assert windowing.create_app_window("Scribble Diffusion", (568, 638), borderless=False) is False
    assert calls == [
        ("named", "Scribble Diffusion", cv2.WINDOW_AUTOSIZE),
        ("center", "Scribble Diffusion", (568, 638)),
    ]
