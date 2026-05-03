"""OpenCV window creation, placement, and native styling helpers."""

import ctypes
from typing import Optional, Tuple

import cv2
import numpy as np


GWL_STYLE = -16
GWL_EXSTYLE = -20
WS_POPUP = 0x80000000
WS_VISIBLE = 0x10000000
WS_EX_WINDOWEDGE = 0x00000100
WS_EX_CLIENTEDGE = 0x00000200
SWP_NOMOVE = 0x0002
SWP_NOZORDER = 0x0004
SWP_FRAMECHANGED = 0x0020
DWMWA_NCRENDERING_POLICY = 2
DWMNCRP_DISABLED = 1


def create_app_window(window_name: str, window_size: Tuple[int, int], borderless: bool = True) -> bool:
    """Create the OpenCV app window and return whether borderless styling was applied."""
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    applied = False
    if borderless:
        applied = apply_borderless_window(window_name, window_size)
    center_window(window_name, window_size)
    return applied


def center_window(window_name: str, window_size: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Best-effort centering for an OpenCV window on the primary display."""
    screen = screen_size()
    if screen is None:
        return None
    x, y = center_position(window_size, screen)
    try:
        cv2.moveWindow(window_name, x, y)
    except cv2.error:
        return None
    return (x, y)


def center_position(window_size: Tuple[int, int], display_size: Tuple[int, int]) -> Tuple[int, int]:
    """Return a non-negative centered top-left position."""
    ww, wh = window_size
    sw, sh = display_size
    return (max(0, (int(sw) - int(ww)) // 2), max(0, (int(sh) - int(wh)) // 2))


def screen_size(user32=None) -> Optional[Tuple[int, int]]:
    """Return primary display size via Win32 metrics when available."""
    try:
        user32 = user32 or ctypes.windll.user32
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    return (width, height)


def apply_borderless_window(
    window_name: str,
    window_size: Tuple[int, int],
    user32=None,
    dwmapi=None,
) -> bool:
    """Apply borderless Win32 popup styling to an existing OpenCV window."""
    try:
        user32 = user32 or ctypes.windll.user32
    except Exception:
        return False

    width, height = int(window_size[0]), int(window_size[1])
    if width <= 0 or height <= 0:
        return False

    _prime_window(window_name, width, height)
    hwnd = int(user32.FindWindowW(None, window_name) or 0)
    if hwnd == 0:
        return False

    _set_window_long(user32, hwnd, GWL_STYLE, WS_POPUP | WS_VISIBLE)
    ex_style = _get_window_long(user32, hwnd, GWL_EXSTYLE)
    _set_window_long(user32, hwnd, GWL_EXSTYLE, ex_style & ~(WS_EX_WINDOWEDGE | WS_EX_CLIENTEDGE))
    user32.SetWindowPos(
        hwnd, None, 0, 0, width, height,
        SWP_NOMOVE | SWP_NOZORDER | SWP_FRAMECHANGED,
    )
    _disable_dwm_shadow(hwnd, dwmapi)
    return True


def _prime_window(window_name: str, width: int, height: int):
    """Show a blank frame so Win32 can find the native OpenCV window handle."""
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.imshow(window_name, blank)
    cv2.waitKeyEx(1)


def _get_window_long(user32, hwnd: int, index: int) -> int:
    getter = getattr(user32, "GetWindowLongPtrW", None) or user32.GetWindowLongW
    return int(getter(hwnd, index))


def _set_window_long(user32, hwnd: int, index: int, value: int):
    setter = getattr(user32, "SetWindowLongPtrW", None) or user32.SetWindowLongW
    setter(hwnd, index, value)


def _disable_dwm_shadow(hwnd: int, dwmapi=None):
    try:
        dwmapi = dwmapi or ctypes.windll.dwmapi
        policy = ctypes.c_int(DWMNCRP_DISABLED)
        dwmapi.DwmSetWindowAttribute(
            hwnd, DWMWA_NCRENDERING_POLICY,
            ctypes.byref(policy), ctypes.sizeof(policy),
        )
    except Exception:
        return
