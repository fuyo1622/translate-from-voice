"""System-wide hotkeys on Windows (RegisterHotKey); a no-op elsewhere.

Hotkeys are bound to a hidden helper window, so they keep working when the
overlay window is recreated (it is whenever it is locked or unlocked).
"""

from __future__ import annotations

import ctypes
import logging
import sys
from typing import Callable

from PySide6.QtCore import QAbstractNativeEventFilter, QCoreApplication
from PySide6.QtWidgets import QWidget

log = logging.getLogger(__name__)

_WM_HOTKEY = 0x0312
_MOD_ALT, _MOD_CONTROL, _MOD_NOREPEAT = 0x0001, 0x0002, 0x4000

# Ctrl+Alt+<key>
HOTKEYS = {
    "P": "pause or resume",
    "L": "lock or unlock the overlay",
    "O": "show or hide the original text",
    "H": "hide or show the overlay",
    "S": "open the settings",
    "=": "bigger text",
    "-": "smaller text",
    "K": "save the last minute of audio and captions (for feedback)",
}

_VIRTUAL_KEYS = {"=": 0xBB, "-": 0xBD}  # VK_OEM_PLUS (the =/+ key), VK_OEM_MINUS; letters are their own codes


class _Filter(QAbstractNativeEventFilter):
    def __init__(self, handlers: dict[int, Callable[[], None]]):
        super().__init__()
        self._handlers = handlers

    def nativeEventFilter(self, event_type, message):
        if event_type == b"windows_generic_MSG":
            from ctypes import wintypes

            msg = wintypes.MSG.from_address(int(message))
            if msg.message == _WM_HOTKEY and msg.wParam in self._handlers:
                self._handlers[msg.wParam]()
                return True, 0
        return False, 0


class GlobalHotkeys:
    def __init__(self):
        self._window: QWidget | None = None
        self._filter: _Filter | None = None
        self._handlers: dict[int, Callable[[], None]] = {}
        self.registered: list[str] = []

    def register(self, bindings: dict[str, Callable[[], None]]) -> None:
        """bindings: key (a letter, "=" or "-") -> callback, each bound to Ctrl+Alt+<key>."""
        if sys.platform != "win32":
            return
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._window = QWidget()
        self._window.setWindowTitle("tfv hotkeys")
        hwnd = int(self._window.winId())  # creates the native window without showing it
        for i, (key, callback) in enumerate(bindings.items(), start=1):
            if user32.RegisterHotKey(hwnd, i, _MOD_CONTROL | _MOD_ALT | _MOD_NOREPEAT, _VIRTUAL_KEYS.get(key, ord(key.upper()))):
                self._handlers[i] = callback
                self.registered.append(f"Ctrl+Alt+{key.upper()}")
            elif (error := ctypes.get_last_error()) == 1409:  # ERROR_HOTKEY_ALREADY_REGISTERED
                log.warning("Ctrl+Alt+%s is taken by another program", key.upper())
            else:
                log.warning("Could not register Ctrl+Alt+%s (Windows error %d)", key.upper(), error)
        self._filter = _Filter(self._handlers)
        QCoreApplication.instance().installNativeEventFilter(self._filter)

    def unregister(self) -> None:
        if sys.platform != "win32" or self._window is None:
            return
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        hwnd = int(self._window.winId())
        for i in self._handlers:
            user32.UnregisterHotKey(hwnd, i)
        self._handlers.clear()
