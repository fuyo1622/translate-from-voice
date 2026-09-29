"""Ask Windows which output device is the default, via the Core Audio COM API (ctypes only).

PortAudio caches its device list, so it cannot notice that the user plugged in
earphones; this query is cheap enough to poll every couple of seconds.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

_CLSID_MMDeviceEnumerator = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
_IID_IMMDeviceEnumerator = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"
_CLSCTX_ALL = 0x17
_E_RENDER, _E_CONSOLE = 0, 0


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _guid(text: str) -> _GUID:
    g = _GUID()
    ctypes.oledll.ole32.CLSIDFromString(ctypes.c_wchar_p(text), ctypes.byref(g))
    return g


def _method(obj: ctypes.c_void_p, index: int, *argtypes):
    """Callable for the COM method at vtable `index` of `obj`."""
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)
    return lambda *args: prototype(vtable[index])(obj, *args)


def _release(obj: ctypes.c_void_p) -> None:
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)(vtable[2])(obj)


def default_output_id() -> str | None:
    """Endpoint id of the default playback device, or None if there is none.

    COM must be initialized on the calling thread (see `com_thread`).
    """
    enumerator = ctypes.c_void_p()
    ctypes.oledll.ole32.CoCreateInstance(
        ctypes.byref(_guid(_CLSID_MMDeviceEnumerator)), None, _CLSCTX_ALL,
        ctypes.byref(_guid(_IID_IMMDeviceEnumerator)), ctypes.byref(enumerator),
    )  # fmt: skip
    device = ctypes.c_void_p()
    try:
        # IMMDeviceEnumerator::GetDefaultAudioEndpoint is vtable slot 4
        _method(enumerator, 4, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p))(_E_RENDER, _E_CONSOLE, ctypes.byref(device))
    except OSError:  # E_NOTFOUND: no output device at all
        return None
    finally:
        _release(enumerator)
    try:
        text = ctypes.c_wchar_p()
        _method(device, 5, ctypes.POINTER(ctypes.c_wchar_p))(ctypes.byref(text))  # IMMDevice::GetId
        value = text.value
        ctypes.windll.ole32.CoTaskMemFree(text)
        return value
    finally:
        _release(device)


class com_thread:
    """Context manager initializing COM (multithreaded) for the current thread."""

    def __enter__(self):
        ctypes.windll.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
        return self

    def __exit__(self, *exc):
        ctypes.windll.ole32.CoUninitialize()
