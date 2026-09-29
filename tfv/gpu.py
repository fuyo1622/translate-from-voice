"""Find out whether faster-whisper (CTranslate2) can use an NVIDIA GPU here.

CTranslate2 loads cuBLAS, and cuDNN loads its sub-libraries, lazily by bare
DLL name. A missing cuDNN sub-library aborts the whole process instead of
raising, so we check for every library before choosing CUDA.
"""

from __future__ import annotations

import ctypes
import functools
import importlib.util
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

_WINDOWS_CUDA_DLLS = (
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
)

_dll_dir_handles = []  # os.add_dll_directory handles must stay alive


def add_nvidia_wheel_dirs() -> list[str]:
    """Make DLLs from pip's nvidia-*-cu12 wheels loadable (Windows only).

    Lazy loads go through the normal LoadLibrary search, which uses PATH, so
    the directories are prepended to PATH as well as registered with
    os.add_dll_directory.
    """
    if sys.platform != "win32":
        return []
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return []
    dirs = [
        str(bin_dir)
        for root in spec.submodule_search_locations
        for bin_dir in sorted(Path(root).glob("*/bin"))
        if any(bin_dir.glob("*.dll"))
    ]
    path = os.environ.get("PATH", "").split(os.pathsep)
    for d in dirs:
        _dll_dir_handles.append(os.add_dll_directory(d))
    os.environ["PATH"] = os.pathsep.join(dirs + [p for p in path if p not in dirs])
    return dirs


@functools.lru_cache(maxsize=1)
def cuda_status() -> tuple[bool, str]:
    """(usable, reason) for running CTranslate2 on CUDA."""
    add_nvidia_wheel_dirs()
    try:
        import ctranslate2
    except ImportError:
        return False, "ctranslate2 is not installed"
    try:
        if ctranslate2.get_cuda_device_count() < 1:
            return False, "no NVIDIA GPU or driver found"
    except Exception as exc:  # broken driver installs raise here
        return False, f"CUDA driver error: {exc}"
    if sys.platform == "win32":
        missing = []
        for name in _WINDOWS_CUDA_DLLS:
            try:
                ctypes.WinDLL(name, winmode=0)  # winmode=0: search PATH like LoadLibrary does
            except OSError:
                missing.append(name)
        if missing:
            return False, f"missing {', '.join(missing)} (install with: pip install -e .[nvidia])"
    return True, "ok"
