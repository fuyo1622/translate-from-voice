"""Run llama.cpp's llama-server as a helper process for local LLM translation.

The official prebuilt binaries are downloaded on first use. The Vulkan build
runs on NVIDIA, AMD and Intel GPUs (and on the CPU with no GPU at all); Apple
Silicon uses the Metal build. `--fit` lets llama.cpp decide how many layers fit
in the VRAM that is left after speech recognition has loaded.
"""

from __future__ import annotations

import collections
import logging
import os
import platform
import re
import secrets
import socket
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from tfv import models

log = logging.getLogger(__name__)

BUILD = "b11255"
_RELEASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{BUILD}/"

# variant -> [(asset, sha256)]; checksums from the GitHub release.
RUNTIMES: dict[str, list[tuple[str, str]]] = {
    "win-vulkan": [(f"llama-{BUILD}-bin-win-vulkan-x64.zip", "3dae91806111ac4a464fb5a6db45188ba870a7e89bfa0a203750115fcef20a31")],
    "win-cpu": [(f"llama-{BUILD}-bin-win-cpu-x64.zip", "13bef0d15c315b153603a610a6652e2460ae9202bb1b22832eed225ccbc7fcc7")],
    "win-cuda": [
        (f"llama-{BUILD}-bin-win-cuda-12.4-x64.zip", "62b356610e08112efbf13b6b55c1dc9ba66c5fba61bd99464be154f4ccf3fa65"),
        ("cudart-llama-bin-win-cuda-12.4-x64.zip", "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
    ],
    "macos-metal": [(f"llama-{BUILD}-bin-macos-arm64.tar.gz", "108c641ea8a279e046faddc6f38b7b24736f83f70a3e33c71eafc8d7bebeb777")],
    "linux-vulkan": [(f"llama-{BUILD}-bin-ubuntu-vulkan-x64.tar.gz", "751edacb2ec7d2cf4cfe2ecb5e931b71e1b4df6e8a199147fe303f803fa546a5")],
    "linux-cpu": [(f"llama-{BUILD}-bin-ubuntu-x64.tar.gz", "30733bba8d90fff5a54e26b1c75cb055d422c45595056349699a84eb362c4e92")],
}


def default_variant() -> str:
    """The runtime build that suits this OS (GPU use is decided per device later)."""
    if sys.platform == "win32":
        system32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        return "win-vulkan" if (system32 / "vulkan-1.dll").exists() else "win-cpu"
    if sys.platform == "darwin":
        if platform.machine() != "arm64":
            raise RuntimeError("local LLM translation needs an Apple Silicon Mac")
        return "macos-metal"
    import ctypes.util

    return "linux-vulkan" if ctypes.util.find_library("vulkan") else "linux-cpu"


def ensure_runtime(variant: str | None = None, progress: models.Progress | None = None) -> Path:
    """Path to llama-server, downloading and unpacking the build if needed."""
    variant = variant or default_variant()
    if variant not in RUNTIMES:
        raise ValueError(f"unknown llama.cpp build {variant!r}; choose from {', '.join(RUNTIMES)}")
    root = models.models_dir() / "runtime" / f"llama-{BUILD}-{variant}"
    marker = root / ".tfv-complete"
    if not marker.exists():
        log.info("Downloading llama.cpp %s (%s)", BUILD, variant)
        root.mkdir(parents=True, exist_ok=True)
        for asset, sha256 in RUNTIMES[variant]:
            archive = root / asset
            models.download(_RELEASE + asset, archive, sha256=sha256, progress=progress, label=f"llama.cpp {variant}")
            if asset.endswith(".zip"):
                with zipfile.ZipFile(archive) as z:
                    z.extractall(root)
            else:
                with tarfile.open(archive) as t:
                    # the "data" filter blocks path tricks; older Pythons lack it
                    t.extractall(root, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
            archive.unlink()
        marker.write_text("ok")
    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    found = sorted(root.rglob(exe_name))
    if not found:
        raise RuntimeError(f"llama-server not found in {root}")
    return found[0]


@dataclass
class GpuDevice:
    id: str
    """llama.cpp device name, e.g. "Vulkan1" or "CUDA0"."""
    name: str
    total_mib: int
    free_mib: int

    @property
    def integrated(self) -> bool:
        # Intel's Iris/UHD iGPUs report shared system RAM as their memory
        return "intel" in self.name.lower() and "arc" not in self.name.lower()

    @property
    def software(self) -> bool:
        """Vulkan run on the CPU (Mesa's llvmpipe, SwiftShader): slower than llama.cpp's own CPU code,
        and it reports all of the RAM as free memory, so it would otherwise win over a real GPU."""
        return any(s in self.name.lower() for s in ("llvmpipe", "lavapipe", "swiftshader"))


def list_devices(exe: Path) -> list[GpuDevice]:
    out = subprocess.run(
        [str(exe), "--list-devices"], capture_output=True, text=True, timeout=60, creationflags=_NO_WINDOW
    ).stdout
    pattern = re.compile(r"^\s*(\S+): (.+?) \((\d+) MiB, (\d+) MiB free\)", re.MULTILINE)
    return [GpuDevice(m[1], m[2], int(m[3]), int(m[4])) for m in pattern.finditer(out)]


def best_device(devices: list[GpuDevice]) -> GpuDevice | None:
    """The discrete GPU with the most free memory, or None (integrated and software devices don't count)."""
    discrete = [d for d in devices if not d.integrated and not d.software]
    return max(discrete, key=lambda d: d.free_mib, default=None)


_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class LlamaServer:
    """One llama-server process serving one model on a free localhost port."""

    def __init__(
        self,
        model_path: Path,
        variant: str | None = None,
        device: str = "auto",
        ctx_size: int = 4096,
        threads: int | None = None,
        vram_margin_mib: int = 512,
        extra_args: list[str] | None = None,
    ):
        self.model_path = Path(model_path)
        self.extra_args = extra_args or []
        self.variant = variant or default_variant()
        self.device = device
        """"auto" (best GPU), "none" (CPU only), or a llama.cpp device id like "Vulkan1"."""
        self.ctx_size = ctx_size
        self.threads = threads
        self.vram_margin_mib = vram_margin_mib
        self.port = 0
        # llama-server accepts requests from any local program or web page unless it has a key
        self.api_key = secrets.token_urlsafe(24)
        self._proc: subprocess.Popen | None = None
        self._log: collections.deque[str] = collections.deque(maxlen=200)
        self.gpu: GpuDevice | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, timeout: float = 300) -> None:
        import httpx

        exe = ensure_runtime(self.variant)
        device = self.device
        if device == "auto":
            self.gpu = best_device(list_devices(exe)) if not self.variant.endswith("cpu") else None
            device = self.gpu.id if self.gpu else "none"
        self.port = _free_port()
        self._log.clear()
        cmd = [
            str(exe), "-m", str(self.model_path),
            "--host", "127.0.0.1", "--port", str(self.port),
            "-c", str(self.ctx_size), "-np", "1",
            "--no-webui", "--reasoning", "off", "--offline",
            "--api-key", self.api_key,
            "-dev", device,
            # Captions are short, independent requests: skip the RAM prompt cache (8 GB by
            # default) and keep few checkpoints. Hybrid models such as Qwen3.5 store large
            # states there, and a failed allocation crashes the server on 16 GB machines.
            "--cache-ram", "0", "--ctx-checkpoints", "4",
            *self.extra_args,
        ]  # fmt: skip
        if device != "none":
            cmd += ["--fit", "on", "--fit-target", str(self.vram_margin_mib)]
        if self.threads:
            cmd += ["-t", str(self.threads)]
        log.debug("Starting %s", " ".join(cmd).replace(self.api_key, "***"))
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            creationflags=_NO_WINDOW,
        )  # fmt: skip
        _kill_with_parent(self._proc)
        threading.Thread(target=self._read_log, args=(self._proc,), name="tfv-llama-log", daemon=True).start()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("llama-server exited while loading:\n" + "\n".join(list(self._log)[-15:]))
            try:
                if httpx.get(self.url + "/health", timeout=2).status_code == 200:
                    log.info("llama-server ready on %s (%s)", device, self.model_path.name)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"llama-server did not become ready within {timeout:.0f}s")

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def log_tail(self, lines: int = 15) -> str:
        return "\n".join(list(self._log)[-lines:])

    def restart(self) -> None:
        """Start again (on a new port) after the process died."""
        self.stop()
        self.start()

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def _read_log(self, proc: subprocess.Popen) -> None:
        """Keep the recent output in memory and all of it in llama-server.log, for crash reports."""
        path = models.models_dir().parent / "llama-server.log"
        try:
            if path.exists() and path.stat().st_size > 5_000_000:
                path.replace(path.with_name("llama-server.old.log"))
            sink = open(path, "a", encoding="utf-8")
        except OSError:
            sink = None
        with sink or open(os.devnull, "w", encoding="utf-8") as out:
            out.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {self.model_path.name} ====\n")
            for line in proc.stdout:
                self._log.append(line.rstrip())
                out.write(line)
                out.flush()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_job = None


def _kill_with_parent(proc: subprocess.Popen) -> None:
    """Windows: put the helper in a job object that dies with this process, so a crash
    or a killed terminal never leaves llama-server holding GPU memory."""
    global _job
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if _job is None:

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", ctypes.c_uint64 * 6),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        job = kernel32.CreateJobObjectW(None, None)
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            log.debug("SetInformationJobObject failed: %d", ctypes.get_last_error())
            return
        _job = job
    if not kernel32.AssignProcessToJobObject(wintypes.HANDLE(_job), wintypes.HANDLE(proc._handle)):
        log.debug("AssignProcessToJobObject failed: %d", ctypes.get_last_error())
