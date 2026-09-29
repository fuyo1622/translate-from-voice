"""Probe what this machine can run: GPUs and their memory, CPU, RAM, power."""

from __future__ import annotations

import functools
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

_VENDOR_IDS = {"10de": "nvidia", "1002": "amd", "1022": "amd", "8086": "intel", "106b": "apple"}


@dataclass
class GPU:
    name: str
    vendor: str
    """"nvidia", "amd", "intel", "apple" or "other"."""
    vram_mb: int
    free_mb: int | None = None
    """Known for NVIDIA (nvidia-smi); None elsewhere."""

    @property
    def integrated(self) -> bool:
        if self.vendor == "apple":
            return False  # unified memory is the whole point on Apple Silicon
        intel_igpu = self.vendor == "intel" and "arc" not in self.name.lower()
        return intel_igpu or self.vram_mb < 1536


@dataclass
class Hardware:
    os: str
    """"windows", "macos" or "linux"."""
    arch: str
    cpu: str
    physical_cores: int
    logical_cores: int
    ram_mb: int
    ram_free_mb: int
    gpus: list[GPU] = field(default_factory=list)
    cuda: bool = False
    """CTranslate2 (Whisper) can use CUDA here: NVIDIA driver plus cuBLAS/cuDNN."""
    cuda_note: str = ""
    on_battery: bool | None = None

    @property
    def gpu(self) -> GPU | None:
        """The discrete (or Apple) GPU with the most memory, if any."""
        usable = [g for g in self.gpus if not g.integrated]
        return max(usable, key=lambda g: g.vram_mb, default=None)

    def summary(self) -> str:
        gpu = self.gpu
        parts = [
            f"{self.cpu} ({self.physical_cores} cores)",
            f"{self.ram_mb / 1024:.0f} GB RAM",
            f"{gpu.name} ({gpu.vram_mb / 1024:.0f} GB)" if gpu else "no discrete GPU",
        ]
        if gpu and gpu.vendor == "nvidia":
            parts.append("CUDA ok" if self.cuda else f"CUDA unavailable: {self.cuda_note}")
        if self.on_battery:
            parts.append("on battery")
        return ", ".join(parts)


@functools.lru_cache(maxsize=1)
def probe() -> Hardware:
    import psutil

    from tfv import gpu as cuda

    vm = psutil.virtual_memory()
    battery = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    os_name = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    hw = Hardware(
        os=os_name,
        arch=platform.machine().lower(),
        cpu=_cpu_name(),
        physical_cores=psutil.cpu_count(logical=False) or 1,
        logical_cores=psutil.cpu_count() or 1,
        ram_mb=vm.total // 2**20,
        ram_free_mb=vm.available // 2**20,
        gpus=_gpus(os_name, vm.total // 2**20),
        on_battery=None if battery is None else not battery.power_plugged,
    )
    if any(g.vendor == "nvidia" for g in hw.gpus):
        hw.cuda, hw.cuda_note = cuda.cuda_status()
    log.debug("Hardware: %s", hw)
    return hw


def _cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            if m := re.search(r"^model name\s*:\s*(.+)$", f.read(), re.MULTILINE):
                return m[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown CPU"


def _gpus(os_name: str, ram_mb: int) -> list[GPU]:
    if os_name == "macos":
        if platform.machine() == "arm64":
            # Metal can use roughly two thirds of unified memory by default
            return [GPU("Apple Silicon GPU", "apple", ram_mb * 2 // 3)]
        return []
    gpus = _windows_gpus() if os_name == "windows" else _linux_gpus()
    nvidia = _nvidia_smi()
    for g in gpus:
        if g.vendor == "nvidia" and nvidia:
            name, total, free = nvidia.pop(0)
            g.vram_mb, g.free_mb = total, free
    return gpus


def _windows_gpus() -> list[GPU]:
    import winreg

    gpus = []
    root = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root) as k:
        for i in range(winreg.QueryInfoKey(k)[0]):
            try:
                with winreg.OpenKey(k, winreg.EnumKey(k, i)) as sub:
                    values = {}
                    for name in ("DriverDesc", "MatchingDeviceId", "HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                        try:
                            values[name] = winreg.QueryValueEx(sub, name)[0]
                        except OSError:
                            pass
            except OSError:  # "Properties" and similar keys are not readable
                continue
            memory = values.get("HardwareInformation.qwMemorySize") or values.get("HardwareInformation.MemorySize")
            if isinstance(memory, bytes):
                memory = int.from_bytes(memory[:8], "little")
            vendor_id = re.search(r"ven_([0-9a-f]{4})", str(values.get("MatchingDeviceId", "")).lower())
            if not memory or not vendor_id or "DriverDesc" not in values:
                continue  # virtual or basic display adapters
            gpus.append(GPU(values["DriverDesc"], _VENDOR_IDS.get(vendor_id[1], "other"), int(memory) // 2**20))
    return gpus


def _linux_gpus() -> list[GPU]:
    gpus = []
    for dev in sorted(os.listdir("/sys/class/drm")) if os.path.isdir("/sys/class/drm") else []:
        base = f"/sys/class/drm/{dev}/device"
        if not re.fullmatch(r"card\d+", dev) or not os.path.exists(f"{base}/vendor"):
            continue
        with open(f"{base}/vendor") as f:
            vendor = _VENDOR_IDS.get(f.read().strip().removeprefix("0x"), "other")
        vram = 0
        if os.path.exists(f"{base}/mem_info_vram_total"):  # amdgpu
            with open(f"{base}/mem_info_vram_total") as f:
                vram = int(f.read()) // 2**20
        gpus.append(GPU(f"{vendor} GPU ({dev})", vendor, vram))
    return gpus


def nvidia_free_mb() -> int | None:
    """Free memory on the first NVIDIA GPU right now (not cached), if there is one."""
    rows = _nvidia_smi()
    return rows[0][2] if rows else None


def _nvidia_smi() -> list[tuple[str, int, int]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout  # fmt: skip
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.strip().splitlines():
        name, total, free = (p.strip() for p in line.rsplit(",", 2))
        rows.append((name, int(total), int(free)))
    return rows
