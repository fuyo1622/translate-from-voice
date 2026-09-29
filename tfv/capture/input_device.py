"""Capture from an input device (microphone or virtual cable) on any OS, via sounddevice.

Uses: translating someone speaking in the room; system audio on macOS through
a loopback driver such as BlackHole; system audio on Linux through a
PulseAudio/PipeWire monitor source.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import soxr

from tfv import SAMPLE_RATE
from tfv.capture.base import AudioCallback, AudioSource

log = logging.getLogger(__name__)


def _host_api() -> tuple[int | None, int]:
    """(host API to list, its default input); Windows lists every device once per API, so keep WASAPI."""
    import sys

    import sounddevice as sd

    if sys.platform == "win32":
        for i, api in enumerate(sd.query_hostapis()):
            if "WASAPI" in api["name"]:
                return i, api["default_input_device"]
    return None, sd.default.device[0]


def list_input_devices() -> list[dict]:
    import sounddevice as sd

    api, default = _host_api()
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"], "rate": int(d["default_samplerate"]), "default": i == default}
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0 and (api is None or d["hostapi"] == api)
    ]


class InputDeviceSource(AudioSource):
    """`device` is an index, a name substring, or None for the default input."""

    def __init__(self, device: str | None = None, block_ms: int = 100):
        self._query = device
        self._block_ms = block_ms
        self._stream = None

    def _find_device(self) -> dict:
        import sounddevice as sd

        if self._query is None:
            _, default = _host_api()
            return {**sd.query_devices(default), "index": default}
        devices = list_input_devices()
        query = self._query.lower()
        matches = [d for d in devices if query == str(d["index"])] or [d for d in devices if query in d["name"].lower()]
        if not matches:
            raise ValueError(f"no input device matches {self._query!r}; list them with `python -m tfv devices --inputs`")
        chosen = matches[0]
        return {**sd.query_devices(chosen["index"]), "index": chosen["index"]}

    def start(self, on_audio: AudioCallback, on_end: Callable[[], None]) -> None:
        import sounddevice as sd

        dev = self._find_device()
        rate = int(dev["default_samplerate"])
        channels = min(2, int(dev["max_input_channels"]))
        self.description = f"input {dev['name']} ({rate} Hz)"
        resampler = soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="float32")

        def callback(indata, frames, time_info, status):
            try:
                mono = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]
                out = resampler.resample_chunk(np.ascontiguousarray(mono, dtype=np.float32))
                if len(out):
                    on_audio(out)
            except Exception:
                log.exception("audio callback failed")

        self._stream = sd.InputStream(
            device=dev["index"], samplerate=rate, channels=channels, dtype="float32",
            blocksize=rate * self._block_ms // 1000, callback=callback,
        )  # fmt: skip
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
