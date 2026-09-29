"""Record what Windows is playing, via WASAPI loopback (no virtual cable needed)."""

from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np
import soxr

from tfv import SAMPLE_RATE
from tfv.capture.base import AudioCallback, AudioSource

log = logging.getLogger(__name__)


def list_output_devices() -> list[dict]:
    """Loopback-capable output devices: index, name, channels, rate, default."""
    import pyaudiowpatch as pyaudio

    pa = pyaudio.PyAudio()
    try:
        default = pa.get_default_wasapi_loopback()
        return [
            {
                "index": dev["index"],
                "name": dev["name"].removesuffix(" [Loopback]"),
                "channels": dev["maxInputChannels"],
                "rate": int(dev["defaultSampleRate"]),
                "default": default is not None and dev["index"] == default["index"],
            }
            for dev in pa.get_loopback_device_info_generator()
        ]
    finally:
        pa.terminate()


class LoopbackSource(AudioSource):
    """Captures an output device (speaker or earphone) through WASAPI loopback.

    `device` selects the output by index or name substring; None follows the
    Windows default output, including when it changes while running (e.g.
    earphones plugged in); `on_device_change` is then called with the new name.

    WASAPI delivers nothing while no application is playing sound, so
    consumers must not rely on a steady stream of chunks.
    """

    def __init__(self, device: str | None = None, block_ms: int = 100):
        self._query = device
        self._block_ms = block_ms
        self._pa = None
        self._stream = None
        self._on_audio: AudioCallback | None = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self.on_device_change: Callable[[str], None] | None = None

    def _find_device(self, pa) -> dict:
        if self._query is None:
            try:
                return pa.get_default_wasapi_loopback()
            except LookupError:
                raise RuntimeError("no default output device to capture; is a speaker or earphone connected?") from None
        query = self._query.lower()
        devices = list(pa.get_loopback_device_info_generator())
        matches = [d for d in devices if query == str(d["index"])] or [d for d in devices if query in d["name"].lower()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f"no output device matches {self._query!r}; list them with `python -m tfv devices`")
        names = ", ".join(f"[{d['index']}] {d['name'].removesuffix(' [Loopback]')}" for d in matches)
        raise ValueError(f"{self._query!r} matches several output devices: {names}; be more specific or use the index")

    def start(self, on_audio: AudioCallback, on_end: Callable[[], None]) -> None:
        self._on_audio = on_audio
        self._stopped.clear()
        with self._lock:
            self._open()
        if self._query is None:
            threading.Thread(target=self._follow_default, name="tfv-device-watch", daemon=True).start()

    def stop(self) -> None:
        self._stopped.set()
        with self._lock:
            self._close()

    def _open(self) -> None:
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()  # a fresh instance sees devices plugged in since the last one
        try:
            dev = self._find_device(self._pa)
        except Exception:
            self._pa.terminate()
            self._pa = None
            raise
        rate, channels = int(dev["defaultSampleRate"]), dev["maxInputChannels"]
        self.description = f"{dev['name'].removesuffix(' [Loopback]')} ({rate} Hz, {channels} ch)"
        resampler = soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="float32")
        on_audio = self._on_audio

        def callback(data, frame_count, time_info, status):
            try:
                x = np.frombuffer(data, dtype=np.float32)
                if channels > 1:
                    x = x.reshape(-1, channels).mean(axis=1)
                y = resampler.resample_chunk(x)
                if len(y):
                    on_audio(y)
            except Exception:  # an exception here would silently kill the stream
                log.exception("audio callback failed")
            return None, pyaudio.paContinue

        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=rate * self._block_ms // 1000,
            stream_callback=callback,
        )

    def _close(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def _follow_default(self) -> None:
        """Reopen the capture when Windows switches the default output device."""
        from tfv.capture.win_audio import com_thread, default_output_id

        with com_thread():
            current = default_output_id()
            while not self._stopped.wait(2.0):
                try:
                    latest = default_output_id()
                except OSError as exc:
                    log.debug("Could not query the default output: %s", exc)
                    continue
                if latest == current or latest is None:
                    continue
                current = latest
                self.reopen()

    def reopen(self) -> None:
        """Close and reopen the capture on the (possibly new) device."""
        with self._lock:
            if self._stopped.is_set():
                return
            self._close()
            try:
                self._open()
            except Exception as exc:
                log.warning("Could not capture the new output device: %s", exc)
                return
        log.info("Output device changed; now capturing %s", self.description)
        if self.on_device_change:
            self.on_device_change(self.description)
