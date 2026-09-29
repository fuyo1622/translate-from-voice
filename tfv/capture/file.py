from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from tfv import SAMPLE_RATE
from tfv.capture.base import AudioCallback, AudioSource


class FileSource(AudioSource):
    """Feeds an audio or video file through the pipeline, for testing without playback.

    With `realtime=True` the file is paced like live audio, so latency figures
    are meaningful; otherwise it is fed as fast as the pipeline accepts it.
    """

    def __init__(self, path: str | Path, realtime: bool = False, block_ms: int = 100):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.realtime = realtime
        self.description = f"file {self.path.name}"
        self._block = SAMPLE_RATE * block_ms // 1000
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, on_audio: AudioCallback, on_end: Callable[[], None]) -> None:
        from faster_whisper.audio import decode_audio  # PyAV-based decoder, any format ffmpeg reads

        audio = decode_audio(str(self.path), sampling_rate=SAMPLE_RATE)

        def run() -> None:
            t0 = time.monotonic()
            for i in range(0, len(audio), self._block):
                if self._stop.is_set():
                    break
                on_audio(audio[i : i + self._block])
                if self.realtime:
                    ahead = (i + self._block) / SAMPLE_RATE - (time.monotonic() - t0)
                    if ahead > 0:
                        self._stop.wait(ahead)
            on_end()

        self._thread = threading.Thread(target=run, name="tfv-file", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
