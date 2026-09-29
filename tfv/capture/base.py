from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

AudioCallback = Callable[[np.ndarray], None]


class AudioSource(ABC):
    """Produces 16 kHz mono float32 audio.

    `start` must return quickly; audio is delivered from a background thread.
    """

    description = ""
    realtime = True
    """False when the source can produce audio faster than real time (files)."""

    @abstractmethod
    def start(self, on_audio: AudioCallback, on_end: Callable[[], None]) -> None:
        """Begin delivering chunks to `on_audio`; call `on_end` if the audio runs out."""

    @abstractmethod
    def stop(self) -> None:
        """Stop delivering audio and release the device."""
