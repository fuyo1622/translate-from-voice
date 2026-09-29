from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from tfv import SAMPLE_RATE


@dataclass
class ASRResult:
    text: str
    language: str
    compute_s: float = 0.0


class ASRBackend(ABC):
    """Speech recognizer for one utterance of 16 kHz mono float32 audio in a known language."""

    name = "?"
    description = ""

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> ASRResult: ...

    def warmup(self) -> None:
        """Run once so lazy library loading and kernel setup happen before real audio."""
        rng = np.random.default_rng(0)
        self.transcribe((rng.standard_normal(SAMPLE_RATE) * 0.01).astype(np.float32))

    def close(self) -> None:
        """Release the model (and its GPU memory)."""
