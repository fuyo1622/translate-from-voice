"""Speech detection that holds up under background music and noise.

A small streaming denoiser (GTCRN, via sherpa-onnx) cleans a copy of the audio
for the VAD only; the recognizer still gets the original. On Japanese TV
speech with music as loud as the voice, this cut the lines the VAD missed
from 8 of 30 to 1 and the error rate from 49% to 21% (EVAL.md), and it did
not hurt clean speech. Denoising what the recognizer hears was worse.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tfv import SAMPLE_RATE


class VadDenoiser:
    """Turns each chunk into (original, denoised) of equal length, aligned sample for sample.

    The denoiser returns its output about 20 ms late, in whole frames, so the
    original audio is held back until its denoised counterpart is ready.
    """

    def __init__(self, model_path: str | Path, threads: int = 1):
        import sherpa_onnx

        config = sherpa_onnx.OnlineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=str(model_path)), num_threads=threads
            )
        )
        self._denoiser = sherpa_onnx.OnlineSpeechDenoiser(config)
        self._held = np.zeros(0, np.float32)

    def push(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._held = np.concatenate([self._held, audio.astype(np.float32, copy=False)])
        clean = np.asarray(self._denoiser(audio, SAMPLE_RATE).samples, np.float32)
        return self._pair(clean)

    def flush(self) -> tuple[np.ndarray, np.ndarray]:
        """The audio still held back; then start fresh (what follows a gap isn't continuous)."""
        pair = self._pair(np.asarray(self._denoiser.flush().samples, np.float32))
        self.reset()
        return pair

    def reset(self) -> None:
        self._denoiser.reset()
        self._held = np.zeros(0, np.float32)

    def _pair(self, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = min(len(clean), len(self._held))
        original, self._held = self._held[:n], self._held[n:]
        return original, clean[:n]
