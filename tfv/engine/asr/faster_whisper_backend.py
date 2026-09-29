from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from tfv import models
from tfv.engine.asr.base import ASRBackend, ASRResult

log = logging.getLogger(__name__)


class FasterWhisperBackend(ASRBackend):
    """Whisper via CTranslate2. Best quality on NVIDIA GPUs; usable on CPU with small models.

    `model` is a size name ("large-v3-turbo", "small", ...) or a model directory.
    """

    name = "faster-whisper"

    def __init__(
        self,
        language: str,
        model: str = "large-v3-turbo",
        device: str = "cuda",
        compute_type: str = "auto",
        beam_size: int = 5,
        cpu_threads: int = 0,
    ):
        from faster_whisper import WhisperModel

        if compute_type == "auto":
            compute_type = "int8_float16" if device == "cuda" else "int8"
        path = Path(model) if Path(model).is_dir() else models.faster_whisper(model)
        t0 = time.perf_counter()
        self._model = WhisperModel(str(path), device=device, compute_type=compute_type, cpu_threads=cpu_threads)
        log.info("Loaded Whisper %s on %s (%s) in %.1fs", path.name, device, compute_type, time.perf_counter() - t0)
        self.description = f"faster-whisper {path.name.removeprefix('faster-whisper-')} on {device} ({compute_type})"
        self._language = language
        self._options = dict(
            language=language,
            beam_size=beam_size,
            vad_filter=False,  # segments are already cut by our VAD
            condition_on_previous_text=False,  # stops hallucinations from snowballing
            without_timestamps=True,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        t0 = time.perf_counter()
        segments, _ = self._model.transcribe(audio, **self._options)
        text = "".join(s.text for s in segments).strip()
        return ASRResult(text=text, language=self._language, compute_s=time.perf_counter() - t0)
