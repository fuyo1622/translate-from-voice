from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np

from tfv import SAMPLE_RATE, models
from tfv.engine.asr.base import ASRBackend, ASRResult

log = logging.getLogger(__name__)


class SenseVoiceBackend(ASRBackend):
    """SenseVoice-Small via sherpa-onnx: fast on CPU, strong on Chinese, good on ja/en."""

    name = "sensevoice"

    def __init__(self, language: str, num_threads: int | None = None, model_dir: str | None = None):
        self._path = Path(model_dir) if model_dir else models.ensure("sensevoice-small").path
        # physical cores, roughly; more threads than that only adds contention
        self._threads = num_threads or max(1, min(4, (os.cpu_count() or 2) // 2))
        t0 = time.perf_counter()
        self._recognizer = self._load(language)
        self._language = language
        log.info("Loaded SenseVoice-Small on CPU (%d threads) in %.1fs", self._threads, time.perf_counter() - t0)
        self.description = f"SenseVoice-Small on cpu ({self._threads} threads)"

    def _load(self, language: str):
        import sherpa_onnx

        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(self._path / "model.int8.onnx"),
            tokens=str(self._path / "tokens.txt"),
            num_threads=self._threads,
            language=language,
            use_itn=True,  # punctuation and written-form numbers
        )

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        t0 = time.perf_counter()
        recognizer = self._recognizer
        stream = recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, audio)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        return ASRResult(text=text, language=self._language, compute_s=time.perf_counter() - t0)

    def set_language(self, language: str) -> None:
        # the language is fixed per recognizer; a new one loads in about two seconds
        self._recognizer = self._load(language)
        self._language = language

    def close(self) -> None:
        self._recognizer = None
