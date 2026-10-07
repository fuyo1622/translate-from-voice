from __future__ import annotations

import logging
import os
import time

import numpy as np

from tfv import SAMPLE_RATE, models
from tfv.engine.asr.base import ASRBackend, ASRResult

log = logging.getLogger(__name__)

_PAD = np.zeros(int(0.9 * SAMPLE_RATE), np.float32)
"""Silence around each utterance, as the ReazonSpeech library adds, so the last syllable isn't lost."""


class ReazonSpeechBackend(ASRBackend):
    """ReazonSpeech k2 v2 (a Zipformer transducer) via sherpa-onnx: Japanese only, trained on
    35,000 hours of Japanese TV, fast on the CPU."""

    name = "reazonspeech"

    def __init__(self, language: str, num_threads: int | None = None):
        if language != "ja":
            raise ValueError("ReazonSpeech recognizes Japanese only")
        import sherpa_onnx

        path = models.ensure("reazonspeech-k2-v2").path
        self._threads = num_threads or max(1, min(4, (os.cpu_count() or 2) // 2))
        t0 = time.perf_counter()
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(path / "encoder-epoch-99-avg-1.int8.onnx"),
            decoder=str(path / "decoder-epoch-99-avg-1.onnx"),
            joiner=str(path / "joiner-epoch-99-avg-1.int8.onnx"),
            tokens=str(path / "tokens.txt"),
            num_threads=self._threads,
            decoding_method="greedy_search",
        )
        log.info("Loaded ReazonSpeech k2 on CPU (%d threads) in %.1fs", self._threads, time.perf_counter() - t0)
        self.description = f"ReazonSpeech k2 on cpu ({self._threads} threads)"

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        t0 = time.perf_counter()
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.concatenate([_PAD, audio.astype(np.float32, copy=False), _PAD]))
        self._recognizer.decode_stream(stream)
        return ASRResult(text=stream.result.text.strip(), language="ja", compute_s=time.perf_counter() - t0)

    def set_language(self, language: str) -> None:
        if language != "ja":
            raise NotImplementedError  # the session loads another recognizer
