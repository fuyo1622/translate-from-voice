"""Paid speech recognition: OpenAI transcription and Deepgram.

Each utterance (already cut by our VAD) is sent as one short WAV request.
"""

from __future__ import annotations

import io
import time
import wave

import numpy as np

from tfv import SAMPLE_RATE, usage
from tfv.engine.asr.base import ASRBackend, ASRResult

OPENAI_DEFAULT_MODEL = "gpt-4o-mini-transcribe"
DEEPGRAM_DEFAULT_MODEL = "nova-3"
_DEEPGRAM_LANGUAGES = {"zh": "zh-TW", "en": "en", "ja": "ja"}


def to_wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


class OpenAITranscriber(ASRBackend):
    name = "openai"

    def __init__(self, api_key: str, language: str, model: str | None = None, timeout: float = 20.0, transport=None):
        import httpx

        self._model = model or OPENAI_DEFAULT_MODEL
        self._language = language
        self._http = httpx.Client(
            base_url="https://api.openai.com", timeout=timeout, headers={"Authorization": f"Bearer {api_key}"}, transport=transport
        )
        self.description = f"OpenAI {self._model}"

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        t0 = time.perf_counter()
        r = self._http.post(
            "/v1/audio/transcriptions",
            data={"model": self._model, "language": self._language, "response_format": "json"},
            files={"file": ("utterance.wav", to_wav(audio), "audio/wav")},
        )
        r.raise_for_status()
        usage.record("openai", self._model, audio_s=len(audio) / SAMPLE_RATE)
        return ASRResult(text=r.json().get("text", "").strip(), language=self._language, compute_s=time.perf_counter() - t0)

    def close(self) -> None:
        self._http.close()


class DeepgramTranscriber(ASRBackend):
    name = "deepgram"

    def __init__(self, api_key: str, language: str, model: str | None = None, timeout: float = 20.0, transport=None):
        import httpx

        self._model = model or DEEPGRAM_DEFAULT_MODEL
        self._language = language
        self._http = httpx.Client(
            base_url="https://api.deepgram.com", timeout=timeout, headers={"Authorization": f"Token {api_key}"}, transport=transport
        )
        self.description = f"Deepgram {self._model}"

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        t0 = time.perf_counter()
        r = self._http.post(
            "/v1/listen",
            params={"model": self._model, "language": _DEEPGRAM_LANGUAGES[self._language], "punctuate": "true", "smart_format": "true"},
            content=to_wav(audio),
            headers={"Content-Type": "audio/wav"},
        )
        r.raise_for_status()
        usage.record("deepgram", self._model, audio_s=len(audio) / SAMPLE_RATE)
        channels = r.json()["results"]["channels"]
        text = channels[0]["alternatives"][0]["transcript"] if channels and channels[0]["alternatives"] else ""
        return ASRResult(text=text.strip(), language=self._language, compute_s=time.perf_counter() - t0)

    def close(self) -> None:
        self._http.close()


def create_cloud_asr(choice, language: str, api_key: str | None) -> ASRBackend:
    if not api_key:
        raise RuntimeError(f"no API key for {choice.backend}; add one with `python -m tfv keys set {choice.backend}`")
    if choice.backend == "openai":
        return OpenAITranscriber(api_key, language, choice.model)
    if choice.backend == "deepgram":
        return DeepgramTranscriber(api_key, language, choice.model)
    raise ValueError(f"unknown speech recognition service {choice.backend!r}")
