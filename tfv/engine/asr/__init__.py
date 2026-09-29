"""Speech recognition backends."""

from __future__ import annotations

import logging

from tfv import languages
from tfv.engine.asr.base import ASRBackend, ASRResult

log = logging.getLogger(__name__)


def create_asr(choice, *, language: str, threads: int | None = None, api_key: str | None = None) -> ASRBackend:
    """Build and warm up the recognizer described by a resources.Choice."""
    if language not in languages.SUPPORTED:
        raise ValueError(f"unsupported language {language!r}; choose from {', '.join(languages.SUPPORTED)}")
    if choice.backend == "faster-whisper":
        from tfv.engine.asr.faster_whisper_backend import FasterWhisperBackend

        device = "cuda" if choice.device in ("cuda", "gpu") else "cpu"
        asr: ASRBackend = FasterWhisperBackend(language, choice.model, device=device, cpu_threads=threads or 0)
    elif choice.backend == "sensevoice":
        from tfv.engine.asr.sensevoice_backend import SenseVoiceBackend

        asr = SenseVoiceBackend(language, num_threads=threads)
    else:
        from tfv.engine.asr.cloud import create_cloud_asr

        asr = create_cloud_asr(choice, language=language, api_key=api_key)
    asr.warmup()
    return asr


__all__ = ["ASRBackend", "ASRResult", "create_asr"]
