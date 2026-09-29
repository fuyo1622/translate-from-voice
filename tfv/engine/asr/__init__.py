"""Speech recognition backends and the logic that picks one for this machine."""

from __future__ import annotations

import logging

from tfv import gpu, languages
from tfv.engine.asr.base import ASRBackend, ASRResult

log = logging.getLogger(__name__)

BACKENDS = ("auto", "faster-whisper", "sensevoice")


def create_asr(
    backend: str = "auto",
    *,
    language: str,
    compute: str = "auto",
    model: str | None = None,
    threads: int | None = None,
) -> ASRBackend:
    """Build and warm up a recognizer for speech in `language`.

    "auto" uses Whisper large-v3-turbo on an NVIDIA GPU when CUDA works, and
    SenseVoice-Small on the CPU otherwise. Phase 3 replaces this rule with
    hardware profiles.
    """
    if language not in languages.SUPPORTED:
        raise ValueError(f"unsupported language {language!r}; choose from {', '.join(languages.SUPPORTED)}")

    if backend == "auto":
        cuda_ok, reason = gpu.cuda_status()
        use_whisper = cuda_ok and compute != "cpu"
        backend = "faster-whisper" if use_whisper else "sensevoice"
        if not use_whisper:
            log.info("Using SenseVoice on CPU (%s)", "--compute cpu" if cuda_ok else f"CUDA unavailable: {reason}")

    if backend == "sensevoice":
        if compute == "cuda":
            raise ValueError("the sensevoice backend runs on CPU only")
        asr: ASRBackend = _sensevoice(language=language, num_threads=threads, model_dir=model)
        asr.warmup()
        return asr

    if backend == "faster-whisper":
        return _whisper(language, compute, model, threads)

    raise ValueError(f"unknown ASR backend {backend!r}; choose from {', '.join(BACKENDS)}")


def _whisper(language, compute, model, threads) -> ASRBackend:
    from tfv.engine.asr.faster_whisper_backend import FasterWhisperBackend

    device = compute
    if compute in ("auto", "cuda"):
        cuda_ok, reason = gpu.cuda_status()
        if not cuda_ok:
            if compute == "cuda":
                raise RuntimeError(f"CUDA unavailable: {reason}")
            log.info("CUDA unavailable (%s); running Whisper on CPU", reason)
        device = "cuda" if cuda_ok else "cpu"

    def build(dev: str) -> ASRBackend:
        # turbo is too slow on most CPUs for live captions; small keeps up
        size = model or ("large-v3-turbo" if dev == "cuda" else "small")
        asr = FasterWhisperBackend(language, size, device=dev, cpu_threads=threads or 0)
        asr.warmup()
        return asr

    try:
        return build(device)
    except Exception as exc:
        if device != "cuda" or compute == "cuda":
            raise
        log.warning("Whisper failed on the GPU (%s); falling back to CPU", exc)
        return build("cpu")


def _sensevoice(**kwargs) -> ASRBackend:
    from tfv.engine.asr.sensevoice_backend import SenseVoiceBackend

    return SenseVoiceBackend(**kwargs)


__all__ = ["ASRBackend", "ASRResult", "BACKENDS", "create_asr"]
