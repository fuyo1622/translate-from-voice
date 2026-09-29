"""Build and run the whole engine from one config: shared by the CLI and the UI."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tfv.engine.pipeline import Caption, Pipeline
from tfv.engine.segmenter import SegmenterConfig

log = logging.getLogger(__name__)

StatusCallback = Callable[[str], None]


@dataclass
class SessionConfig:
    source_lang: str
    target_lang: str | None = None
    """None: transcribe only."""
    audio_device: str | None = None
    file: Path | None = None
    realtime: bool = False
    asr: str = "auto"
    asr_model: str | None = None
    asr_compute: str = "auto"
    mt: str = "auto"
    """"auto", "none", or a translation backend name."""
    mt_model: str | None = None
    mt_compute: str = "auto"
    mt_url: str | None = None
    mt_api_key: str | None = None
    threads: int | None = None
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)


class Session:
    """Owns the source, models and pipeline for one run."""

    def __init__(self, config: SessionConfig, on_caption: Callable[[Caption], None], on_status: StatusCallback | None = None):
        self.config = config
        self._on_caption = on_caption
        self._status = on_status or (lambda msg: log.info("%s", msg))
        self.pipeline: Pipeline | None = None
        self.asr = None
        self.translator = None

    def start(self) -> None:
        """Load everything (downloading models on first use) and start listening. Blocks while loading."""
        from tfv import models
        from tfv.engine.asr import create_asr
        from tfv.engine.segmenter import Segmenter
        from tfv.engine.textproc import TextPostProcessor
        from tfv.engine.vad import SileroVAD

        cfg = self.config
        source = self._make_source()
        self._status("Loading speech recognition…")
        self.asr = create_asr(cfg.asr, language=cfg.source_lang, compute=cfg.asr_compute, model=cfg.asr_model, threads=cfg.threads)
        self._status(f"Recognizer: {self.asr.description}")
        if cfg.target_lang and cfg.target_lang != cfg.source_lang and cfg.mt != "none":
            # after ASR, so a local LLM sizes itself to the VRAM that is left
            self._status("Loading translation…")
            self.translator = self._make_translator(cfg.target_lang)
            self._status(f"Translator: {self.translator.description}")
        vad = SileroVAD(models.ensure("silero-vad").main_file)
        self.pipeline = Pipeline(
            source, Segmenter(vad, cfg.segmenter), self.asr, self._on_caption,
            source_lang=cfg.source_lang, target_lang=cfg.target_lang, translator=self.translator,
            postprocess=TextPostProcessor(),
        )  # fmt: skip
        self.pipeline.start()
        self._status(f"Listening to {source.description}")

    def wait(self, timeout: float | None = None) -> bool:
        return self.pipeline.wait(timeout) if self.pipeline else True

    def stop(self) -> None:
        if self.pipeline:
            self.pipeline.stop()
        if self.translator:
            self.translator.close()

    def _make_source(self):
        cfg = self.config
        if cfg.file:
            from tfv.capture.file import FileSource

            return FileSource(cfg.file, realtime=cfg.realtime)
        from tfv.capture.loopback import LoopbackSource

        return LoopbackSource(cfg.audio_device)

    def _make_translator(self, target: str):
        from tfv.engine.mt import create_translator

        cfg = self.config
        backend = cfg.mt
        if backend == "auto":
            backend = _default_mt_backend()
        return create_translator(
            backend, source=cfg.source_lang, target=target, model=cfg.mt_model, compute=cfg.mt_compute,
            url=cfg.mt_url, api_key=cfg.mt_api_key, threads=cfg.threads,
        )  # fmt: skip


def _default_mt_backend() -> str:
    """Local LLM when there is a discrete GPU for it, NLLB on the CPU otherwise."""
    from tfv.engine.mt.llama_server import best_device, ensure_runtime, list_devices

    try:
        return "llama" if best_device(list_devices(ensure_runtime())) else "nllb"
    except Exception as exc:
        log.info("No GPU for a local LLM (%s); using NLLB", exc)
        return "nllb"
