"""Build and run the whole engine from one config: shared by the CLI and the UI."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tfv import hardware, keys, resources
from tfv.engine.pipeline import Caption, Pipeline
from tfv.engine.segmenter import SegmenterConfig
from tfv.resources import Choice, Ladder

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
    profile: str = "auto"
    asr: str = "auto"
    """"auto" (the profile's ladder) or one choice, e.g. "whisper-small@cuda"."""
    mt: str = "auto"
    """"auto", "none", or one choice, e.g. "nllb-600m@cpu" or "deepl"."""
    mt_url: str | None = None
    """Base URL for mt="openai-compatible:<model>"."""
    threads: int | None = None
    adapt: bool = True
    """Step down to lighter models while running if live audio cannot be kept up with."""
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)


class Session:
    """Owns the source, models and pipeline for one run."""

    def __init__(self, config: SessionConfig, on_caption: Callable[[Caption], None], on_status: StatusCallback | None = None):
        self.config = config
        self._on_caption = on_caption
        self._status = on_status or (lambda msg: log.info("%s", msg))
        self.hardware: hardware.Hardware | None = None
        self.profile: resources.Profile | None = None
        self.pipeline: Pipeline | None = None
        self.asr_ladder: Ladder | None = None
        self.mt_ladder: Ladder | None = None
        self.asr = None
        self.translator = None
        self._monitor: resources.Monitor | None = None
        self._swap_lock = threading.Lock()

    def start(self) -> None:
        """Load everything (downloading models on first use) and start listening. Blocks while loading."""
        from tfv import models
        from tfv.engine.segmenter import Segmenter
        from tfv.engine.textproc import TextPostProcessor
        from tfv.engine.vad import SileroVAD

        cfg = self.config
        source = self._make_source()
        self.hardware = hw = hardware.probe()
        self.profile = resources.choose_profile(hw, cfg.profile)
        self._status(f"{hw.summary()} → profile {self.profile.name}")

        self.asr_ladder = self._asr_ladder(cfg.source_lang)
        self.asr = self.asr_ladder.load()
        self._status(f"Recognition: {self.asr.description}")
        if self._wants_translation(cfg.target_lang):
            # after recognition, so a local LLM sizes itself to the VRAM that is left
            self.mt_ladder = self._mt_ladder(cfg.source_lang, cfg.target_lang)
            self.translator = self.mt_ladder.load()
            self._status(f"Translation: {self.translator.description}")

        vad = SileroVAD(models.ensure("silero-vad").main_file)
        self.pipeline = Pipeline(
            source, Segmenter(vad, cfg.segmenter), self.asr, self._on_caption,
            source_lang=cfg.source_lang, target_lang=cfg.target_lang, translator=self.translator,
            postprocess=TextPostProcessor(),
        )  # fmt: skip
        self.pipeline.start()
        if source.realtime and cfg.adapt:
            self._monitor = resources.Monitor(self.pipeline, self._step_down)
            self._monitor.start()
        self._status(f"Listening to {source.description}")

    def wait(self, timeout: float | None = None) -> bool:
        return self.pipeline.wait(timeout) if self.pipeline else True

    def stop(self) -> None:
        if self._monitor:
            self._monitor.stop()
        if self.pipeline:
            self.pipeline.stop()
        for backend in (self.translator, self.asr):
            if backend is not None and hasattr(backend, "close"):
                backend.close()

    def set_languages(self, source_lang: str, target_lang: str | None) -> None:
        """Switch languages while running (reloads the recognizer if the spoken language changed)."""
        with self._swap_lock:
            cfg, p = self.config, self.pipeline
            if source_lang != cfg.source_lang:
                self._status(f"Loading recognition for {source_lang}…")
                self.asr_ladder = self._asr_ladder(source_lang)
                new_asr = self.asr_ladder.load()
                _close(p.replace_asr(new_asr, source_lang))
                self.asr, cfg.source_lang = new_asr, source_lang
            if not self._wants_translation(target_lang):
                _close(p.replace_translator(None))
                p.set_target(target_lang)
                self.translator, self.mt_ladder = None, None
            elif self.translator is None:
                self._status("Loading translation…")
                self.mt_ladder = self._mt_ladder(source_lang, target_lang)
                self.translator = self.mt_ladder.load()
                p.replace_translator(self.translator, target_lang)
            else:
                p.set_target(target_lang)
            cfg.target_lang = target_lang
            self._status("Ready")

    # -- internals ---------------------------------------------------------------

    def _wants_translation(self, target_lang: str | None) -> bool:
        return bool(target_lang) and target_lang != self.config.source_lang and self.config.mt != "none"

    def _asr_ladder(self, language: str) -> Ladder:
        from tfv.engine.asr import create_asr

        cfg, hw = self.config, self.hardware
        rungs = self.profile.asr if cfg.asr == "auto" else [Choice.parse(cfg.asr)]
        return Ladder(
            "recognition", rungs,
            build=lambda c: create_asr(c, language=language, threads=cfg.threads, api_key=keys.get(c.backend)),
            unfit=lambda c: resources.unfit_reason(c, hw, keys.has),
            bench=lambda asr: resources.bench_asr(asr, language), limit=resources.ASR_MAX_RTF, status=self._status,
        )  # fmt: skip

    def _mt_ladder(self, source: str, target: str) -> Ladder:
        from tfv.engine.mt import create_translator

        cfg, hw = self.config, self.hardware
        rungs = self.profile.mt if cfg.mt == "auto" else [Choice.parse(cfg.mt)]
        return Ladder(
            "translation", rungs,
            build=lambda c: create_translator(
                c, source=source, target=target, url=cfg.mt_url, api_key=keys.get(c.backend), threads=cfg.threads
            ),
            unfit=lambda c: resources.unfit_reason(c, hw, keys.has),
            bench=lambda t: resources.bench_mt(t, source, target), limit=resources.MT_MAX_S, status=self._status,
        )  # fmt: skip

    def _step_down(self, stage: str, reason: str) -> bool:
        """Called by the monitor when a stage cannot keep up with live audio."""
        with self._swap_lock:
            ladder = self.asr_ladder if stage == "asr" else self.mt_ladder
            if ladder is None or not ladder.can_step_down:
                return False
            self._status(f"{ladder.kind.capitalize()} is falling behind ({reason}); switching to a lighter option…")
            try:
                new = ladder.step_down()
            except RuntimeError as exc:
                log.warning("%s", exc)
                return False
            if stage == "asr":
                _close(self.pipeline.replace_asr(new))
                self.asr = new
            else:
                _close(self.pipeline.replace_translator(new))
                self.translator = new
            self._status(f"{ladder.kind.capitalize()}: {new.description}")
            return True

    def _make_source(self):
        cfg = self.config
        if cfg.file:
            from tfv.capture.file import FileSource

            return FileSource(cfg.file, realtime=cfg.realtime)
        from tfv.capture.loopback import LoopbackSource

        source = LoopbackSource(cfg.audio_device)
        source.on_device_change = lambda name: self._status(f"Output changed; now listening to {name}")
        return source


def _close(backend) -> None:
    if backend is not None and hasattr(backend, "close"):
        backend.close()
