"""Build and run the whole engine from one config: shared by the CLI and the UI."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tfv import hardware, keys, models, resources
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
    capture: str = "output"
    """"output": what the computer plays (Windows loopback); "input": a microphone or virtual cable."""
    audio_device: str | None = None
    """Device name/index for `capture`; None = the system default."""
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

    def __init__(
        self,
        config: SessionConfig,
        on_caption: Callable[[Caption], None],
        on_status: StatusCallback | None = None,
        source=None,
    ):
        self.config = config
        self._on_caption = on_caption
        self._status = on_status or (lambda msg: log.info("%s", msg))
        self._source = source  # an AudioSource to use instead of building one from the config
        self.hardware: hardware.Hardware | None = None
        self.profile: resources.Profile | None = None
        self.pipeline: Pipeline | None = None
        self.asr_ladder: Ladder | None = None
        self.mt_ladder: Ladder | None = None
        self.asr = None
        self.translator = None
        self._monitor: resources.Monitor | None = None
        self._swap_lock = threading.Lock()
        self._last_progress = 0.0

    def start(self) -> None:
        """Load everything (downloading models on first use) and start listening. Blocks while loading."""
        from tfv.engine.segmenter import Segmenter
        from tfv.engine.textproc import TextPostProcessor
        from tfv.engine.vad import SileroVAD

        cfg = self.config
        source = self._source or self._make_source()
        models.set_progress_listener(self._download_progress)
        self.hardware = hw = hardware.probe()
        self.profile = resources.choose_profile(hw, cfg.profile)
        self._status(f"{hw.summary()} → profile {self.profile.name}")

        self.asr_ladder = self._asr_ladder()
        self.asr = self.asr_ladder.load()
        self._status(f"Recognition: {self.asr.description}")
        if self._wants_translation(cfg.target_lang):
            # after recognition, so a local LLM sizes itself to the VRAM that is left
            self.mt_ladder = self._mt_ladder()
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
        models.set_progress_listener(None)
        if self._monitor:
            self._monitor.stop()
        if self.pipeline:
            self.pipeline.stop()
        for backend in (self.translator, self.asr):
            if backend is not None and hasattr(backend, "close"):
                backend.close()

    def pause(self) -> None:
        if self.pipeline:
            self.pipeline.pause()

    def resume(self) -> None:
        if self.pipeline:
            self.pipeline.resume()

    def describe(self) -> list[str]:
        """What is running, for status displays."""
        lines = [f"Recognition: {self.asr.description}" if self.asr else "Recognition: not loaded"]
        lines.append(f"Translation: {self.translator.description}" if self.translator else "Translation: off")
        return lines

    def stats(self) -> dict | None:
        return self.pipeline.stats() if self.pipeline else None

    def set_languages(self, source_lang: str, target_lang: str | None) -> None:
        """Switch languages while running (reloads the recognizer if the spoken language changed)."""
        with self._swap_lock:
            cfg, p, profile = self.config, self.pipeline, self.profile
            old_source, old_target = cfg.source_lang, cfg.target_lang
            cfg.source_lang, cfg.target_lang = source_lang, target_lang  # ladders read these when loading
            new_asr_ladder = cfg.asr == "auto" and profile.asr_for(source_lang) != profile.asr_for(old_source)
            new_mt_ladder = cfg.mt == "auto" and profile.mt_for(target_lang) != profile.mt_for(old_target)
            if new_asr_ladder:
                # Another recognizer suits this language (e.g. SenseVoice on the CPU so the GPU can
                # hold a larger translator): free the translator first so both are sized together.
                self._status(f"Loading recognition for {source_lang}…")
                _close(p.replace_translator(None))
                self.translator, self.mt_ladder = None, None
                self.asr_ladder = self._asr_ladder()
                new_asr = self.asr_ladder.load()
                _close(p.replace_asr(new_asr, source_lang))
                self.asr = new_asr
            elif source_lang != old_source:
                self._status(f"Switching recognition to {source_lang}…")
                try:
                    self.asr.set_language(source_lang)  # same model, no reload
                    p.replace_asr(self.asr, source_lang)  # same object: only updates the pipeline's language
                except NotImplementedError:
                    new_asr = self.asr_ladder.load(max(self.asr_ladder.index, 0))
                    _close(p.replace_asr(new_asr, source_lang))
                    self.asr = new_asr
            if not self._wants_translation(target_lang):
                _close(p.replace_translator(None))
                p.set_target(target_lang)
                self.translator, self.mt_ladder = None, None
            elif self.translator is None or new_mt_ladder:
                self._status("Loading translation…")
                _close(p.replace_translator(None))  # free its memory before loading the next one
                self.translator = None
                self.mt_ladder = self._mt_ladder()
                self.translator = self.mt_ladder.load()
                p.replace_translator(self.translator, target_lang)
            else:
                p.set_target(target_lang)
            self._status("Ready")

    # -- internals ---------------------------------------------------------------

    def _download_progress(self, label: str, done: int, total: int) -> None:
        now = time.monotonic()
        if now - self._last_progress >= 1.0 or done == total:
            self._last_progress = now
            pct = f"{100 * done // total}% of {total / 1e6:,.0f} MB" if total else f"{done / 1e6:,.0f} MB"
            self._status(f"Downloading {label}: {pct} (first use only)")

    def _wants_translation(self, target_lang: str | None) -> bool:
        return bool(target_lang) and target_lang != self.config.source_lang and self.config.mt != "none"

    def _asr_ladder(self) -> Ladder:
        """Ladder for the profile's recognizers; loads for the spoken language current at load time."""
        from tfv.engine.asr import create_asr

        cfg, hw = self.config, self.hardware
        rungs = self.profile.asr_for(cfg.source_lang) if cfg.asr == "auto" else [Choice.parse(cfg.asr)]
        return Ladder(
            "recognition", rungs,
            build=lambda c: create_asr(c, language=cfg.source_lang, threads=cfg.threads, api_key=keys.get(c.backend)),
            unfit=lambda c: resources.unfit_reason(c, hw, keys.has),
            bench=lambda asr: resources.bench_asr(asr, cfg.source_lang), limit=resources.ASR_MAX_RTF, status=self._status,
        )  # fmt: skip

    def _mt_ladder(self) -> Ladder:
        from tfv.engine.mt import create_translator

        cfg, hw = self.config, self.hardware
        rungs = self.profile.mt_for(cfg.target_lang) if cfg.mt == "auto" else [Choice.parse(cfg.mt)]
        return Ladder(
            "translation", rungs,
            build=lambda c: create_translator(
                c, source=cfg.source_lang, target=cfg.target_lang, url=cfg.mt_url, api_key=keys.get(c.backend), threads=cfg.threads
            ),
            unfit=lambda c: resources.unfit_reason(c, hw, keys.has),
            bench=lambda t: resources.bench_mt(t, cfg.source_lang, cfg.target_lang), limit=resources.MT_MAX_S, status=self._status,
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
        return make_source(self.config, self._status)


def make_source(cfg: SessionConfig, on_status: StatusCallback | None = None):
    """The audio source a config asks for: a file, an input device, or Windows output loopback."""
    import sys

    if cfg.file:
        from tfv.capture.file import FileSource

        return FileSource(cfg.file, realtime=cfg.realtime)
    if cfg.capture == "input" or sys.platform != "win32":
        # outside Windows, system audio arrives through a virtual input (BlackHole, a Pulse monitor)
        from tfv.capture.input_device import InputDeviceSource

        return InputDeviceSource(cfg.audio_device)
    from tfv.capture.loopback import LoopbackSource

    source = LoopbackSource(cfg.audio_device)
    if on_status:
        source.on_device_change = lambda name: on_status(f"Output changed; now listening to {name}")
    return source


def _close(backend) -> None:
    if backend is not None and hasattr(backend, "close"):
        backend.close()
