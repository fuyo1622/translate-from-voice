"""Build and run the whole engine from one config: shared by the CLI and the UI."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from tfv import hardware, keys, languages, models, resources
from tfv.engine.pipeline import SONG_MAX_RUN_S, Caption, Pipeline
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
    partials: bool = False
    """Live text: also emit "partial" captions while someone is still talking."""
    vad_denoise: bool = True
    """Let the VAD hear a denoised copy, so speech under music or noise isn't missed."""
    separate_vocals: bool = True
    """On lines with music, take the music out before recognition (songs, talk over loud music),
    if a graphics chip can do it fast enough (tfv.engine.separate)."""
    song_mode: bool = False
    """While music plays, separate continuously so the VAD hears the voices too (finds more singing
    under loud music; captions about a second later). Needs a fast graphics chip."""
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
        self.separator = None
        self.separator_note: str | None = None
        """Why there is no music filter, when it was wanted."""
        self._monitor: resources.Monitor | None = None
        self._swap_lock = threading.Lock()
        self._last_progress = 0.0

    def start(self) -> None:
        """Load everything (downloading models on first use) and start listening. Blocks while loading."""
        from tfv.engine.furigana import Furigana
        from tfv.engine.segmenter import Segmenter
        from tfv.engine.textproc import TextPostProcessor
        from tfv.engine.vad import SileroVAD

        cfg = self.config
        source = self._source or self._make_source()
        models.set_progress_listener(self._download_progress)
        furigana = Furigana()
        if cfg.source_lang == "ja":  # load the dictionary while the models load
            threading.Thread(target=_quietly, args=(furigana.load,), name="tfv-furigana", daemon=True).start()
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
        if cfg.separate_vocals:  # last, so it sees what graphics memory the models left
            self.separator = self._load_separator()

        vad = SileroVAD(models.ensure("silero-vad").main_file)
        denoiser = None
        if cfg.vad_denoise:
            from tfv.engine.denoise import VadDenoiser

            try:
                denoiser = VadDenoiser(models.ensure("gtcrn").main_file)
            except Exception as exc:  # e.g. offline on the first start
                log.warning("Background filter unavailable (%s); speech under music may be missed", exc)
        self.pipeline = Pipeline(
            source, Segmenter(vad, cfg.segmenter), self.asr, self._on_caption,
            source_lang=cfg.source_lang, target_lang=cfg.target_lang, translator=self.translator,
            postprocess=TextPostProcessor(), readings=furigana.segments, vad_denoiser=denoiser, separator=self.separator,
            partials=cfg.partials,
        )  # fmt: skip
        self.pipeline.song_mode_allowed = cfg.song_mode
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
        for backend in (self.translator, self.asr, self.separator):
            if backend is not None and hasattr(backend, "close"):
                backend.close()

    def pause(self) -> None:
        if self.pipeline:
            self.pipeline.pause()

    def resume(self) -> None:
        if self.pipeline:
            self.pipeline.resume()

    def set_partials(self, on: bool) -> None:
        """Turn live text on or off while running."""
        self.config.partials = on
        if self.pipeline:
            self.pipeline.partials = on

    def set_separate_vocals(self, on: bool) -> None:
        """Turn the music filter on or off while running (loading it takes a few seconds;
        turning it off frees its graphics memory)."""
        self.config.separate_vocals = on
        with self._swap_lock:
            if on and self.separator is None:
                self.separator = self._load_separator()
                if self.pipeline:
                    self.pipeline.separator, self.pipeline.separator_error = self.separator, None
            elif not on and self.separator is not None:
                if self.pipeline:
                    self.pipeline.separator = None
                _close(self.separator)  # waits for a line it is separating
                self.separator = None

    def set_song_mode(self, on: bool) -> None:
        """Song mode on or off while running (it starts with the next music)."""
        self.config.song_mode = on
        if self.pipeline:
            self.pipeline.song_mode_allowed = on

    def _filter_line(self) -> str:
        line = f"Music filter: {self.separator.description}"
        if getattr(self.separator, "run_s", 1.0) > SONG_MAX_RUN_S:
            return line + ("; song mode needs a faster graphics chip" if self.config.song_mode else "")
        return line + ("; song mode on" if self.config.song_mode else "; song mode off")

    def _load_separator(self):
        """The music filter on a suitable graphics chip, or None (with the reason in separator_note)."""
        from tfv.engine import separate

        if (reason := separate.unavailable_reason()) is not None:
            self.separator_note = f"not available: {reason}"
            return None
        self._status("Loading the music filter (for songs and loud music)…")
        try:
            separator = separate.load()
        except Exception as exc:  # e.g. offline on first use, or a graphics driver problem
            log.exception("The music filter could not be loaded")
            self.separator_note = f"could not be loaded ({exc})"
            return None
        if separator is None:
            self.separator_note = "not available: no graphics chip with room for it that runs it fast enough"
            return None
        self.separator_note = None
        self.separator = separator
        self._status(self._filter_line())
        return separator

    def describe(self) -> list[str]:
        """What is running, for status displays."""
        lines = [f"Recognition: {self.asr.description}" if self.asr else "Recognition: not loaded"]
        translating = self.translator is not None and self._wants_translation(self.config.target_lang)
        lines.append(f"Translation: {self.translator.description}" if translating else "Translation: off")
        if self.pipeline and self.pipeline.separator_error:
            lines.append(f"Music filter: stopped after an error ({self.pipeline.separator_error})")
        elif self.separator is not None:
            lines.append(self._filter_line())
        elif self.config.separate_vocals and self.separator_note:
            lines.append(f"Music filter: {self.separator_note}")
        else:
            lines.append("Music filter: off")
        return lines

    def summary(self) -> str:
        """The languages now in effect, e.g. after a switch."""
        cfg = self.config
        spoken = languages.LABELS[cfg.source_lang]
        if not self._wants_translation(cfg.target_lang):
            return f"Now: {spoken}, captions only"
        if self.translator is None:
            return f"Now: {spoken}; translation could not be loaded (see the log)"
        return f"Now: {spoken} → {languages.LABELS[cfg.target_lang]}"

    def stats(self) -> dict | None:
        return self.pipeline.stats() if self.pipeline else None

    def save_recent(self) -> Path:
        """Save the last minute of audio and the captions shown for it in the feedback folder,
        so a bad stretch can be examined later. Returns the WAV file."""
        import wave

        audio, start, captions = self.pipeline.recent() if self.pipeline else (np.zeros(0), 0.0, [])
        if not len(audio):
            raise RuntimeError("nothing heard yet")
        folder = models.feedback_dir()
        folder.mkdir(parents=True, exist_ok=True)
        stem = folder / time.strftime("%Y-%m-%d_%H-%M-%S")
        if stem.with_suffix(".wav").exists():  # saved twice in a second
            stem = stem.with_name(stem.name + f"_{time.monotonic_ns() % 1000:03d}")
        with wave.open(str(stem.with_suffix(".wav")), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        latest: dict[int, object] = {}
        for c in captions:  # a caption's last state: translated (or not) after its text
            latest[c.id] = c
        lines = [self.summary(), *self.describe(), "", "seconds in the WAV | recognized text | translation"]
        for c in sorted(latest.values(), key=lambda c: c.start):
            lines.append(f"{max(0.0, c.start - start):6.1f}-{c.end - start:6.1f} | {c.text} | {c.translation or ''}")
        stem.with_suffix(".txt").write_text("\n".join(lines) + "\n", "utf-8")
        return stem.with_suffix(".wav")

    def set_languages(self, source_lang: str, target_lang: str | None) -> None:
        """Switch languages while running, reloading as little as possible.

        The loaded translator handles every language pair, so a new target only retargets it
        (the model best for that language is picked at the next start), and it stays loaded
        while translation is off, so switching back is instant. A new spoken language loads
        the recognizer its profile prefers; the translator is freed first only if that
        recognizer wants the GPU, so the two are sized together.
        """
        with self._swap_lock:
            cfg, p, profile = self.config, self.pipeline, self.profile
            old_source = cfg.source_lang
            cfg.source_lang, cfg.target_lang = source_lang, target_lang  # ladders read these when loading
            spoken = languages.LABELS[source_lang]
            if cfg.asr == "auto" and profile.asr_for(source_lang) != profile.asr_for(old_source):
                ladder = self._asr_ladder()
                if ladder.rungs[0].device in ("cuda", "gpu") and self.translator is not None:
                    _close(p.replace_translator(None))  # e.g. Whisper for English: give it the GPU first
                    self.translator, self.mt_ladder = None, None
                self._status(f"Loading recognition for {spoken}…")
                new_asr = ladder.load()
                self.asr_ladder = ladder
                _close(p.replace_asr(new_asr, source_lang))
                self.asr = new_asr
            elif source_lang != old_source:
                self._status(f"Switching recognition to {spoken}…")
                try:
                    self.asr.set_language(source_lang)  # same model, no reload
                    p.replace_asr(self.asr, source_lang)  # same object: only updates the pipeline's language
                except NotImplementedError:
                    new_asr = self.asr_ladder.load(max(self.asr_ladder.index, 0))
                    _close(p.replace_asr(new_asr, source_lang))
                    self.asr = new_asr
            if self._wants_translation(target_lang) and self.translator is None:
                self._status(f"Loading the translator for {languages.LABELS[target_lang]} (about 20 s)…")
                self.mt_ladder = self._mt_ladder()
                self.translator = self.mt_ladder.load()
                p.replace_translator(self.translator, target_lang)
            else:
                p.set_target(target_lang)  # retarget; with translation off the translator just waits

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


def _quietly(load) -> None:
    try:
        load()
    except Exception as exc:  # captions then just have no readings
        log.warning("Furigana unavailable: %s", exc)
