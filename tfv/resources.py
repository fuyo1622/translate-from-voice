"""Resource manager: choose a hardware profile, load the best rung that works, adapt while running.

A *rung* is one way to run a stage ("whisper-large-v3-turbo@cuda"). Loading
walks the profile's ladder from the top and skips rungs that do not fit in
free memory, fail to load, or benchmark too slowly for live audio. While
running, `Monitor` watches the pipeline and steps down when a stage cannot
keep up. It never steps back up by itself (restart or change settings for that).
"""

from __future__ import annotations

import logging
import threading
import time
import wave
from dataclasses import dataclass, field
from importlib import resources as package_files
from pathlib import Path
from typing import Callable, Generic, TypeVar

import numpy as np

from tfv import hardware, languages, models

log = logging.getLogger(__name__)

CLOUD_BACKENDS = {"deepl", "claude", "openai", "deepgram"}


@dataclass(frozen=True)
class Choice:
    backend: str
    """faster-whisper, sensevoice, nllb, llama, openai-compatible, or a cloud service."""
    model: str | None
    device: str
    """cuda, gpu, cpu or cloud."""

    @classmethod
    def parse(cls, text: str) -> Choice:
        """"<catalog model>@<device>", "<model dir>@<device>" or "<service>[:<model>]"."""
        name, _, device = text.strip().partition("@")
        if name in models.CATALOG:
            info = models.CATALOG[name]
            default = "cpu" if info.engine in ("sensevoice", "nllb") else "gpu" if info.engine == "llama" else "cuda"
            return cls(info.engine, name, device or default)
        if Path(name).is_dir():  # a local CTranslate2 Whisper model
            return cls("faster-whisper", name, device or "cuda")
        backend, _, model = name.partition(":")
        if backend in CLOUD_BACKENDS or backend == "openai-compatible":
            return cls(backend, model or None, "cloud")
        raise ValueError(f"unknown model or service {text!r}; see `python -m tfv models`")

    def __str__(self) -> str:
        if self.device == "cloud":
            return f"{self.backend}:{self.model}" if self.model else self.backend
        return f"{self.model}@{self.device}"


@dataclass
class Profile:
    name: str
    summary: str
    when: dict
    asr: list[Choice]
    mt: list[Choice]
    auto: bool = True
    asr_by_lang: dict[str, list[Choice]] = field(default_factory=dict)
    """Ladders for a particular spoken language (profiles.yaml keys asr_zh, asr_en, asr_ja)."""
    mt_by_lang: dict[str, list[Choice]] = field(default_factory=dict)
    """Ladders for a particular target language (keys mt_zh, mt_en, mt_ja)."""

    def asr_for(self, source_lang: str) -> list[Choice]:
        return self.asr_by_lang.get(source_lang, self.asr)

    def mt_for(self, target_lang: str | None) -> list[Choice]:
        return self.mt_by_lang.get(target_lang or "", self.mt)

    def matches(self, hw: hardware.Hardware) -> bool:
        gpu = hw.gpu
        checks = {
            "os": lambda v: hw.os in v,
            "arch": lambda v: hw.arch in v,
            "gpu_vendor": lambda v: gpu is not None and gpu.vendor in v,
            "min_vram_mb": lambda v: gpu is not None and gpu.vram_mb >= v,
            "cuda": lambda v: hw.cuda == v,
            "min_ram_mb": lambda v: hw.ram_mb >= v,
            "min_cores": lambda v: hw.physical_cores >= v,
        }
        unknown = set(self.when) - set(checks)
        if unknown:
            raise ValueError(f"profile {self.name}: unknown condition(s) {', '.join(sorted(unknown))}")
        return all(checks[key](value) for key, value in self.when.items())


def load_profiles(path: Path | None = None) -> list[Profile]:
    import yaml

    text = Path(path).read_text("utf-8") if path else package_files.files("tfv").joinpath("profiles.yaml").read_text("utf-8")

    def ladder(items) -> list[Choice]:
        return [Choice.parse(c) for c in items]

    return [
        Profile(
            name=p["name"], summary=p.get("summary", ""), when=p.get("when") or {},
            asr=ladder(p["asr"]), mt=ladder(p["mt"]), auto=p.get("auto", True),
            asr_by_lang={lang: ladder(p[f"asr_{lang}"]) for lang in languages.SUPPORTED if f"asr_{lang}" in p},
            mt_by_lang={lang: ladder(p[f"mt_{lang}"]) for lang in languages.SUPPORTED if f"mt_{lang}" in p},
        )  # fmt: skip
        for p in yaml.safe_load(text)["profiles"]
    ]


def choose_profile(hw: hardware.Hardware, name: str = "auto") -> Profile:
    profiles = load_profiles()
    if name != "auto":
        for p in profiles:
            if p.name == name:
                return p
        raise ValueError(f"unknown profile {name!r}; choose from {', '.join(p.name for p in profiles)}")
    for p in profiles:
        if p.auto and p.matches(hw):
            return p
    raise RuntimeError("no profile matches this machine")  # cpu-low matches everything


# -- fitting -------------------------------------------------------------------

VRAM_MARGIN_MB = 300


def unfit_reason(choice: Choice, hw: hardware.Hardware, has_key: Callable[[str], bool] = lambda s: False) -> str | None:
    """Why `choice` cannot run on this machine right now, or None if it should."""
    if choice.device == "cloud":
        if choice.backend == "openai-compatible":
            return None
        return None if has_key(choice.backend) else f"no API key for {choice.backend}"
    info = models.CATALOG.get(choice.model or "")
    size = info.size_mb if info else 1600
    if choice.device == "cuda":
        if not hw.cuda:
            return f"CUDA unavailable ({hw.cuda_note or 'no NVIDIA GPU'})"
        free = hardware.nvidia_free_mb()
        need = int(size * 0.55) + 300  # int8 weights + working buffers
        if free is not None and free - VRAM_MARGIN_MB < need:
            return f"needs ~{need} MB VRAM, {free} MB free"
    elif choice.device == "gpu":
        from tfv.engine.mt.llama_server import best_device, ensure_runtime, list_devices

        try:
            gpu = best_device(list_devices(ensure_runtime()))
        except Exception as exc:
            return f"no GPU runtime ({exc})"
        if gpu is None:
            return "no discrete GPU"
        free = gpu.free_mib
        if "nvidia" in gpu.name.lower():
            # Vulkan's figure misses what CUDA (Whisper) holds; nvidia-smi sees all of it.
            # Overcommitting makes Windows page VRAM and everything becomes 10x slower.
            free = hardware.nvidia_free_mb() or free
        need = size + 300  # weights + 4k context + compute buffers (measured +60..230 MB, EVAL.md)
        if free - VRAM_MARGIN_MB < need:
            return f"needs ~{need} MB VRAM, {free} MB free on {gpu.name}"
    else:  # cpu
        import psutil

        free = psutil.virtual_memory().available // 2**20
        if free < size + 800:
            return f"needs ~{size + 800} MB RAM, {free} MB free"
    return None


# -- benchmarks -----------------------------------------------------------------

ASR_MAX_RTF = 0.5
"""Recognition must run at least twice as fast as real time to keep up with live speech."""
MT_MAX_S = 3.0

_BENCH_TEXT = {
    "zh": "今天的天氣很好，我們一起去公園散步吧。",
    "en": "Today we are going to talk about how machine translation works in real time.",
    "ja": "今日はリアルタイム翻訳の仕組みについてお話しします。",
}


def bench_audio(language: str) -> np.ndarray:
    """A 7-9 s speech clip in `language` (Windows text-to-speech), 16 kHz mono."""
    clip = package_files.files("tfv").joinpath(f"assets/bench_{language}.wav")
    with package_files.as_file(clip) as path, wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def bench_asr(asr, language: str) -> tuple[float, str]:
    """Real-time factor on a short clip (compute time / audio time).

    The clip must be in the recognizer's language: Whisper forced to the
    wrong language falls back to slow retries and would look too slow.
    """
    audio = bench_audio(language)
    t0 = time.perf_counter()
    asr.transcribe(audio)
    rtf = (time.perf_counter() - t0) / (len(audio) / 16000)
    return rtf, f"{rtf:.2f}x real time"


def bench_mt(translator, source: str, target: str) -> tuple[float, str]:
    t0 = time.perf_counter()
    translator.translate(_BENCH_TEXT[source], source, target)
    seconds = time.perf_counter() - t0
    return seconds, f"{seconds:.2f}s per line"


# -- ladders ---------------------------------------------------------------------

T = TypeVar("T")


@dataclass
class Ladder(Generic[T]):
    """Loads the first workable rung and can step down to the next ones later."""

    kind: str
    """"recognition" or "translation", for messages."""
    rungs: list[Choice]
    build: Callable[[Choice], T]
    """Load and warm up a backend; raise if it cannot run."""
    unfit: Callable[[Choice], str | None]
    bench: Callable[[T], tuple[float, str]] | None = None
    limit: float = 0.0
    status: Callable[[str], None] = lambda msg: log.info("%s", msg)
    index: int = -1
    skipped: list[str] = field(default_factory=list)

    @property
    def current(self) -> Choice | None:
        return self.rungs[self.index] if 0 <= self.index < len(self.rungs) else None

    @property
    def can_step_down(self) -> bool:
        return self.index + 1 < len(self.rungs)

    def load(self, start: int = 0) -> T:
        """Load the first rung from `start` that fits, loads and is fast enough."""
        errors = []
        for i in range(start, len(self.rungs)):
            choice = self.rungs[i]
            last = i == len(self.rungs) - 1
            if not last and (reason := self.unfit(choice)):
                self._skip(choice, reason, errors)
                continue
            self.status(f"Loading {self.kind}: {choice}…")
            try:
                backend = self.build(choice)
            except Exception as exc:
                log.debug("Loading %s failed", choice, exc_info=True)
                self._skip(choice, f"failed to load: {exc}", errors)
                continue
            if self.bench and not last:
                score, note = self.bench(backend)
                if score > self.limit:
                    _close(backend)
                    self._skip(choice, f"too slow ({note})", errors)
                    continue
                log.info("%s %s: %s", self.kind.capitalize(), choice, note)
            self.index = i
            return backend
        raise RuntimeError(f"no {self.kind} option works here: " + "; ".join(errors))

    def step_down(self) -> T | None:
        """Load the next rung below the current one; None if there is none."""
        if not self.can_step_down:
            return None
        return self.load(self.index + 1)

    def _skip(self, choice: Choice, reason: str, errors: list[str]) -> None:
        message = f"{choice}: {reason}"
        errors.append(message)
        self.skipped.append(message)
        log.info("Skipping %s %s", self.kind, message)


def _close(backend) -> None:
    close = getattr(backend, "close", None)
    if close:
        close()


# -- runtime adaptation ---------------------------------------------------------------


class Monitor:
    """Watches a live pipeline and asks for a lighter rung when a stage falls behind."""

    def __init__(self, pipeline, step_down: Callable[[str, str], bool], interval: float = 5.0, cooldown: float = 30.0):
        self._pipeline = pipeline
        self._step_down = step_down
        self._interval = interval
        self._cooldown = cooldown
        self._stop = threading.Event()
        self._last_change = time.monotonic()
        self._dropped = (0, 0, 0)
        self._thread = threading.Thread(target=self._run, name="tfv-monitor", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.check()
            except Exception:
                log.exception("Resource monitor failed")

    def check(self) -> None:
        stats = self._pipeline.stats()
        counts = (stats["dropped_utterances"], stats["dropped_translations"], stats.get("failed_translations", 0))
        new_asr_drops, new_mt_drops, new_mt_failures = (now - before for now, before in zip(counts, self._dropped))
        self._dropped = counts
        if time.monotonic() - self._last_change < self._cooldown:
            return
        stage, reason = None, ""
        if new_asr_drops or (stats["captions"] >= 4 and stats["asr_rtf"] > 0.7):
            stage, reason = "asr", f"recognition at {stats['asr_rtf']:.2f}x real time, {new_asr_drops} skipped"
        elif new_mt_failures >= 3:
            stage, reason = "mt", f"{new_mt_failures} translations failed"
        elif new_mt_drops or (stats["translated"] >= 3 and stats["mt_s"] > 4.0):
            stage, reason = "mt", f"translation takes {stats['mt_s']:.1f}s per line, {new_mt_drops} skipped"
        if stage and self._step_down(stage, reason):
            self._last_change = time.monotonic()
            self._pipeline.reset_stats()
