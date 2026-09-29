"""Wires capture, segmentation, recognition and translation together on background threads.

    source ──audio──► segmenter thread ──utterances──► ASR thread ──captions──► MT thread
                                                          │                        │
                                                          └──── on_caption ◄───────┘

The capture side never blocks. When recognition or translation falls behind
live audio, the oldest waiting work is dropped (and counted) so captions stay
current; `stats()` exposes the numbers the resource manager uses to downgrade.
Backends can be swapped while running (see replace_asr / replace_translator).
"""

from __future__ import annotations

import collections
import itertools
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from tfv.capture.base import AudioSource
from tfv.engine.asr.base import ASRBackend
from tfv.engine.mt.base import Translator
from tfv.engine.segmenter import Segmenter, SpeechSegment

log = logging.getLogger(__name__)

_END = object()


@dataclass
class Caption:
    id: int
    start: float
    """Seconds since the stream started."""
    end: float
    forced: bool
    """Cut mid-speech at the maximum length; the sentence continues in the next caption."""
    source_lang: str
    text: str
    target_lang: str | None = None
    translation: str | None = None
    stage: str = "text"
    """"text" (recognized), "translated", or "untranslated" (translation failed or was skipped)."""
    asr_s: float = 0.0
    mt_s: float = 0.0
    waited_s: float = 0.0
    """How long the utterance waited for the recognizer."""
    speech_ended_at: float = field(default_factory=time.monotonic)

    @property
    def duration(self) -> float:
        return self.end - self.start


class Pipeline:
    def __init__(
        self,
        source: AudioSource,
        segmenter: Segmenter,
        asr: ASRBackend,
        on_caption: Callable[[Caption], None],
        *,
        source_lang: str,
        target_lang: str | None = None,
        translator: Translator | None = None,
        postprocess: Callable[[str, str], str | None] = lambda text, lang: text or None,
        idle_flush_s: float = 0.6,
        max_waiting: int = 4,
        context_lines: int = 3,
    ):
        self._source = source
        self._segmenter = segmenter
        self._asr = asr
        self._translator = translator
        self._on_caption = on_caption
        self._postprocess = postprocess
        self.source_lang = source_lang
        self.target_lang = target_lang
        self._idle_flush_s = idle_flush_s
        live = source.realtime
        # files are processed completely; live audio drops stale work instead
        self._utterances: queue.Queue = queue.Queue(maxsize=max_waiting if live else 0)
        self._to_translate: queue.Queue = queue.Queue(maxsize=max_waiting + 2 if live else 0)
        self._audio: queue.Queue = queue.Queue()
        self._asr_lock = threading.Lock()
        self._mt_lock = threading.Lock()
        self._context: collections.deque[tuple[str, str]] = collections.deque(maxlen=context_lines)
        self._last_translated_end = 0.0
        self._ids = itertools.count(1)
        self._recent: collections.deque[Caption] = collections.deque(maxlen=12)
        self._stopping = threading.Event()
        self._paused = threading.Event()
        self._done = threading.Event()
        self._threads: list[threading.Thread] = []
        self.dropped_utterances = 0
        self.dropped_translations = 0

    # -- control -------------------------------------------------------------

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._segment_loop, name="tfv-segmenter", daemon=True),
            threading.Thread(target=self._asr_loop, name="tfv-asr", daemon=True),
            threading.Thread(target=self._mt_loop, name="tfv-mt", daemon=True),
        ]
        for t in self._threads:
            t.start()
        self._source.start(self._audio.put, lambda: self._audio.put(_END))

    def wait(self, timeout: float | None = None) -> bool:
        """True once the source has ended and every caption has been handled."""
        return self._done.wait(timeout)

    def stop(self) -> None:
        """Stop now, discarding work that has not been done yet."""
        self._stopping.set()
        self._source.stop()
        self._audio.put(_END)
        for t in self._threads:
            t.join(timeout=5)

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def pause(self) -> None:
        """Ignore incoming audio until resume(); captions already in progress still finish."""
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def replace_asr(self, asr: ASRBackend, source_lang: str | None = None) -> ASRBackend:
        """Swap the recognizer once the current utterance is done; returns the old one to close."""
        with self._asr_lock:
            old, self._asr = self._asr, asr
            if source_lang and source_lang != self.source_lang:
                self.source_lang = source_lang
                self._context.clear()
        return old

    def replace_translator(self, translator: Translator | None, target_lang: str | None = None) -> Translator | None:
        """Swap the translator (or turn translation off with None); returns the old one to close."""
        with self._mt_lock:
            old, self._translator = self._translator, translator
            if target_lang:
                self.target_lang = target_lang
            self._context.clear()
        return old

    def set_target(self, target_lang: str | None) -> None:
        with self._mt_lock:
            self.target_lang = target_lang
            self._context.clear()

    def stats(self) -> dict:
        """Recent performance: how fast recognition/translation run and what was dropped."""
        recent = list(self._recent)
        speech = sum(c.duration for c in recent) or 1.0
        translated = [c for c in recent if c.stage == "translated"]
        return {
            "captions": len(recent),
            "translated": len(translated),
            "asr_rtf": sum(c.asr_s for c in recent) / speech,
            "asr_s": sum(c.asr_s for c in recent) / len(recent) if recent else 0.0,
            "mt_s": sum(c.mt_s for c in translated) / len(translated) if translated else 0.0,
            "waited_s": max((c.waited_s for c in recent), default=0.0),
            "backlog": self._utterances.qsize() + self._to_translate.qsize(),
            "dropped_utterances": self.dropped_utterances,
            "dropped_translations": self.dropped_translations,
        }

    def reset_stats(self) -> None:
        """Forget recent timings, e.g. after swapping a backend."""
        self._recent.clear()

    # -- threads ---------------------------------------------------------------

    def _segment_loop(self) -> None:
        last_audio = time.monotonic()
        idle_handled = True
        while True:
            try:
                chunk = self._audio.get(timeout=0.1)
            except queue.Empty:
                # WASAPI sends nothing while nothing plays, so silence never
                # arrives to close the utterance; close it on a timeout instead.
                if not idle_handled and time.monotonic() - last_audio >= self._idle_flush_s:
                    self._put_latest(self._utterances, self._segmenter.flush(), "utterance")
                    idle_handled = True
                continue
            if chunk is _END:
                self._put_latest(self._utterances, self._segmenter.flush(), "utterance")
                self._utterances.put(_END)
                return
            if self._paused.is_set():
                self._segmenter.flush()  # discard the partial utterance
                continue
            last_audio, idle_handled = time.monotonic(), False
            for segment in self._segmenter.push(chunk):
                self._put_latest(self._utterances, segment, "utterance")

    def _asr_loop(self) -> None:
        try:
            while (segment := self._utterances.get()) is not _END:
                if not self._stopping.is_set():
                    self._recognize(segment)
        finally:
            self._to_translate.put(_END)

    def _recognize(self, segment: SpeechSegment) -> None:
        waited = time.monotonic() - segment.created_at
        with self._asr_lock:
            source_lang = self.source_lang
            try:
                result = self._asr.transcribe(segment.audio)
            except Exception:
                log.exception("Recognition failed for a %.1fs utterance", segment.duration)
                return
        text = self._postprocess(result.text, source_lang)
        if not text:
            return
        caption = Caption(
            id=next(self._ids), start=segment.start, end=segment.end, forced=segment.forced,
            source_lang=source_lang, text=text, asr_s=result.compute_s, waited_s=waited,
            speech_ended_at=segment.created_at,
        )  # fmt: skip
        self._recent.append(caption)
        target = self.target_lang
        translate = self._translator is not None and bool(target) and target != source_lang
        if translate:
            caption.target_lang = target  # lets displays show that a translation is coming
        self._emit(caption)
        if translate:
            self._put_latest(self._to_translate, caption, "translation")

    def _mt_loop(self) -> None:
        try:
            while (caption := self._to_translate.get()) is not _END:
                if not self._stopping.is_set():
                    self._translate(caption)
        finally:
            self._done.set()

    def _translate(self, caption: Caption) -> None:
        with self._mt_lock:
            translator, target = self._translator, self.target_lang
            if translator is None or target != caption.target_lang:
                # translation was turned off or retargeted meanwhile
                caption.stage = "untranslated"
                self._emit(caption)
                return
            if caption.start - self._last_translated_end > 30:
                self._context.clear()  # a long pause usually means a new scene
            context = list(self._context) if translator.uses_context else []
            t0 = time.perf_counter()
            try:
                translation = translator.translate(caption.text, caption.source_lang, target, context)
            except Exception as exc:
                log.warning("Translation failed: %s", exc)
                translation = ""
            caption.mt_s = time.perf_counter() - t0
            text = self._postprocess(translation, target) if translation else None
            if text:
                self._context.append((caption.text, text))
                self._last_translated_end = caption.end
        caption.translation = text
        caption.stage = "translated" if text else "untranslated"
        self._emit(caption)

    def _emit(self, caption: Caption) -> None:
        try:
            self._on_caption(caption)
        except Exception:
            log.exception("Caption handler failed")

    def _put_latest(self, q: queue.Queue, item, what: str) -> None:
        """Queue `item`; if the queue is full, drop the oldest entry to make room."""
        if item is None or self._stopping.is_set():
            return
        while True:
            try:
                q.put_nowait(item)
                return
            except queue.Full:
                try:
                    dropped = q.get_nowait()
                except queue.Empty:
                    continue
                if what == "utterance":
                    self.dropped_utterances += 1
                else:
                    self.dropped_translations += 1
                    dropped.stage = "untranslated"
                    self._emit(dropped)
                log.warning("Falling behind live audio; skipped one %s", what)
