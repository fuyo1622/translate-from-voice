"""Wires a source, the segmenter and a recognizer together on background threads.

    source thread ──audio──► segmenter thread ──utterances──► recognizer thread ──► on_utterance

The capture side never blocks. If recognition falls behind live audio, the
oldest waiting utterances are dropped (and counted) so captions stay current.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from tfv.capture.base import AudioSource
from tfv.engine.asr.base import ASRBackend, ASRResult
from tfv.engine.segmenter import Segmenter, SpeechSegment

log = logging.getLogger(__name__)

_END = object()


@dataclass
class Utterance:
    segment: SpeechSegment
    result: ASRResult
    text: str
    """Display-ready text (post-processed)."""
    queued_s: float
    """How long the utterance waited for the recognizer."""


class Pipeline:
    def __init__(
        self,
        source: AudioSource,
        segmenter: Segmenter,
        asr: ASRBackend,
        on_utterance: Callable[[Utterance], None],
        postprocess: Callable[[str, str], str | None] = lambda text, lang: text or None,
        idle_flush_s: float = 0.6,
        max_waiting: int = 4,
    ):
        self._source = source
        self._segmenter = segmenter
        self._asr = asr
        self._on_utterance = on_utterance
        self._postprocess = postprocess
        self._idle_flush_s = idle_flush_s
        self._audio: queue.Queue = queue.Queue()
        # files are processed completely; live audio drops stale utterances instead
        self._utterances: queue.Queue = queue.Queue(maxsize=max_waiting if source.realtime else 0)
        self._stopping = threading.Event()
        self._done = threading.Event()
        self._threads: list[threading.Thread] = []
        self.dropped = 0

    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._segment_loop, name="tfv-segmenter", daemon=True),
            threading.Thread(target=self._asr_loop, name="tfv-asr", daemon=True),
        ]
        for t in self._threads:
            t.start()
        self._source.start(self._audio.put, lambda: self._audio.put(_END))

    def wait(self, timeout: float | None = None) -> bool:
        """True once the source has ended and every utterance has been handled."""
        return self._done.wait(timeout)

    def stop(self) -> None:
        """Stop now, discarding utterances that have not been recognized yet."""
        self._stopping.set()
        self._source.stop()
        self._audio.put(_END)
        for t in self._threads:
            t.join(timeout=5)

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
                    self._enqueue(self._segmenter.flush())
                    idle_handled = True
                continue
            if chunk is _END:
                self._enqueue(self._segmenter.flush())
                self._utterances.put(_END)
                return
            last_audio, idle_handled = time.monotonic(), False
            for segment in self._segmenter.push(chunk):
                self._enqueue(segment)

    def _enqueue(self, segment: SpeechSegment | None) -> None:
        if segment is None or self._stopping.is_set():
            return
        while True:
            try:
                self._utterances.put_nowait(segment)
                return
            except queue.Full:
                try:
                    self._utterances.get_nowait()
                    self.dropped += 1
                    log.warning("Recognition is falling behind; dropped an utterance (%d so far)", self.dropped)
                except queue.Empty:
                    pass

    def _asr_loop(self) -> None:
        try:
            while (segment := self._utterances.get()) is not _END:
                if self._stopping.is_set():
                    continue
                queued = time.monotonic() - segment.created_at
                try:
                    result = self._asr.transcribe(segment.audio)
                except Exception:
                    log.exception("Recognition failed for a %.1fs utterance", segment.duration)
                    continue
                text = self._postprocess(result.text, result.language)
                if text:
                    self._on_utterance(Utterance(segment, result, text, queued))
        finally:
            self._done.set()
