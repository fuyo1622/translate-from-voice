"""Wires capture, segmentation, recognition and translation together on background threads.

    source ──audio──► segmenter thread ──utterances──► ASR thread ──captions──► MT thread
                            │ (live snapshots)          ▲  │                        │
                            └───────────────────────────┘  └──── on_caption ◄──────┘

The capture side never blocks. When recognition or translation falls behind
live audio, the oldest waiting work is dropped (and counted) so captions stay
current; `stats()` exposes the numbers the resource manager uses to downgrade.
Backends can be swapped while running (see replace_asr / replace_translator).

With `partials` on, the utterance still being spoken is recognized again every
`partial_interval` seconds and emitted as a "partial" caption (live text) with
the id its final caption will have; finished utterances always go first.

With a `separator`, lines in which the VAD's denoiser found music are recognized
from the voices alone, with the music taken out (songs, talk over loud music).
If the separator is fast enough, song mode goes further while music plays: the
VAD and the recognizer hear the voices separated a second at a time, which
finds much more of the singing than the denoised copy does.
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

import numpy as np

from tfv import SAMPLE_RATE
from tfv.capture.base import AudioSource
from tfv.engine.asr.base import ASRBackend
from tfv.engine.mt.base import Translator
from tfv.engine.segmenter import Segmenter, SpeechSegment
from tfv.engine.separate import StreamingVoices

log = logging.getLogger(__name__)

_END = object()
LIVE_SEPARATION_RTF = 0.25
"""Live text uses the separator only if it takes at most this long per second of audio."""
SONG_ENTER_DB = 8.0
"""Song mode starts when the VAD's denoiser removes this much (dB) over the last 2 s: at least 13 dB
under music as loud as singing, 5-14 dB under music as loud as speech, 0-2 dB for most clean speech."""
SONG_EXIT_DB = 4.0
SONG_EXIT_S = 3.0
"""... and ends after the removal has stayed under SONG_EXIT_DB this long."""
SONG_MAX_RUN_S = 0.75
"""Song mode needs a model run every ~0.75 s; slower separators only do whole lines."""
SONG_VAD_THRESHOLD = 0.3
"""In song mode the VAD hears separated voices, which it can trust more: on a song with music 5 dB
louder than the singing, 0.3 found 81% of it (0.5: 64%), with fewer errors (EVAL.md)."""


class _MusicLevel:
    """How much the VAD's denoiser removed from the last couple of seconds of audio, in dB."""

    def __init__(self, seconds: float = 2.0):
        self._parts: collections.deque[tuple[int, float, float]] = collections.deque()
        self._limit = int(seconds * SAMPLE_RATE)

    def add(self, original: np.ndarray, clean: np.ndarray) -> None:
        self._parts.append((len(original), float(original @ original), float(clean @ clean)))
        while sum(n for n, _, _ in self._parts) - self._parts[0][0] >= self._limit:
            self._parts.popleft()

    @property
    def db(self) -> float | None:
        """None while it is (nearly) silent."""
        n = sum(p[0] for p in self._parts)
        heard, kept = sum(p[1] for p in self._parts), sum(p[2] for p in self._parts)
        if not n or heard < n * 1e-7:
            return None
        return 10 * float(np.log10(heard / max(kept, 1e-12)))

    def clear(self) -> None:
        self._parts.clear()


@dataclass
class _Discarded:
    """The segmenter dropped this utterance as noise: withdraw its live text."""

    utterance: int


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
    """"partial" (still being spoken), "text" (recognized), "translated", "untranslated"
    (translation failed or was skipped), or "dropped" (the live text was noise: remove it)."""
    ruby: list | None = None
    """For Japanese: `text` split into (text, reading or None) pairs, for furigana."""
    asr_s: float = 0.0
    sep_s: float = 0.0
    """Seconds spent taking the music out first (0: the line had none, or no separator)."""
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
        readings: Callable[[str], list] | None = None,
        vad_denoiser=None,
        separator=None,
        music_db: float = 4.0,
        partials: bool = False,
        partial_interval: float = 0.6,
        idle_flush_s: float = 0.6,
        history_s: float = 60.0,
        max_waiting: int = 4,
        context_lines: int = 3,
    ):
        self._source = source
        self._segmenter = segmenter
        self._asr = asr
        self._translator = translator
        self._on_caption = on_caption
        self._postprocess = postprocess
        self._readings = readings
        """Furigana for Japanese captions (see tfv.engine.furigana)."""
        self._denoiser = vad_denoiser
        """Cleans what the VAD hears, so speech under music isn't missed (see tfv.engine.denoise)."""
        self.separator = separator
        """Takes the music out from under voices before recognition (see tfv.engine.separate), on
        lines from which the denoiser removed at least `music_db` (all lines without a denoiser).
        Settable while running; None turns it off."""
        self.separator_error: str | None = None
        """Why the separator was turned off while running, if it failed."""
        self._music_db = music_db
        self.song_mode_allowed = True
        """Use song mode when the separator is fast enough (settable while running)."""
        self._song: StreamingVoices | None = None  # song mode is on (segmenter thread only)
        self._vad_threshold = segmenter.config.threshold  # outside song mode
        self._level = _MusicLevel()
        self._quiet_samples = 0  # in song mode: how long the music has been gone
        self.song_mode_s = 0.0
        """Seconds of audio heard in song mode so far."""
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.partials = partials
        """Emit live text for the utterance still being spoken (settable while running)."""
        self._partial_interval = partial_interval
        self._partial_job: SpeechSegment | None = None  # the latest snapshot waiting for the recognizer
        self._partial_lock = threading.Lock()
        self._partial_cost = 0.0  # seconds the last live recognition took
        self._last_partial_at = 0.0
        self._caption_ids: dict[int, int] = {}  # utterance number -> caption id (ASR thread only)
        self._last_final = 0  # highest utterance number recognized in full
        self._idle_flush_s = idle_flush_s
        live = source.realtime
        # files are processed completely; live audio drops stale work instead
        self._utterances: queue.Queue = queue.Queue(maxsize=max_waiting if live else 0)
        self._to_translate: queue.Queue = queue.Queue(maxsize=max_waiting + 2 if live else 0)
        self._audio: queue.Queue = queue.Queue()
        self._history = np.zeros(int(history_s * SAMPLE_RATE), np.float32)
        """The last minute of audio (a ring buffer), for "save the last minute" feedback."""
        self._history_pos = 0
        self._history_filled = False
        self._history_total = 0  # samples heard since the start, i.e. the stream time captions use
        self._history_lock = threading.Lock()
        self._caption_log: collections.deque[Caption] = collections.deque(maxlen=200)
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
        self.failed_translations = 0

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

    def recent(self) -> tuple[np.ndarray, float, list[Caption]]:
        """The last minute of audio (oldest first), the stream time it starts at, and the
        captions finished in it."""
        with self._history_lock:
            if self._history_filled:
                audio = np.concatenate([self._history[self._history_pos :], self._history[: self._history_pos]])
            else:
                audio = self._history[: self._history_pos].copy()
            start = (self._history_total - len(audio)) / SAMPLE_RATE
        logged = list(self._caption_log)  # a copy: other threads keep adding captions
        captions = [c for c in logged if c.stage in ("text", "translated", "untranslated") and c.end > start]
        return audio, start, captions

    def _remember(self, chunk: np.ndarray) -> None:
        with self._history_lock:
            n, size = len(chunk), len(self._history)
            self._history_total += n
            if n >= size:
                self._history[:] = chunk[-size:]
                self._history_pos, self._history_filled = 0, True
                return
            end = self._history_pos + n
            if end <= size:
                self._history[self._history_pos : end] = chunk
            else:
                self._history[self._history_pos :] = chunk[: size - self._history_pos]
                self._history[: end - size] = chunk[size - self._history_pos :]
                self._history_filled = True
            self._history_pos = end % size
            self._history_filled = self._history_filled or end == size

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
            "separated": sum(1 for c in recent if c.sep_s),
            "sep_s": sum(c.sep_s for c in recent) / max(1, sum(1 for c in recent if c.sep_s)),
            "mt_s": sum(c.mt_s for c in translated) / len(translated) if translated else 0.0,
            "waited_s": max((c.waited_s for c in recent), default=0.0),
            "backlog": self._utterances.qsize() + self._to_translate.qsize(),
            "song_mode": self._song is not None,
            "song_mode_s": self.song_mode_s,
            "dropped_utterances": self.dropped_utterances,
            "dropped_translations": self.dropped_translations,
            "failed_translations": self.failed_translations,
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
                    self._close_utterance()
                    idle_handled = True
                continue
            if chunk is _END:
                self._close_utterance()
                self._utterances.put(_END)
                return
            if self._paused.is_set():
                self._close_utterance(keep=False)
                continue
            last_audio, idle_handled = time.monotonic(), False
            self._remember(chunk)
            if self._denoiser is not None:  # the VAD judges a denoised copy; utterances keep the original
                original, clean = self._denoiser.push(chunk)
                self._level.add(original, clean)
            else:
                original, clean = chunk, None
            self._update_song_mode(len(original))
            if self._song is not None:
                segments = self._push_song(original)
            else:
                segments = self._segmenter.push(original, clean)
            for segment in segments:
                self._put_latest(self._utterances, segment, "utterance")
            self._forward_discarded()
            if self.partials and self._source.realtime:
                self._offer_partial()

    def _update_song_mode(self, samples: int) -> None:
        """Start song mode while music plays, if the separator can keep up; end it after the music."""
        level = self._level.db
        if self._song is None:
            separator = self.separator
            if (
                separator is None or not self.song_mode_allowed or level is None or level < SONG_ENTER_DB
                or getattr(separator, "run_s", 1.0) > SONG_MAX_RUN_S
            ):  # fmt: skip
                return
            self._finish_segment()  # what was said before goes the usual way
            hop = max(0.75, 1.6 * separator.run_s)  # leaves the card time for the translator
            self._song, self._quiet_samples = StreamingVoices(separator, hop=hop), 0
            self._vad_threshold = self._segmenter.config.threshold
            self._segmenter.set_threshold(min(self._vad_threshold, SONG_VAD_THRESHOLD))
            log.info("Song mode on (the denoiser removed %.0f dB; a separator run every %.2f s)", level, hop)
            return
        self.song_mode_s += samples / SAMPLE_RATE
        self._quiet_samples = self._quiet_samples + samples if level is not None and level < SONG_EXIT_DB else 0
        if self.separator is None or not self.song_mode_allowed or self._quiet_samples >= SONG_EXIT_S * SAMPLE_RATE:
            self._end_song_mode()

    def _push_song(self, audio: np.ndarray) -> list[SpeechSegment]:
        try:
            blocks = self._song.push(audio)
        except Exception as exc:
            self._song_failed(exc)
            return []
        segments = []
        for block in blocks:
            segments += self._segmenter.push(block, block)
        for segment in segments:
            segment.separated = True
        return segments

    def _end_song_mode(self, keep: bool = True) -> None:
        song, self._song = self._song, None
        if song is None:
            return
        log.info("Song mode off")
        self._segmenter.set_threshold(self._vad_threshold)
        try:
            block = song.flush() if keep else np.zeros(0, np.float32)
        except Exception as exc:
            self._song_failed(exc)
            return
        segments = self._segmenter.push(block, block) if len(block) else []
        segments.append(self._segmenter.flush())
        for segment in segments:
            if segment is None:
                continue
            segment.separated = True
            if keep:
                self._put_latest(self._utterances, segment, "utterance")
            else:
                self._segmenter.discarded.append(segment.utterance)

    def _song_failed(self, exc: Exception) -> None:
        log.exception("Separating the voices failed; recognizing the original audio from now on")
        self._song = None
        self._segmenter.set_threshold(self._vad_threshold)
        self.separator, self.separator_error = None, str(exc)
        if (segment := self._segmenter.flush()) is not None:
            self._segmenter.discarded.append(segment.utterance)

    def _finish_segment(self) -> None:
        """End the utterance in progress, keeping the denoiser running (a switch of what the VAD hears)."""
        if (segment := self._segmenter.flush()) is not None:
            self._put_latest(self._utterances, segment, "utterance")

    def _close_utterance(self, keep: bool = True) -> None:
        """The audio went quiet, ended or was paused: finish the utterance in progress (or drop it)."""
        if self._song is not None:
            self._end_song_mode(keep)
        self._level.clear()
        done = []
        if self._denoiser is not None:
            original, clean = self._denoiser.flush()
            done += self._segmenter.push(original, clean) if len(original) else []
        done.append(self._segmenter.flush())
        for segment in done:
            if segment is None:
                continue
            if keep:
                self._put_latest(self._utterances, segment, "utterance")
            else:
                self._segmenter.discarded.append(segment.utterance)  # withdraw its live text
        self._forward_discarded()

    def _forward_discarded(self) -> None:
        for utterance in self._segmenter.discarded:
            try:
                self._utterances.put_nowait(_Discarded(utterance))
            except queue.Full:
                pass  # recognition is behind; the next finished utterance withdraws it
        self._segmenter.discarded.clear()

    def _offer_partial(self) -> None:
        """Hand the recognizer the utterance spoken so far, at most once per interval."""
        now = time.monotonic()
        if now - self._last_partial_at < max(self._partial_interval, 1.5 * self._partial_cost):
            return
        snapshot = self._segmenter.snapshot()
        if snapshot is None or snapshot.duration < 0.8:
            return
        snapshot.separated = self._song is not None
        with self._partial_lock:
            self._partial_job = snapshot  # replaces one the recognizer hasn't got to yet
        self._last_partial_at = now

    def _take_partial(self) -> SpeechSegment | None:
        with self._partial_lock:
            job, self._partial_job = self._partial_job, None
        return job

    def _asr_loop(self) -> None:
        try:
            while True:
                try:
                    segment = self._utterances.get(timeout=0.05)
                except queue.Empty:  # nothing finished: time for live text
                    if (job := self._take_partial()) is not None and not self._stopping.is_set():
                        self._recognize_partial(job)
                    continue
                if segment is _END:
                    break
                if isinstance(segment, _Discarded):
                    self._last_final = max(self._last_final, segment.utterance)
                    self._withdraw(segment.utterance, 0.0)
                elif not self._stopping.is_set():
                    self._recognize(segment)
        finally:
            self._to_translate.put(_END)

    def _withdraw(self, utterance: int, at: float) -> None:
        """Remove the live text shown for an utterance that will have no caption."""
        if (shown := self._caption_ids.pop(utterance, None)) is not None:
            self._emit(Caption(shown, at, at, False, self.source_lang, "", stage="dropped"))

    def _caption_id(self, utterance: int) -> int:
        if utterance not in self._caption_ids:
            self._caption_ids[utterance] = next(self._ids)
        return self._caption_ids[utterance]

    def _ruby(self, text: str, lang: str) -> list | None:
        if self._readings is None or lang != "ja":
            return None
        try:
            return self._readings(text)
        except Exception:
            log.exception("Could not look up readings")
            self._readings = None
            return None

    def _voices(self, segment: SpeechSegment, live: bool = False) -> tuple[np.ndarray, float]:
        """What to recognize: the voices alone if the line has music and a separator is on;
        and the seconds that took. Live text skips a slow separator, so finished lines don't
        wait behind it."""
        separator = self.separator
        if separator is None or segment.separated or (segment.noise_db is not None and segment.noise_db < self._music_db):
            return segment.audio, 0.0
        if live and getattr(separator, "rtf", 0.0) > LIVE_SEPARATION_RTF:
            return segment.audio, 0.0
        t0 = time.perf_counter()
        try:
            return separator.vocals(segment.audio), time.perf_counter() - t0
        except Exception as exc:  # e.g. the graphics chip ran out of memory: don't keep failing
            if self.separator is separator:  # (not turned off meanwhile)
                log.exception("Separating the voices failed; recognizing the original audio from now on")
                self.separator, self.separator_error = None, str(exc)
            return segment.audio, 0.0

    def _recognize_partial(self, segment: SpeechSegment) -> None:
        if segment.utterance <= self._last_final:
            return  # finished meanwhile
        audio, sep_s = self._voices(segment, live=True)
        with self._asr_lock:
            source_lang = self.source_lang
            try:
                result = self._asr.transcribe(audio)
            except Exception:
                log.debug("Live recognition failed", exc_info=True)
                return
        self._partial_cost = result.compute_s + sep_s
        text = self._postprocess(result.text, source_lang)
        if not text or segment.utterance <= self._last_final:
            return
        target = self.target_lang
        translating = self._translator is not None and bool(target) and target != source_lang
        self._emit(
            Caption(
                id=self._caption_id(segment.utterance), start=segment.start, end=segment.end, forced=False,
                source_lang=source_lang, text=text, target_lang=target if translating else None, stage="partial",
                ruby=self._ruby(text, source_lang), asr_s=result.compute_s,
            )  # fmt: skip
        )

    def _recognize(self, segment: SpeechSegment) -> None:
        waited = time.monotonic() - segment.created_at
        audio, sep_s = self._voices(segment)
        with self._asr_lock:
            source_lang = self.source_lang
            try:
                result = self._asr.transcribe(audio)
            except Exception:
                log.exception("Recognition failed for a %.1fs utterance", segment.duration)
                result = None
        self._last_final = max(self._last_final, segment.utterance)
        shown = self._caption_ids.pop(segment.utterance, None)  # its live text, if any was shown
        for skipped in [u for u in self._caption_ids if u < self._last_final]:  # dropped while recognition was behind
            self._withdraw(skipped, segment.start)
        text = self._postprocess(result.text, source_lang) if result else None
        if not text:
            if shown is not None:  # the live text turned out to be noise
                self._emit(Caption(shown, segment.start, segment.end, segment.forced, source_lang, "", stage="dropped"))
            return
        caption = Caption(
            id=shown or next(self._ids), start=segment.start, end=segment.end, forced=segment.forced,
            source_lang=source_lang, text=text, ruby=self._ruby(text, source_lang), asr_s=result.compute_s,
            sep_s=sep_s, waited_s=waited, speech_ended_at=segment.created_at,
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
                self.failed_translations += 1
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
        if caption.stage != "partial":
            self._caption_log.append(caption)
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
