"""Cut a continuous audio stream into utterances using per-window speech probabilities."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from tfv import SAMPLE_RATE


@dataclass
class SpeechSegment:
    audio: np.ndarray
    """16 kHz mono float32, including a little padding around the speech."""
    start: float
    """Seconds since the stream started."""
    end: float
    forced: bool = False
    """True if cut because it reached the maximum length, not at a pause."""
    created_at: float = field(default_factory=time.monotonic)

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE


@dataclass
class SegmenterConfig:
    threshold: float = 0.5
    """Speech probability that starts an utterance."""
    min_speech_ms: int = 250
    """Utterances with less speech than this are dropped as noise."""
    min_silence_ms: int = 500
    """A pause this long ends an utterance."""
    pad_ms: int = 200
    """Audio kept before and after the speech."""
    max_segment_s: float = 12.0
    """Longer speech is split at its quietest point so captions keep flowing."""


class Segmenter:
    """Streaming utterance segmenter.

    `vad` maps a window of `window_size` samples to a speech probability and is
    called on consecutive windows. Speech continues while the probability stays
    above `threshold - 0.15` (hysteresis), and ends after `min_silence_ms` below it.
    """

    def __init__(
        self,
        vad: Callable[[np.ndarray], float],
        config: SegmenterConfig | None = None,
        window_size: int = 512,
    ):
        self.config = cfg = config or SegmenterConfig()
        self._vad = vad
        self._win = window_size

        def windows(ms: float) -> int:
            return max(1, round(ms / 1000 * SAMPLE_RATE / window_size))

        self._neg_threshold = max(cfg.threshold - 0.15, 0.01)
        self._min_speech = windows(cfg.min_speech_ms)
        self._min_silence = windows(cfg.min_silence_ms)
        self._pad = windows(cfg.pad_ms)
        self._max_len = max(windows(cfg.max_segment_s * 1000), self._min_silence + 2)

        self._pending = np.zeros(0, dtype=np.float32)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=self._pad)
        self._pos = 0  # windows consumed since the stream started
        self._reset_segment()

    @property
    def active(self) -> bool:
        """True while inside an utterance."""
        return self._in_speech

    def push(self, audio: np.ndarray) -> list[SpeechSegment]:
        """Feed audio of any length; returns the utterances it completed."""
        buf = np.concatenate([self._pending, audio]) if len(self._pending) else np.asarray(audio, np.float32)
        n = len(buf) // self._win
        done = []
        for i in range(n):
            window = buf[i * self._win : (i + 1) * self._win]
            if (seg := self._step(window, self._vad(window))) is not None:
                done.append(seg)
        self._pending = buf[n * self._win :].copy()
        return done

    def flush(self) -> SpeechSegment | None:
        """End of stream or the stream went idle: close the current utterance, if any."""
        seg = None
        if self._in_speech:
            seg = self._finish(drop_tail=self._silence_run - self._pad)
        # whatever follows an idle gap is not continuous with what came before
        self._pre_roll.clear()
        self._pending = np.zeros(0, dtype=np.float32)
        if hasattr(self._vad, "reset"):
            self._vad.reset()
        return seg

    def _reset_segment(self) -> None:
        self._in_speech = False
        self._windows: list[np.ndarray] = []
        self._probs: list[float] = []
        self._start = 0  # window index where the current utterance starts
        self._silence_run = 0  # trailing windows below the end threshold

    def _step(self, window: np.ndarray, prob: float) -> SpeechSegment | None:
        index = self._pos
        self._pos += 1
        if not self._in_speech:
            if prob < self.config.threshold:
                self._pre_roll.append(window)
                return None
            self._in_speech = True
            self._windows = [*self._pre_roll, window]
            self._probs = [0.0] * len(self._pre_roll) + [prob]
            self._start = index - len(self._pre_roll)
            self._pre_roll.clear()
            return None

        self._windows.append(window)
        self._probs.append(prob)
        if prob < self._neg_threshold:
            self._silence_run += 1
            if self._silence_run >= self._min_silence:
                return self._finish(drop_tail=self._silence_run - self._pad)
        else:
            self._silence_run = 0
        if len(self._windows) >= self._max_len:
            return self._split()
        return None

    def _finish(self, drop_tail: int) -> SpeechSegment | None:
        keep = len(self._windows) - max(drop_tail, 0)
        windows, probs, start = self._windows[:keep], self._probs[:keep], self._start
        dropped = self._windows[keep:]
        self._reset_segment()
        self._pre_roll.extend(dropped)  # silence after this utterance pads the next one
        if sum(p >= self.config.threshold for p in probs) < self._min_speech:
            return None
        return self._segment(windows, start, forced=False)

    def _split(self) -> SpeechSegment:
        # Cut at the least speech-like window in the last third (the latest one
        # on ties), so a word is less likely to be chopped in half. The rest
        # starts the next utterance.
        n = len(self._windows)
        lo = n - max(n // 3, 1)
        tail = np.asarray(self._probs[lo:])
        cut = n - 1 - int(np.argmin(tail[::-1]))
        seg = self._segment(self._windows[:cut], self._start, forced=True)
        self._windows, self._probs = self._windows[cut:], self._probs[cut:]
        self._start += cut
        self._silence_run = 0
        for p in reversed(self._probs):
            if p >= self._neg_threshold:
                break
            self._silence_run += 1
        return seg

    def _segment(self, windows: list[np.ndarray], start_window: int, forced: bool) -> SpeechSegment:
        audio = np.concatenate(windows)
        start = start_window * self._win / SAMPLE_RATE
        return SpeechSegment(audio=audio, start=start, end=start + len(audio) / SAMPLE_RATE, forced=forced)
