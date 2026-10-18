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
    utterance: int = 0
    """Counts up per utterance; live snapshots of an utterance share its number."""
    noise_db: float | None = None
    """How much of the audio's energy the VAD's denoiser took out, in dB: high under music or
    noise, near 0 for clean speech. None when the VAD judged the audio itself."""
    separated: bool = False
    """The audio is already the voices without the music (song mode)."""

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
    max_segment_s: float = 8.0
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
        self._pending_vad = np.zeros(0, dtype=np.float32)
        self._pre_roll: deque[tuple[np.ndarray, tuple[float, float] | None]] = deque(maxlen=self._pad)
        self._pos = 0  # windows consumed since the stream started
        self._utterance = 0
        self.discarded: list[int] = []
        """Utterances dropped as too short to be speech (the caller clears the list)."""
        self._reset_segment()

    def set_threshold(self, threshold: float) -> None:
        """Change how sure the VAD must be (e.g. while it hears separated singing)."""
        self.config.threshold = threshold
        self._neg_threshold = max(threshold - 0.15, 0.01)

    @property
    def active(self) -> bool:
        """True while inside an utterance."""
        return self._in_speech

    def snapshot(self) -> SpeechSegment | None:
        """The utterance in progress so far, for live text; None between utterances."""
        if not self._in_speech:
            return None
        return self._segment(self._windows, self._energy, self._start, forced=False)

    def push(self, audio: np.ndarray, vad_audio: np.ndarray | None = None) -> list[SpeechSegment]:
        """Feed audio of any length; returns the utterances it completed.

        `vad_audio`, the same length, is what the VAD judges instead (e.g. denoised);
        the utterances are still cut from `audio`.
        """
        buf = np.concatenate([self._pending, audio]) if len(self._pending) else np.asarray(audio, np.float32)
        if vad_audio is None:
            vad_buf = buf
        else:
            vad_buf = np.concatenate([self._pending_vad, vad_audio]) if len(self._pending_vad) else np.asarray(vad_audio, np.float32)
        n = len(buf) // self._win
        done = []
        for i in range(n):
            window = buf[i * self._win : (i + 1) * self._win]
            vad_window = vad_buf[i * self._win : (i + 1) * self._win]
            energy = None if vad_audio is None else (float(window @ window), float(vad_window @ vad_window))
            if (seg := self._step(window, self._vad(vad_window), energy)) is not None:
                done.append(seg)
        self._pending = buf[n * self._win :].copy()
        self._pending_vad = vad_buf[n * self._win :].copy()
        return done

    def flush(self) -> SpeechSegment | None:
        """End of stream or the stream went idle: close the current utterance, if any."""
        seg = None
        if self._in_speech:
            seg = self._finish(drop_tail=self._silence_run - self._pad)
        # whatever follows an idle gap is not continuous with what came before
        self._pre_roll.clear()
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_vad = np.zeros(0, dtype=np.float32)
        if hasattr(self._vad, "reset"):
            self._vad.reset()
        return seg

    def _reset_segment(self) -> None:
        self._in_speech = False
        self._windows: list[np.ndarray] = []
        self._energy: list[tuple[float, float] | None] = []
        """Per window: (energy of the audio, energy of what the VAD judged), if they differ."""
        self._probs: list[float] = []
        self._start = 0  # window index where the current utterance starts
        self._silence_run = 0  # trailing windows below the end threshold

    def _step(self, window: np.ndarray, prob: float, energy: tuple[float, float] | None = None) -> SpeechSegment | None:
        index = self._pos
        self._pos += 1
        if not self._in_speech:
            if prob < self.config.threshold:
                self._pre_roll.append((window, energy))
                return None
            self._in_speech = True
            self._utterance += 1
            self._windows = [w for w, _ in self._pre_roll] + [window]
            self._energy = [e for _, e in self._pre_roll] + [energy]
            self._probs = [0.0] * len(self._pre_roll) + [prob]
            self._start = index - len(self._pre_roll)
            self._pre_roll.clear()
            return None

        self._windows.append(window)
        self._energy.append(energy)
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
        windows, energy, probs, start = self._windows[:keep], self._energy[:keep], self._probs[:keep], self._start
        dropped = list(zip(self._windows[keep:], self._energy[keep:]))
        self._reset_segment()
        self._pre_roll.extend(dropped)  # silence after this utterance pads the next one
        # Silero's memory of the last speaker can keep a new voice after a short pause
        # below the threshold for seconds (measured on TV speech): start each utterance fresh.
        if hasattr(self._vad, "reset"):
            self._vad.reset()
        if sum(p >= self.config.threshold for p in probs) < self._min_speech:
            self.discarded.append(self._utterance)  # live text already shown for it must go
            return None
        return self._segment(windows, energy, start, forced=False)

    def _split(self) -> SpeechSegment:
        # Cut at the least speech-like window in the last third (the latest one
        # on ties), so a word is less likely to be chopped in half. The rest
        # starts the next utterance.
        n = len(self._windows)
        lo = n - max(n // 3, 1)
        tail = np.asarray(self._probs[lo:])
        cut = n - 1 - int(np.argmin(tail[::-1]))
        seg = self._segment(self._windows[:cut], self._energy[:cut], self._start, forced=True)
        self._utterance += 1  # the rest continues as the next caption
        self._windows, self._energy, self._probs = self._windows[cut:], self._energy[cut:], self._probs[cut:]
        self._start += cut
        self._silence_run = 0
        for p in reversed(self._probs):
            if p >= self._neg_threshold:
                break
            self._silence_run += 1
        return seg

    def _segment(self, windows: list[np.ndarray], energy: list, start_window: int, forced: bool) -> SpeechSegment:
        audio = np.concatenate(windows)
        start = start_window * self._win / SAMPLE_RATE
        end = start + len(audio) / SAMPLE_RATE
        noise_db = None
        if energy and all(e is not None for e in energy):
            heard, kept = sum(e[0] for e in energy), sum(e[1] for e in energy)
            noise_db = 10 * float(np.log10((heard + 1e-9) / (kept + 1e-9)))
        return SpeechSegment(audio=audio, start=start, end=end, forced=forced, utterance=self._utterance, noise_db=noise_db)
