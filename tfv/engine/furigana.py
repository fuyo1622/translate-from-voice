"""Readings for the kanji in Japanese text (furigana), from Sudachi's dictionary.

`segments("持ち帰りの弁当")` gives [("持", "も"), ("ち", None), ("帰", "かえ"), ("りの", None),
("弁当", "べんとう")]: the reading goes over the kanji only, never over the kana around them.
"""

from __future__ import annotations

import logging
import re
import threading

log = logging.getLogger(__name__)

Segment = tuple[str, "str | None"]
"""Text and the reading shown above it (None: no reading)."""

_KANJI_CHARS = "㐀-䶿一-鿿豈-﫿々〆ヶ"
_KANJI = re.compile(f"[{_KANJI_CHARS}]")
_RUNS = re.compile(f"[{_KANJI_CHARS}]+|[^{_KANJI_CHARS}]+")


def to_hiragana(text: str) -> str:
    return "".join(chr(ord(ch) - 0x60) if "ァ" <= ch <= "ヶ" else ch for ch in text)


def split_okurigana(surface: str, reading: str) -> list[Segment]:
    """Spread a word's reading over its kanji runs: ("食べる", "たべる") -> [("食", "た"), ("べる", None)]."""
    runs = _RUNS.findall(surface)
    pattern = "".join("(.+?)" if _KANJI.match(run) else f"({re.escape(to_hiragana(run))})" for run in runs)
    match = re.fullmatch(pattern, reading)
    if match is None:  # an irregular reading: put it over the whole word
        return [(surface, reading)]
    return [(run, part if _KANJI.match(run) else None) for run, part in zip(runs, match.groups())]


def _merge(segments: list[Segment]) -> list[Segment]:
    merged: list[Segment] = []
    for text, reading in segments:
        if reading is None and merged and merged[-1][1] is None:
            merged[-1] = (merged[-1][0] + text, None)
        else:
            merged.append((text, reading))
    return merged


class Furigana:
    """Loads the dictionary once (about 0.7 s); then a line takes a few milliseconds."""

    def __init__(self):
        self._tokenizer = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._tokenizer is None:
                from sudachipy import Dictionary, SplitMode

                self._tokenizer = Dictionary(dict="core").tokenizer(SplitMode.C)

    def segments(self, text: str) -> list[Segment]:
        if not _KANJI.search(text):
            return [(text, None)]
        self.load()
        out: list[Segment] = []
        with self._lock:  # Sudachi tokenizers are not thread-safe
            morphemes = [(m.surface(), m.reading_form()) for m in self._tokenizer.tokenize(text)]
        for surface, reading in morphemes:
            reading = to_hiragana(reading)
            if not _KANJI.search(surface) or not reading or _KANJI.search(reading):
                out.append((surface, None))  # kana, Latin, numbers, or a word the dictionary can't read
            else:
                out += split_okurigana(surface, reading)
        return _merge(out)
