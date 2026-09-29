from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

Context = Sequence[tuple[str, str]]
"""Recent (source, translation) pairs, oldest first."""


class Translator(ABC):
    """Translates one caption line between two supported languages."""

    name = "?"
    description = ""
    uses_context = False
    """Whether translate() makes use of the previous lines."""

    @abstractmethod
    def translate(self, text: str, source: str, target: str, context: Context = ()) -> str: ...

    def warmup(self, source: str, target: str) -> None:
        """Translate once so model loading and first-call setup happen before real captions."""
        self.translate({"zh": "你好。", "en": "Hello.", "ja": "こんにちは。"}[source], source, target)

    def close(self) -> None:
        """Release models and helper processes."""
