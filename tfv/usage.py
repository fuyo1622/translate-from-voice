"""Count what paid services were used this session and estimate the cost.

Prices are list prices in USD and change over time; they only drive the
estimate shown to the user. Claude prices are from Anthropic's model table
(2026-06); the others are approximate, so check the provider's pricing page.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

# (service, model) -> prices; tokens per million, audio per minute, characters per million
PRICES: dict[tuple[str, str], dict[str, float]] = {
    ("claude", "claude-opus-5"): {"input_mtok": 5.0, "output_mtok": 25.0},
    ("claude", "claude-opus-5-5"): {"input_mtok": 4.0, "output_mtok": 20.0},
    ("claude", "claude-sonnet-5"): {"input_mtok": 2.0, "output_mtok": 10.0},
    ("claude", "claude-haiku-4-5"): {"input_mtok": 1.0, "output_mtok": 5.0},
    ("claude", "claude-fable-5-1"): {"input_mtok": 10.0, "output_mtok": 50.0},
    ("openai", "gpt-6-luna"): {"input_mtok": 0.1, "output_mtok": 0.5},
    ("openai", "gpt-6-sol"): {"input_mtok": 2.0, "output_mtok": 10.0},
    ("openai", "gpt-4o-mini-transcribe"): {"audio_min": 0.003},
    ("openai", "gpt-4o-transcribe"): {"audio_min": 0.006},
    ("deepgram", "nova-3"): {"audio_min": 0.0043},
    ("deepl", ""): {"chars_mtok": 25.0},
}


@dataclass
class Meter:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    characters: int = 0
    audio_s: float = 0.0

    def cost(self, prices: dict[str, float] | None) -> float | None:
        if not prices:
            return None
        return (
            self.input_tokens / 1e6 * prices.get("input_mtok", 0)
            + self.output_tokens / 1e6 * prices.get("output_mtok", 0)
            + self.characters / 1e6 * prices.get("chars_mtok", 0)
            + self.audio_s / 60 * prices.get("audio_min", 0)
        )


_meters: dict[tuple[str, str], Meter] = {}
_lock = threading.Lock()


def record(service: str, model: str = "", *, input_tokens: int = 0, output_tokens: int = 0, characters: int = 0, audio_s: float = 0.0) -> None:
    with _lock:
        m = _meters.setdefault((service, model), Meter())
        m.requests += 1
        m.input_tokens += input_tokens
        m.output_tokens += output_tokens
        m.characters += characters
        m.audio_s += audio_s


def total_cost() -> float:
    with _lock:
        return sum(m.cost(PRICES.get(key)) or 0.0 for key, m in _meters.items())


def summary() -> list[str]:
    """One line per paid service used, e.g. "claude claude-opus-5: 12 requests, 3,120 tokens, ~$0.03"."""
    lines = []
    with _lock:
        for (service, model), m in _meters.items():
            parts = [f"{m.requests} requests"]
            if m.input_tokens or m.output_tokens:
                parts.append(f"{m.input_tokens + m.output_tokens:,} tokens")
            if m.characters:
                parts.append(f"{m.characters:,} characters")
            if m.audio_s:
                parts.append(f"{m.audio_s / 60:.1f} audio min")
            cost = m.cost(PRICES.get((service, model)))
            if cost is not None:
                parts.append(f"~${cost:.3f}")
            lines.append(f"{service} {model}".strip() + ": " + ", ".join(parts))
    return lines


def reset() -> None:
    with _lock:
        _meters.clear()
