"""User preferences for the desktop app, stored as JSON in the per-user config folder."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class Prefs:
    source_lang: str = "ja"
    target_lang: str | None = "zh"
    """None: captions only, no translation."""
    capture: str = "output"
    """"output": what the computer plays; "input": a microphone or virtual cable."""
    audio_device: str | None = None
    """None follows the system default (for output: speaker or earphones, whichever is active)."""
    remote_url: str = ""
    """If set, models run on this `tfv serve` engine instead of locally."""
    profile: str = "auto"
    asr: str = "auto"
    mt: str = "auto"
    mt_url: str = ""
    adapt: bool = True
    vad_denoise: bool = True
    """Find speech under background music and noise (the speech detector hears a denoised copy)."""
    # overlay
    show_original: bool = True
    show_together: bool = True
    """Hold the original until its translation is ready, so both lines appear at the same time
    (only when live text is off)."""
    live_text: bool = True
    """Show the words while they are still being spoken; the translation follows each phrase."""
    furigana: bool = True
    """Small readings above kanji in Japanese lines."""
    font_size: int = 22
    opacity: float = 0.55
    """Background panel opacity (the text stays opaque)."""
    lines: int = 2
    """How many recent captions to keep on screen."""
    clear_after_s: int = 8
    locked: bool = False
    """Locked: clicks pass through the overlay to the window below."""
    unlock_button: bool = True
    """While locked, a small button at the bar's corner unlocks it."""
    geometry: list[int] = field(default_factory=list)
    """[x, y, width] of the overlay; empty = bottom center of the primary screen."""
    tip_shown: bool = False
    """The first-start notice about where the menu is has been shown."""

    @classmethod
    def load(cls) -> Prefs:
        path = prefs_path()
        try:
            data = json.loads(path.read_text("utf-8"))
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError) as exc:
            log.warning("Ignoring unreadable settings %s: %s", path, exc)
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self) -> None:
        path = prefs_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(path)

    def engine_key(self) -> tuple:
        """Settings whose change requires reloading the models."""
        return (self.capture, self.audio_device, self.remote_url, self.profile, self.asr, self.mt, self.mt_url, self.adapt, self.vad_denoise)


def prefs_path() -> Path:
    if env := os.environ.get("TFV_CONFIG_DIR"):
        base = Path(env)
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "tfv"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "tfv"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "tfv"
    return base / "settings.json"
