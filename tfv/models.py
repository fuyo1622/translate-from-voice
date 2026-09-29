"""Where models are stored and how they are fetched.

A minimal stand-in for the Phase 3 model manager: each model is downloaded on
first use into the models directory and reused afterwards without touching the
network.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

SILERO_VAD_URL = "https://github.com/snakers4/silero-vad/raw/v6.2/src/silero_vad/data/silero_vad.onnx"
SILERO_VAD_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"

SENSEVOICE_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
SENSEVOICE_FILES = ("model.int8.onnx", "tokens.txt")

# Files faster-whisper needs from a CTranslate2 model repository.
_WHISPER_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]


def models_dir() -> Path:
    """Model cache: $TFV_MODELS_DIR, else the per-user app data directory."""
    if env := os.environ.get("TFV_MODELS_DIR"):
        base = Path(env)
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "tfv" / "models"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "tfv" / "models"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "tfv" / "models"
    base.mkdir(parents=True, exist_ok=True)
    return base


def silero_vad() -> Path:
    path = models_dir() / "silero_vad_v6.2.onnx"
    if not path.exists():
        _download(SILERO_VAD_URL, path, SILERO_VAD_SHA256)
    return path


def sensevoice() -> Path:
    """Directory holding the int8 SenseVoice-Small model for sherpa-onnx."""
    target = models_dir() / "sensevoice-small-2024-07-17"
    missing = [f for f in SENSEVOICE_FILES if not (target / f).exists()]
    if missing:
        from huggingface_hub import hf_hub_download

        log.info("Downloading SenseVoice-Small (~240 MB) to %s", target)
        for name in missing:
            hf_hub_download(SENSEVOICE_REPO, name, local_dir=target)
    return target


def faster_whisper(size: str) -> Path:
    """Directory holding a CTranslate2 Whisper model, e.g. "large-v3-turbo" or "small"."""
    from faster_whisper.utils import _MODELS

    if size not in _MODELS:
        raise ValueError(f"unknown Whisper model {size!r}; choose from {', '.join(_MODELS)} or pass a model directory")
    target = models_dir() / f"faster-whisper-{size}"
    if not (target / "model.bin").exists():
        from huggingface_hub import snapshot_download

        log.info("Downloading Whisper %s to %s", size, target)
        snapshot_download(_MODELS[size], local_dir=target, allow_patterns=_WHISPER_FILES)
    return target


def _download(url: str, dest: Path, sha256: str | None = None) -> None:
    log.info("Downloading %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        while chunk := resp.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if sha256 and digest.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {url}")
    tmp.replace(dest)
