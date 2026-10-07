"""Model catalog and downloads (the model manager).

Every model the app can use is listed in CATALOG with its size and license.
Models are downloaded on first use into models_dir(), checksum-verified
where the source publishes a checksum, and reused offline afterwards.
Interrupted downloads resume.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

Progress = Callable[[str, int, int], None]
"""progress(label, bytes_done, bytes_total) — called while downloading."""

HF = "https://huggingface.co"
_MARKER = ".tfv-complete"
# a trailing "?" marks a file that only some repositories have
_WHISPER_FILES = ("config.json", "preprocessor_config.json?", "model.bin", "tokenizer.json", "vocabulary.*")


@dataclass(frozen=True)
class ModelInfo:
    id: str
    task: str
    """"vad", "asr" or "mt"."""
    engine: str
    """Which backend loads it: silero, faster-whisper, sensevoice, reazonspeech, nllb, llama."""
    size_mb: int
    license: str
    repo: str = ""
    """Hugging Face repository; `files` are fetched from it (glob patterns allowed)."""
    files: tuple[str, ...] = ()
    url: str = ""
    """Direct download for single-file models not on Hugging Face."""
    sha256: str = ""
    prompt: str = ""
    """For llama models: the prompt format ("chat", "translategemma", "hunyuan")."""
    summary: str = ""
    languages: tuple[str, ...] = ()
    """Spoken languages a recognizer handles; empty = all supported ones."""

    @property
    def path(self) -> Path:
        return models_dir() / self.id

    @property
    def main_file(self) -> Path:
        """The file a backend opens (single-file models), else the model directory."""
        if self.url or self.engine == "llama":
            return self.path / self.files[0]
        return self.path


def _whisper(size: str, repo: str, size_mb: int, summary: str) -> ModelInfo:
    return ModelInfo(f"whisper-{size}", "asr", "faster-whisper", size_mb, "MIT", repo, _WHISPER_FILES, summary=summary)


def _gguf(id: str, repo: str, file: str, size_mb: int, license: str, prompt: str, summary: str) -> ModelInfo:
    return ModelInfo(id, "mt", "llama", size_mb, license, repo, (file,), prompt=prompt, summary=summary)


CATALOG: dict[str, ModelInfo] = {
    m.id: m
    for m in [
        ModelInfo(
            "silero-vad", "vad", "silero", 2, "MIT",
            url="https://github.com/snakers4/silero-vad/raw/v6.2/src/silero_vad/data/silero_vad.onnx",
            files=("silero_vad.onnx",),
            sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
            summary="Silero VAD v6.2: finds speech in the audio",
        ),
        ModelInfo(
            "gtcrn", "vad", "gtcrn", 1, "MIT",
            url="https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx",
            files=("gtcrn_simple.onnx",),
            sha256="e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534",
            summary="GTCRN speech denoiser: lets the VAD find speech under music and noise",
        ),
        _whisper("large-v3-turbo", "mobiuslabsgmbh/faster-whisper-large-v3-turbo", 1620, "Whisper turbo: best speed/quality on NVIDIA GPUs"),
        _whisper("large-v3", "Systran/faster-whisper-large-v3", 3090, "Whisper large-v3: best quality, needs a strong GPU"),
        _whisper("medium", "Systran/faster-whisper-medium", 1530, "Whisper medium"),
        _whisper("small", "Systran/faster-whisper-small", 485, "Whisper small: usable on CPU"),
        ModelInfo(
            "sensevoice-small", "asr", "sensevoice", 240, "FunASR model license",
            repo="csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
            files=("model.int8.onnx", "tokens.txt"),
            summary="SenseVoice-Small int8: fast on CPU, strong on Chinese",
        ),
        ModelInfo(
            "kotoba-whisper-v2.0", "asr", "faster-whisper", 1445, "MIT",
            repo="kotoba-tech/kotoba-whisper-v2.0-faster", files=_WHISPER_FILES, languages=("ja",),
            summary="Kotoba-Whisper v2.0: Whisper distilled for Japanese",
        ),
        ModelInfo(
            "reazonspeech-k2-v2", "asr", "reazonspeech", 161, "Apache-2.0",
            repo="reazon-research/reazonspeech-k2-v2",
            files=("encoder-epoch-99-avg-1.int8.onnx", "decoder-epoch-99-avg-1.onnx", "joiner-epoch-99-avg-1.int8.onnx", "tokens.txt"),
            languages=("ja",),
            summary="ReazonSpeech k2 v2: Japanese only, trained on 35,000 h of TV speech, fast on CPU",
        ),
        ModelInfo(
            "nllb-600m", "mt", "nllb", 640, "CC-BY-NC-4.0 (non-commercial)",
            repo="JustFrederik/nllb-200-distilled-600M-ct2-int8",
            files=("model.bin", "config.json", "shared_vocabulary.txt", "tokenizer.json"),
            summary="NLLB-200 600M int8: small, literal translations, runs on any CPU",
        ),
        _gguf("qwen3.5-0.8b", "unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf", 533, "Apache-2.0", "chat", "Qwen3.5 0.8B: tiny LLM"),
        _gguf("qwen3.5-2b", "unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", 1281, "Apache-2.0", "chat", "Qwen3.5 2B: small LLM, fits beside Whisper on 4 GB GPUs"),
        _gguf("qwen3.5-4b", "unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", 2741, "Apache-2.0", "chat", "Qwen3.5 4B"),
        _gguf("qwen3.5-9b", "unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf", 5681, "Apache-2.0", "chat", "Qwen3.5 9B: needs 8 GB+ VRAM"),
        _gguf("translategemma-4b", "mradermacher/translategemma-4b-it-GGUF", "translategemma-4b-it.Q4_K_M.gguf", 2490, "Gemma Terms of Use", "translategemma", "TranslateGemma 4B: Google's translation model"),
        _gguf("translategemma-12b", "mradermacher/translategemma-12b-it-GGUF", "translategemma-12b-it.Q4_K_M.gguf", 7301, "Gemma Terms of Use", "translategemma", "TranslateGemma 12B: needs 10 GB+ VRAM"),
        _gguf("hunyuan-mt-7b", "mradermacher/Hunyuan-MT-7B-GGUF", "Hunyuan-MT-7B.Q4_K_M.gguf", 4625, "Tencent Hunyuan Community License (not for EU/UK/KR)", "hunyuan", "Hunyuan-MT 7B: strong translation model, needs 6 GB+ VRAM"),
    ]
}


def models_dir() -> Path:
    """Model folder: $TFV_MODELS_DIR, else `models` inside the app folder (the project
    checkout, or next to tfv.exe in a packaged build), so everything stays in one place."""
    if env := os.environ.get("TFV_MODELS_DIR"):
        base = Path(env)
    elif getattr(sys, "frozen", False):
        base = Path(sys.executable).parent / "models"
    else:
        base = Path(__file__).resolve().parent.parent / "models"
    base.mkdir(parents=True, exist_ok=True)
    return base


def get(model_id: str) -> ModelInfo:
    try:
        return CATALOG[model_id]
    except KeyError:
        raise ValueError(f"unknown model {model_id!r}; see `python -m tfv models`") from None


def is_installed(model_id: str) -> bool:
    return (get(model_id).path / _MARKER).exists()


_locks: dict[str, threading.Lock] = {}


def ensure(model_id: str, progress: Progress | None = None) -> ModelInfo:
    """Download the model if needed; safe to call from several threads."""
    info = get(model_id)
    with _locks.setdefault(model_id, threading.Lock()):
        if not is_installed(model_id):
            log.info("Downloading %s (%s MB, license: %s)", info.id, f"{info.size_mb:,}", info.license)
            info.path.mkdir(parents=True, exist_ok=True)
            for name, url, size, sha256 in _plan(info):
                download(url, info.path / name, size=size, sha256=sha256, progress=progress, label=f"{info.id}: {name}")
            (info.path / _MARKER).write_text("ok")
    return info


def remove(model_id: str) -> bool:
    info = get(model_id)
    if not info.path.exists():
        return False
    shutil.rmtree(info.path)
    return True


def _plan(info: ModelInfo) -> list[tuple[str, str, int | None, str | None]]:
    """(file name, url, size, sha256) for every file of a model."""
    if info.url:
        return [(info.files[0], info.url, None, info.sha256 or None)]
    import httpx

    listing = httpx.get(f"{HF}/api/models/{info.repo}/tree/main", timeout=30, follow_redirects=True)
    listing.raise_for_status()
    remote = {f["path"]: f for f in listing.json() if f["type"] == "file"}
    plan = []
    for pattern in info.files:
        optional = pattern.endswith("?")
        matches = fnmatch.filter(remote, pattern.rstrip("?"))
        if not matches and not optional:
            raise RuntimeError(f"{info.repo} has no file matching {pattern!r}")
        for name in matches:
            f = remote[name]
            lfs = f.get("lfs") or {}
            plan.append((name, f"{HF}/{info.repo}/resolve/main/{name}", f.get("size"), lfs.get("oid")))
    return plan


_listener: Progress | None = None


def set_progress_listener(listener: Progress | None) -> None:
    """Receive progress for every download that has no explicit callback (e.g. to show it in the UI)."""
    global _listener
    _listener = listener


def download(
    url: str,
    dest: Path,
    size: int | None = None,
    sha256: str | None = None,
    progress: Progress | None = None,
    label: str = "",
) -> None:
    """Download `url` to `dest`, resuming a previous partial download."""
    import httpx

    progress = progress or _listener
    part = dest.with_name(dest.name + ".part")
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    digest = hashlib.sha256() if sha256 and not done else None
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=httpx.Timeout(30, read=120)) as r:
        if r.status_code == 416:  # already complete
            pass
        else:
            r.raise_for_status()
            if done and r.status_code == 200:  # server ignored the range; start over
                done, digest = 0, hashlib.sha256() if sha256 else None
            total = size or (done + int(r.headers.get("content-length") or 0))
            with open(part, "ab" if done else "wb") as out:
                for chunk in r.iter_bytes(1 << 20):
                    out.write(chunk)
                    if digest:
                        digest.update(chunk)
                    done += len(chunk)
                    if progress:
                        progress(label or dest.name, done, total)
    if sha256:
        actual = digest.hexdigest() if digest else _sha256_file(part)
        if actual != sha256.removeprefix("sha256:"):
            part.unlink(missing_ok=True)
            raise RuntimeError(f"checksum mismatch for {url}; the partial download was removed, try again")
    part.replace(dest)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def disk_usage(model_id: str) -> int:
    """Bytes the model occupies on disk (0 if absent)."""
    path = get(model_id).path
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0
