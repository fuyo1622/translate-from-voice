"""Measure recognition and translation quality, speed and memory on this machine.

Data: a small subset of FLEURS (Google, CC-BY-4.0), whose sentences are the
same across languages, so every clip has a reference transcript and
reference translations. Only the first N clips per language are downloaded
(the archives are streamed, not saved).

Metrics: CER (zh, ja) / WER (en) for recognition, chrF for translation,
seconds per utterance, and memory. The "noisy" condition mixes pink noise
at 10 dB SNR into the clean clips as a rough stand-in for background sound.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import tarfile
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from tfv import hardware, languages, models

log = logging.getLogger(__name__)

FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
_FLEURS_LANG = {"zh": "cmn_hans_cn", "en": "en_us", "ja": "ja_jp"}


def eval_dir() -> Path:
    path = models.models_dir().parent / "eval" / "fleurs"
    path.mkdir(parents=True, exist_ok=True)
    return path


# -- data ------------------------------------------------------------------------


def _read_tsv(text: str) -> list[dict]:
    rows = []
    for fields in csv.reader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(fields) >= 3:
            rows.append({"id": fields[0], "file": fields[1], "text": fields[2]})
    return rows


class _StreamReader(io.RawIOBase):
    """File-like view of an HTTP body, so tarfile can read an archive while it downloads."""

    def __init__(self, chunks):
        self._chunks = chunks
        self._buffer = b""

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self._buffer:
            try:
                self._buffer = next(self._chunks)
            except StopIteration:
                return 0
        n = min(len(b), len(self._buffer))
        b[:n] = self._buffer[:n]
        self._buffer = self._buffer[n:]
        return n


def download(n: int = 20, status: Callable[[str], None] = print) -> Path:
    """Fetch N clips per language plus reference texts for every language; returns the manifest path."""
    import httpx

    target = eval_dir()
    tsv = {}
    for lang, code in _FLEURS_LANG.items():
        r = httpx.get(f"{FLEURS}/{code}/test.tsv", follow_redirects=True, timeout=60)
        r.raise_for_status()
        tsv[lang] = _read_tsv(r.text)
    text_by_id = {lang: {row["id"]: row["text"] for row in rows} for lang, rows in tsv.items()}
    common = set.intersection(*(set(t) for t in text_by_id.values()))
    samples = []
    for lang, code in _FLEURS_LANG.items():
        wanted = {row["file"]: row["id"] for row in tsv[lang] if row["id"] in common}
        taken: set[str] = set()
        status(f"Downloading {n} {lang} clips from FLEURS…")
        with httpx.stream("GET", f"{FLEURS}/{code}/audio/test.tar.gz", follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with tarfile.open(fileobj=io.BufferedReader(_StreamReader(r.iter_bytes(1 << 16))), mode="r|gz") as tar:
                for member in tar:
                    name = Path(member.name).name
                    sid = wanted.get(name)
                    if not member.isfile() or sid is None or sid in taken:
                        continue
                    data = tar.extractfile(member).read()
                    (target / lang).mkdir(exist_ok=True)
                    (target / lang / name).write_bytes(data)
                    taken.add(sid)
                    samples.append({"id": sid, "lang": lang, "file": f"{lang}/{name}", "refs": {l: text_by_id[l][sid] for l in text_by_id}})
                    if len(taken) >= n:
                        break
    manifest = target / "manifest.json"
    manifest.write_text(json.dumps(samples, ensure_ascii=False, indent=1), "utf-8")
    status(f"Saved {len(samples)} clips to {target}")
    return manifest


def load_manifest() -> list[dict]:
    path = eval_dir() / "manifest.json"
    if not path.exists():
        raise RuntimeError("no evaluation data yet; run `python -m tfv eval download` first")
    return json.loads(path.read_text("utf-8"))


# -- metrics -----------------------------------------------------------------------

_tw2sp = None


def _to_simplified(text: str) -> str:
    """Taiwan Traditional → mainland Simplified, including vocabulary (軟體 → 软件), so our
    Taiwan-style output is compared fairly with FLEURS's mainland references."""
    global _tw2sp
    if _tw2sp is None:
        import opencc

        _tw2sp = opencc.OpenCC("tw2sp")
    return _tw2sp.convert(text)


def tokens_for_scoring(text: str, lang: str) -> list[str]:
    """Characters (zh, ja) or words (en), without punctuation or case; Chinese compared in Simplified."""
    text = unicodedata.normalize("NFKC", text).lower()
    if lang == "zh":
        text = _to_simplified(text)
    text = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in text)
    return [ch for ch in text if not ch.isspace()] if lang in ("zh", "ja") else text.split()


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def error_rate(hyps: list[str], refs: list[str], lang: str) -> float:
    """CER for zh/ja, WER for en, over the whole set (total edits / total reference length)."""
    edits = total = 0
    for hyp, ref in zip(hyps, refs):
        r = tokens_for_scoring(ref, lang)
        edits += edit_distance(tokens_for_scoring(hyp, lang), r)
        total += len(r)
    return edits / max(total, 1)


def chrf(hyps: list[str], refs: list[str], order: int = 6, beta: float = 2.0) -> float:
    """Corpus chrF (character n-grams 1..6, recall weighted 2x), 0-100. Whitespace is ignored."""
    match = [0] * order
    hyp_total = [0] * order
    ref_total = [0] * order
    for hyp, ref in zip(hyps, refs):
        h, r = "".join(hyp.split()), "".join(ref.split())
        for n in range(1, order + 1):
            hc = Counter(h[i : i + n] for i in range(len(h) - n + 1))
            rc = Counter(r[i : i + n] for i in range(len(r) - n + 1))
            match[n - 1] += sum((hc & rc).values())
            hyp_total[n - 1] += sum(hc.values())
            ref_total[n - 1] += sum(rc.values())
    precisions = [m / h for m, h in zip(match, hyp_total) if h]
    recalls = [m / r for m, r in zip(match, ref_total) if r]
    if not precisions or not recalls:
        return 0.0
    p, r = sum(precisions) / len(precisions), sum(recalls) / len(recalls)
    return 100 * (1 + beta**2) * p * r / (beta**2 * p + r) if p + r else 0.0


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.arange(len(spectrum), dtype=np.float64)
    freqs[0] = 1.0
    spectrum /= np.sqrt(freqs)  # 1/f power
    spectrum[0] = 0.0  # no DC offset
    noise = np.fft.irfft(spectrum, n)
    return (noise / (np.sqrt(np.mean(noise**2)) + 1e-12)).astype(np.float32)


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio**2))) or 1e-4
    return audio + pink_noise(len(audio), rng) * rms / 10 ** (snr_db / 20)


# -- runs ----------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    scores: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0
    """Mean compute time per utterance."""
    rtf: float = 0.0
    vram_mb: int | None = None
    notes: str = ""


def _vram_used() -> int | None:
    rows = hardware._nvidia_smi()
    return rows[0][1] - rows[0][2] if rows else None


def run_asr(choice_text: str, samples: list[dict], snr_db: float | None, status: Callable[[str], None] = print) -> Result:
    from faster_whisper.audio import decode_audio

    from tfv.engine.asr import create_asr
    from tfv.engine.textproc import TextPostProcessor
    from tfv.resources import Choice

    choice = Choice.parse(choice_text)
    result = Result(choice_text)
    post = TextPostProcessor()
    rng = np.random.default_rng(0)
    base = _vram_used()
    total_s = total_audio = 0.0
    asr = None
    for lang in languages.SUPPORTED:
        clips = [s for s in samples if s["lang"] == lang]
        if not clips:
            continue
        if asr is None:
            asr = create_asr(choice, language=lang)
            if base is not None and (used := _vram_used()) is not None:
                result.vram_mb = used - base
        else:
            asr.set_language(lang)
        hyps, refs = [], []
        for sample in clips:
            audio = decode_audio(str(eval_dir() / sample["file"]), sampling_rate=16000)
            if snr_db is not None:
                audio = add_noise(audio, snr_db, rng)
            out = asr.transcribe(audio)
            total_s += out.compute_s
            total_audio += len(audio) / 16000
            hyps.append(post(out.text, lang) or "")
            refs.append(sample["refs"][lang])
        result.scores[lang] = error_rate(hyps, refs, lang)
        status(f"  {choice_text} {lang}: {'CER' if lang != 'en' else 'WER'} {100 * result.scores[lang]:.1f}%")
    if asr is not None:
        asr.close()
    n = len(samples)
    result.seconds = total_s / max(n, 1)
    result.rtf = total_s / max(total_audio, 1e-9)
    return result


def run_mt(choice_text: str, samples: list[dict], status: Callable[[str], None] = print, limit: int = 30) -> Result:
    """Translate `limit` reference transcripts (text only) in all six directions."""
    from tfv.engine.mt import create_translator
    from tfv.engine.textproc import TextPostProcessor
    from tfv.resources import Choice

    choice = Choice.parse(choice_text)
    result = Result(choice_text)
    post = TextPostProcessor()
    base = _vram_used()
    by_id = {s["id"]: s["refs"] for s in samples}  # every clip id has text in all languages
    sentences = {sid: by_id[sid] for sid in sorted(by_id)[:limit]}
    translator = create_translator(choice, source="en", target="ja")
    if base is not None and (used := _vram_used()) is not None:
        result.vram_mb = used - base
    times = []
    try:
        for source in languages.SUPPORTED:
            for target in languages.SUPPORTED:
                if source == target:
                    continue
                hyps, refs = [], []
                for refs_by_lang in sentences.values():
                    t0 = time.perf_counter()
                    try:
                        out = translator.translate(refs_by_lang[source], source, target)
                    except Exception as exc:
                        log.warning("%s failed: %s", choice_text, exc)
                        out = ""
                    times.append(time.perf_counter() - t0)
                    hyps.append(post(out, target) or "")
                    refs.append(_to_simplified(refs_by_lang[target]) if target == "zh" else refs_by_lang[target])
                if target == "zh":  # FLEURS Chinese references are Simplified
                    hyps = [_to_simplified(h) for h in hyps]
                result.scores[f"{source}→{target}"] = chrf(hyps, refs)
                status(f"  {choice_text} {source}→{target}: chrF {result.scores[f'{source}→{target}']:.1f}")
    finally:
        translator.close()
    result.seconds = sum(times) / max(len(times), 1)
    return result


def report(hw: hardware.Hardware, n_clips: int, asr_clean: list[Result], asr_noisy: list[Result], mt: list[Result]) -> str:
    lines = [
        "# Evaluation results",
        "",
        f"Machine: {hw.summary()}  ",
        f"Data: FLEURS test set (CC-BY-4.0), {n_clips} clips per language; noisy = pink noise at 10 dB SNR.  ",
        "Recognition: CER for Chinese/Japanese, WER for English (lower is better). "
        "Translation: chrF against the FLEURS reference in the target language (higher is better), "
        "translating the reference transcripts so recognition errors don't count.",
        "",
        "## Speech recognition",
        "",
        "| Model | zh CER | zh noisy | en WER | en noisy | ja CER | ja noisy | s / utterance | × real time | VRAM MB |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    noisy = {r.name: r for r in asr_noisy}
    for r in asr_clean:
        nz = noisy.get(r.name, Result(r.name))
        cells = []
        for lang in languages.SUPPORTED:
            cells += [_pct(r.scores.get(lang)), _pct(nz.scores.get(lang))]
        lines.append(f"| {r.name} | " + " | ".join(cells) + f" | {r.seconds:.2f} | {r.rtf:.2f} | {r.vram_mb if r.vram_mb is not None else '–'} |")
    directions = [f"{s}→{t}" for s in languages.SUPPORTED for t in languages.SUPPORTED if s != t]
    lines += ["", "## Translation", "", "| Model | " + " | ".join(directions) + " | mean | s / line | VRAM MB |", "|---" * (len(directions) + 4) + "|"]
    for r in mt:
        mean = sum(r.scores.values()) / max(len(r.scores), 1)
        cells = [f"{r.scores.get(d, 0):.1f}" for d in directions]
        lines.append(f"| {r.name} | " + " | ".join(cells) + f" | **{mean:.1f}** | {r.seconds:.2f} | {r.vram_mb if r.vram_mb is not None else '–'} |")
    return "\n".join(lines) + "\n"


def _pct(value: float | None) -> str:
    return f"{100 * value:.1f}%" if value is not None else "–"
