"""Taking the music out from under voices (UVR MDX-Net "Voc_FT"), so that sung lyrics and talk over
loud music are recognized.

With music as loud as the voice, ReazonSpeech's error rate on sung Japanese lyrics fell from 30.5% to
21.6% (19.6% for the singing alone), and on TV speech from 23.7% to 20.2%; clean audio was unaffected
(EVAL.md). The pipeline uses it only on lines in which the VAD's denoiser found music.

The model needs a graphics chip: through DirectML (any GPU on Windows) one run over ~5.8 s of audio
takes ~0.4 s on an RTX 3050 and ~1.5 s on Intel Iris Xe graphics, but 6-10 s on a 4-core CPU. It runs
in half precision (a copy made once on first use): same output, half the graphics memory, which is
what lets it share a 4 GB card with the translator (in full precision translation slowed 25-fold).

DirectML comes in the onnxruntime-directml package, which can't be installed beside the onnxruntime
package the rest of tfv uses (both are the `onnxruntime` module). So the model runs in a helper
process that imports a private copy of it, downloaded on first use like llama.cpp.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import struct
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import numpy as np

from tfv import SAMPLE_RATE, hardware, models

log = logging.getLogger(__name__)

MODEL_ID = "uvr-mdx-voc-ft"
MUSIC_DB = 4.0
"""Energy (dB) the VAD's denoiser removed from a line, above which the line counts as having music.
Measured: at least 5 dB for every line under music as loud as the voice; about 0-2 dB for most clean
speech (more for TV speech that has some background sound already)."""
MAX_RTF = 0.6
"""A device that separates slower than this (seconds per second of audio) delays captions too much.
A line takes one model run per ~5.8 s started, so a line with music waits up to ~3.5 s more per run."""
NEED_MB = 1100
"""Free graphics memory an NVIDIA card needs for it (measured: ~0.95 GB beside TranslateGemma 4B)."""

ORT_VERSION = "1.23.0"
_PYPI = "https://files.pythonhosted.org/packages/"
# Python tag -> (wheel, sha256), from PyPI: onnxruntime-directml (MIT) runs the model
_WHEELS = {
    "cp310": ("5b/f8/c9282f935b978764bdf13869cccc174267c936efd150ca070e56e23f5d05/onnxruntime_directml-1.23.0-cp310-cp310-win_amd64.whl",
              "184fb5255b544edc2a14e95051f91864ab4dd868425472b54c3f26d2c3b1f456"),
    "cp311": ("1d/47/a117bd93dccff9b35d69a3fdbc2a4d9ae4c6d80148914f8aad6a8a4a852b/onnxruntime_directml-1.23.0-cp311-cp311-win_amd64.whl",
              "87d290a769deededc51ee757770f35a82e52f8765f00880b3a4a2f573a4c6267"),
    "cp312": ("97/ce/516b4317155033a8f810a567a8556c2e2e40cd8a0521fd47f06168a9577f/onnxruntime_directml-1.23.0-cp312-cp312-win_amd64.whl",
              "e621e3c0a251f83da338652508d9d68bfb8a1ad6123fc50fed4da9a05b4950bf"),
    "cp313": ("ba/c1/eb24dfaea7686845aa70506c2335955bcdcb9d673eb0358c38f6e3d0cfec/onnxruntime_directml-1.23.0-cp313-cp313-win_amd64.whl",
              "92bc0681f6a01554645f572b5d384dc9d8ca9811c84f1884c27ca6937802a259"),
}  # fmt: skip
# ... and onnx (Apache-2.0) with what it needs (ml_dtypes: Apache-2.0, protobuf: BSD-3-Clause,
# typing_extensions: PSF-2.0, packaging: Apache-2.0/BSD) let onnxruntime's own converter make the
# half-precision copy of the model
_ONNX_ABI3 = ("38/ce/102fd4a0b2a6d111a9c86745e084c4c68c0ee020eaa359a03a8d43e4646f/onnx-1.23.2-cp312-abi3-win_amd64.whl",
              "9b382ba898a7c142a0801d03cf04ecabced96c1543c7b643a86f0928143802de")
_ONNX = {
    "cp310": ("8b/3e/22cb5797df2aef3d6243ed2c40a3807e7ee3d313b9e22386fc1638b794e5/onnx-1.23.2-cp310-cp310-win_amd64.whl",
              "0100e6c3f30db8ff10876d8cfd0cb27296166d5a612ab37c3998e07e83b3fde8"),
    "cp311": ("81/0d/3aaf8f1fea3430282bd65acb3808d80fbdfeb90f20cfecb4072604e37ca6/onnx-1.23.2-cp311-cp311-win_amd64.whl",
              "a40265d62b7a614041593e11370d316880f9628eb5a0d49d9028c9c0e7f1cc08"),
    "cp312": _ONNX_ABI3,
    "cp313": _ONNX_ABI3,
}
_ML_DTYPES = {
    "cp310": ("f8/63/efc9257a1ef0f53dfc76dedfe70d7d35118fbcdb810bb48cb7323ebd0b87/ml_dtypes-0.6.0-cp310-cp310-win_amd64.whl",
              "03ce583adfce34ad33aa9e1fc7a8344dcf90ea776cc4ef0e5a48d4eae84e5d20"),
    "cp311": ("07/23/8870bb62d6e499d6bcbc1242b9f11689bae00a3d39d3684a9aefad8b6ee6/ml_dtypes-0.6.0-cp311-cp311-win_amd64.whl",
              "3035518e3e19add1a4cac9236ab22888b208a4074912514313ccb2d6d242cde8"),
    "cp312": ("ba/db/9c61ec2760b5cbfb1c6558d5c991a6d8fd3271053c32db20506a9a90272b/ml_dtypes-0.6.0-cp312-cp312-win_amd64.whl",
              "2a3e9d53925597fbffafd2a37048dadeddd0bdaba58058f6ae0869ed709a184d"),
    "cp313": ("e2/55/4561acefa00fa4bcbfb82ca6a48578b41f372cd7dd7cdd6eb4720abc2e5f/ml_dtypes-0.6.0-cp313-cp313-win_amd64.whl",
              "fb87f46b4f7ad7b5d3ad8f4b452b024bd4229d44c8ff934798c1fe656210387a"),
}
_PURE = [
    ("8a/55/b77bda4e5e5f5971fb51b07663694690e9afdb9402136c16a522bd621cad/protobuf-7.36.2-cp310-abi3-win_amd64.whl",
     "a300819d441e078a5608c0d3c709796bb548136058fda017ae51d425b44fd353"),
    ("63/34/ba1c580383c9eada3711951fef0795c80b829a078d72188184bcab9dd527/packaging-26.3-py3-none-any.whl",
     "d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c"),
    ("49/d3/b8441a820a491ddfc024b0b0cf0393375b75ea13866d9c66727e54c2fc80/typing_extensions-4.16.0-py3-none-any.whl",
     "481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8"),
]  # fmt: skip
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
WORKER_FLAG = "--separator-worker"


def unavailable_reason() -> str | None:
    """Why the separator can't run on this computer, or None if it can (given a graphics chip)."""
    if sys.platform != "win32":
        return "it needs Windows for now (DirectML)"
    if any(_python_tag() not in table for table in (_WHEELS, _ONNX, _ML_DTYPES)):
        return f"there is no DirectML build for Python {sys.version_info.major}.{sys.version_info.minor}"
    return None


def _python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def runtime_dir() -> Path:
    return models.models_dir() / "runtime" / f"separator-{ORT_VERSION}-{_python_tag()}"


def ensure_runtime(progress: models.Progress | None = None) -> Path:
    """The folder with the helper's private packages (onnxruntime's DirectML build, and what
    converting the model needs), downloading them if needed (~35 MB)."""
    root = runtime_dir()
    marker = root / ".tfv-complete"
    if not marker.exists():
        log.info("Downloading the DirectML runtime (onnxruntime-directml %s, MIT; onnx, protobuf, packaging)", ORT_VERSION)
        root.mkdir(parents=True, exist_ok=True)
        tag = _python_tag()
        for wheel_path, sha256 in [_WHEELS[tag], _ONNX[tag], _ML_DTYPES[tag], *_PURE]:
            wheel = root / wheel_path.rsplit("/", 1)[1]
            models.download(_PYPI + wheel_path, wheel, sha256=sha256, progress=progress, label="DirectML runtime")
            with zipfile.ZipFile(wheel) as z:  # all of it: onnx keeps DLLs in onnx.libs/, its version in .dist-info
                z.extractall(root)
            wheel.unlink()
        marker.write_text("ok")
    return root


# -- the model ------------------------------------------------------------------------------


class Mdx:
    """MDX-Net inference on 16 kHz mono audio: STFT, the model on ~6 s windows, inverse STFT.

    `session` is an onnxruntime InferenceSession (or anything with the same `run`); `meta` is the
    model's metadata (sample_rate, n_fft, hop_length, dim_f, dim_t), as in sherpa-onnx's exports.
    """

    STEP_S = 5.78
    """Seconds of audio one run of the Voc_FT model covers (its window less the edges)."""

    def __init__(self, session, meta: dict[str, str]):
        self.session = session
        self.rate = int(meta["sample_rate"])
        self.n_fft, self.hop = int(meta["n_fft"]), int(meta["hop_length"])
        self.dim_f, self.dim_t = int(meta["dim_f"]), int(meta["dim_t"])
        if self.n_fft % self.hop:
            raise ValueError("n_fft must be a multiple of hop_length")
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(self.n_fft) / self.n_fft)).astype(np.float32)  # periodic Hann
        self.chunk = self.hop * (self.dim_t - 1)  # samples per model window (center=True gives dim_t frames)
        self.trim = self.n_fft // 2  # each window's edges are less reliable: only its middle is kept
        self.step = self.chunk - 2 * self.trim
        self._norm = self._overlap_add(np.broadcast_to(self.window**2, (1, self.dim_t, self.n_fft)))[0]
        self._norm[self._norm < 1e-8] = 1.0

    def vocals(self, audio: np.ndarray) -> np.ndarray:
        """The voices in `audio` (16 kHz mono float32), without the music; same length."""
        import soxr

        x = soxr.resample(np.asarray(audio, np.float32), SAMPLE_RATE, self.rate)
        n = len(x)
        tail = self.step - n % self.step
        padded = np.concatenate([np.zeros(self.trim), x, np.zeros(tail + self.trim)]).astype(np.float32)
        out = [self._window(padded[i : i + self.chunk])[self.trim : self.trim + self.step] for i in range(0, n + tail, self.step)]
        y = soxr.resample(np.concatenate(out)[:n], self.rate, SAMPLE_RATE).astype(np.float32)
        return np.pad(y, (0, max(0, len(audio) - len(y))))[: len(audio)]

    def _window(self, wave: np.ndarray) -> np.ndarray:
        """One model run on `chunk` samples of mono audio. The model takes stereo: both sides get
        the same spectrogram, and its two outputs are averaged before the one inverse transform
        (the same result as transforming both, at half the work)."""
        pad = self.n_fft // 2
        padded = np.pad(wave, (pad, pad), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(padded, self.n_fft)[:: self.hop][: self.dim_t] * self.window
        spec = np.fft.rfft(frames, axis=-1).T[: self.dim_f]  # (dim_f, frames)
        side = np.stack([spec.real, spec.imag])
        y = self.session.run(None, {"input": np.concatenate([side, side])[None].astype(np.float32, copy=False)})[0][0]
        y = (y[:2] + y[2:]) / 2  # left and right: real, imaginary
        y = np.concatenate([y, np.zeros((2, self.n_fft // 2 + 1 - self.dim_f, self.dim_t), np.float32)], axis=1)
        frames = np.fft.irfft((y[0] + 1j * y[1]).T, n=self.n_fft, axis=-1) * self.window
        return (self._overlap_add(frames[None]) / self._norm)[0, pad : pad + self.chunk]

    def _overlap_add(self, frames: np.ndarray) -> np.ndarray:
        """(channels, frames, n_fft) -> (channels, samples), frames `hop` apart."""
        c, t, _ = frames.shape
        r = self.n_fft // self.hop
        blocks = frames.reshape(c, t, r, self.hop)
        out = np.zeros((c, t + r - 1, self.hop), np.float32)
        for k in range(r):
            out[:, k : k + t] += blocks[:, :, k]
        return out.reshape(c, -1)


class StreamingVoices:
    """The voices of a live stream, separated a little at a time, for song mode: every `hop` seconds,
    one model run over the last `window` seconds, ending `lookahead` seconds after the part it emits
    (the model wants some audio on both sides), of which only the newest `hop` seconds are kept. Costs
    one run per `hop` of audio; audio comes out hop + lookahead + one run after it went in.

    On a song with music 5 dB louder than the singing, the vocals came out 15.0 dB cleaner than the
    mix (16.7 dB when the whole song is separated at once), and the VAD found 74% of the singing in
    them instead of 46% in the denoised mix (EVAL.md)."""

    def __init__(self, separator, hop: float = 0.75, lookahead: float = 0.25, window: float = 5.5):
        self._separator = separator
        self._hop, self._ahead, self._window = (int(x * SAMPLE_RATE) for x in (hop, lookahead, window))
        self._buf = np.zeros(0, np.float32)  # audio not emitted yet, after up to `window` of what preceded it
        self._done = 0  # how much of _buf was emitted

    def push(self, audio: np.ndarray) -> list[np.ndarray]:
        """Add audio; returns the separated blocks that are ready (each `hop` long)."""
        self._buf = np.concatenate([self._buf, np.asarray(audio, np.float32)])
        blocks = []
        while len(self._buf) - self._done >= self._hop + self._ahead:
            blocks.append(self._emit(self._done + self._hop, self._done + self._hop + self._ahead))
        return blocks

    def flush(self) -> np.ndarray:
        """The rest, separated (when the music stops or the stream ends); then start over."""
        block = self._emit(len(self._buf), len(self._buf)) if len(self._buf) > self._done else np.zeros(0, np.float32)
        self._buf, self._done = np.zeros(0, np.float32), 0
        return block

    def _emit(self, upto: int, end: int) -> np.ndarray:
        start = max(0, end - self._window)
        voices = self._separator.vocals(self._buf[start:end])
        block = voices[self._done - start : upto - start]
        self._done = upto
        keep = max(0, self._done - self._window)  # no later run reaches back further
        self._buf, self._done = self._buf[keep:], self._done - keep
        return block


# -- the helper process ---------------------------------------------------------------------
# Messages both ways: 1 byte kind (J: JSON, A: float32 audio), 4 bytes length, then the payload.


def _send(stream, kind: bytes, payload: bytes) -> None:
    stream.write(kind + struct.pack("<I", len(payload)) + payload)
    stream.flush()


def _receive(stream) -> tuple[bytes, bytes]:
    header = stream.read(5)
    if len(header) < 5:
        raise EOFError("the separator process ended")
    size = struct.unpack("<I", header[1:])[0]
    payload = stream.read(size)
    if len(payload) < size:
        raise EOFError("the separator process ended")
    return header[:1], payload


def worker_main(argv: list[str]) -> int:
    """The helper process: `<runtime folder> <model file>`; "-" as the runtime echoes audio back
    unchanged (for tests, without a GPU)."""
    runtime, model = argv
    stdin, stdout = sys.stdin.buffer, sys.stdout.buffer
    if runtime == "-":
        ort = None
        _send(stdout, b"J", json.dumps({"devices": [{"index": 0, "name": "echo", "discrete": True, "memory_mb": 8192}]}).encode())
    else:
        ort = _import_private_onnxruntime(Path(runtime))
        devices = [
            {"index": int(d.device.metadata.get("DxgiAdapterNumber", -1)), "name": d.device.metadata.get("Description", "?"),
             "discrete": d.device.metadata.get("Discrete") == "1", "memory_mb": int(d.device.metadata.get("DxgiVideoMemory", "0 MB").split()[0])}
            for d in ort.get_ep_devices() if d.ep_name == "DmlExecutionProvider"
        ]  # fmt: skip
        _send(stdout, b"J", json.dumps({"devices": devices, "providers": ort.get_available_providers()}).encode())
    mdx = None
    while True:
        try:
            kind, payload = _receive(stdin)
        except EOFError:
            return 0
        try:
            if kind == b"J":
                request = json.loads(payload)
                if ort is None:
                    mdx = "echo"
                    _send(stdout, b"J", json.dumps({"ok": True, "rtf": 0.01}).encode())
                    continue
                mdx = _load(ort, model, request["device"])
                window_s = mdx.step / mdx.rate  # audio one model run covers (~5.8 s); time exactly one
                test = np.random.default_rng(0).standard_normal(int(0.95 * window_s * SAMPLE_RATE)).astype(np.float32) * 0.1
                mdx.vocals(test)  # the first run compiles the GPU shaders; time the second
                t0 = time.perf_counter()
                mdx.vocals(test)
                _send(stdout, b"J", json.dumps({"ok": True, "rtf": (time.perf_counter() - t0) / window_s}).encode())
            elif mdx is None:
                _send(stdout, b"J", json.dumps({"error": "no device loaded"}).encode())
            else:
                audio = np.frombuffer(payload, np.float32)
                vocals = audio if mdx == "echo" else mdx.vocals(audio)
                _send(stdout, b"A", np.ascontiguousarray(vocals, np.float32).tobytes())
        except Exception as exc:  # report it; the client decides what to do
            _send(stdout, b"J", json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode())


def _load(ort, model: str, device: int) -> Mdx:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    if device < 0:
        providers = ["CPUExecutionProvider"]
    else:  # DirectML wants these off
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = [("DmlExecutionProvider", {"device_id": device})]
        model = _half_precision(model)
    session = ort.InferenceSession(model, sess_options=options, providers=providers)
    return Mdx(session, session.get_modelmeta().custom_metadata_map)


def _half_precision(model: str) -> str:
    """A float16 copy of the model, made once next to it with onnxruntime's own converter (~2 s).
    Same output (vocals 20.4 dB cleaner either way), half the graphics memory, and on Intel Iris Xe
    graphics 1.5 s per run instead of 2.4 s."""
    path = Path(model)
    half = path.with_name(path.stem + ".fp16.onnx")
    if not half.exists():
        import warnings

        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # "truncated to 1e-07" for a few tiny weights
            converted = convert_float_to_float16(onnx.load(str(path)), keep_io_types=True)
        tmp = half.with_name(half.name + ".tmp")
        onnx.save(converted, str(tmp))
        tmp.replace(half)
    return str(half)


def _import_private_onnxruntime(runtime: Path):
    """Import the DirectML build from `runtime`, not the onnxruntime installed with tfv."""
    if "onnxruntime" in sys.modules:
        raise RuntimeError("onnxruntime was imported before the DirectML build")
    sys.path.insert(0, str(runtime))
    if getattr(sys, "frozen", False):  # a packaged build finds its bundled copy first otherwise
        import importlib.abc
        import importlib.machinery

        private = ("onnxruntime", "onnx", "google", "ml_dtypes", "packaging", "typing_extensions")

        class _Private(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in private:
                    return importlib.machinery.PathFinder.find_spec(name, path if path else [str(runtime)])
                return None

        sys.meta_path.insert(0, _Private())
    import onnxruntime

    if not str(Path(onnxruntime.__file__).resolve()).startswith(str(runtime.resolve())):
        raise RuntimeError(f"imported {onnxruntime.__file__}, not the DirectML build")
    return onnxruntime


# -- the client -------------------------------------------------------------------------------


class VocalSeparator:
    """Runs the separator in its helper process on the best graphics chip that is fast enough."""

    def __init__(self, runtime: Path | str, model: Path | str):
        if getattr(sys, "frozen", False):  # the console tfv.exe, next to tfv-app.exe, runs it (cli.main)
            command = [str(Path(sys.executable).with_name("tfv.exe")), WORKER_FLAG, str(runtime), str(model)]
        else:
            command = [sys.executable, "-m", "tfv.engine.separate", WORKER_FLAG, str(runtime), str(model)]
        self._proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW,
        )  # fmt: skip
        self._lock = threading.Lock()
        self._stderr: collections.deque[str] = collections.deque(maxlen=20)
        threading.Thread(target=self._log_stderr, name="tfv-separator-log", daemon=True).start()
        hello = self._reply()
        self.devices: list[dict] = hello["devices"]
        self.device: dict | None = None
        self.rtf = 0.0
        """Seconds per second of audio, measured on one model run when the device was chosen."""
        self.seconds = 0.0
        """Time spent separating so far."""

    @property
    def run_s(self) -> float:
        """Seconds one model run (~5.8 s of audio) takes on the chosen device."""
        return self.rtf * Mdx.STEP_S

    @property
    def description(self) -> str:
        return f"UVR MDX-Net on {self.device['name']}" if self.device else "UVR MDX-Net (not loaded)"

    def choose_device(self, free_mb: int | None = None) -> dict | None:
        """Load the model on the first graphics chip that has room and is fast enough; returns it
        (None if there is none). `free_mb`: free memory on an NVIDIA card, if known."""
        for device in self._candidates(self.devices, free_mb):
            reply = self._call(b"J", json.dumps({"device": device["index"]}).encode())
            if "error" in reply:
                log.warning("Separator on %s failed: %s", device["name"], reply["error"])
            elif reply["rtf"] > MAX_RTF:
                log.info("Separator on %s: %.2fx real time, too slow", device["name"], reply["rtf"])
            else:
                self.device, self.rtf = device, reply["rtf"]
                log.info("Separator on %s: %.2fx real time", device["name"], reply["rtf"])
                return device
        return None

    @staticmethod
    def _candidates(devices: list[dict], free_mb: int | None) -> list[dict]:
        """Discrete cards with room first (biggest first), then integrated graphics. In half
        precision it shares a 4 GB RTX 3050 with TranslateGemma 4B: translation took 0.36 s per line,
        0.56 s while separating nonstop (in full precision: 9.6 s, as memory was swapped)."""

        def has_room(d: dict) -> bool:
            if "nvidia" in d["name"].lower() and free_mb is not None:
                return free_mb >= NEED_MB
            return d["memory_mb"] >= 4000  # other cards: no way to see what is in use

        discrete = sorted((d for d in devices if d["discrete"] and has_room(d)), key=lambda d: -d["memory_mb"])
        return discrete + [d for d in devices if not d["discrete"]]

    def vocals(self, audio: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        kind, payload = self._exchange(b"A", np.ascontiguousarray(audio, np.float32).tobytes())
        if kind == b"J":
            raise RuntimeError(json.loads(payload).get("error", "separation failed"))
        self.seconds += time.perf_counter() - t0
        return np.frombuffer(payload, np.float32).copy()

    def close(self) -> None:
        """Stop the helper process (after the line it may be separating), freeing its memory."""
        with self._lock:
            if self._proc.poll() is None:
                try:
                    self._proc.stdin.close()
                    self._proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    self._proc.kill()

    def _call(self, kind: bytes, payload: bytes) -> dict:
        kind, payload = self._exchange(kind, payload)
        return json.loads(payload)

    def _reply(self) -> dict:
        with self._lock:
            kind, payload = self._receive()
        reply = json.loads(payload)
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def _exchange(self, kind: bytes, payload: bytes) -> tuple[bytes, bytes]:
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError("the separator process has stopped")
            try:
                _send(self._proc.stdin, kind, payload)
            except OSError:
                pass  # it ended: _receive says why
            return self._receive()

    def _receive(self) -> tuple[bytes, bytes]:
        try:
            return _receive(self._proc.stdout)
        except EOFError:
            self._proc.wait(timeout=5)
            time.sleep(0.2)  # let the last error lines arrive
            raise RuntimeError("the separator process stopped: " + (" | ".join(list(self._stderr)[-3:]) or "no message")) from None

    def _log_stderr(self) -> None:
        for raw in self._proc.stderr:
            line = raw.decode("utf-8", "replace").rstrip()
            self._stderr.append(line)
            log.debug("separator: %s", line)


def load(progress: models.Progress | None = None) -> VocalSeparator | None:
    """Download what is needed and start the separator on a suitable graphics chip; None (and
    a log line saying why) if this computer has none."""
    if (reason := unavailable_reason()) is not None:
        log.info("No music separation: %s", reason)
        return None
    runtime = ensure_runtime(progress)
    model = models.ensure(MODEL_ID, progress).main_file
    separator = VocalSeparator(runtime, model)
    if separator.choose_device(hardware.nvidia_free_mb()) is None:
        separator.close()
        log.info("No music separation: no graphics chip with room that runs it fast enough")
        return None
    return separator


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == WORKER_FLAG:
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        raise SystemExit(worker_main(sys.argv[2:]))
