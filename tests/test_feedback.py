""""Save the last minute": the pipeline keeps recent audio and captions; the session writes them out."""

import os
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import Qt

from tfv import SAMPLE_RATE
from tfv.engine.pipeline import Caption, Pipeline
from tfv.engine.segmenter import Segmenter
from tfv.session import Session, SessionConfig

from test_pipeline import GrowingASR, PacedSource, TaggingTranslator, fake_vad, speech

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def make_pipeline(audio=np.zeros(1, np.float32), history_s=60.0, realtime=False):
    return Pipeline(
        PacedSource(audio, realtime), Segmenter(fake_vad), GrowingASR(), lambda c: None,
        source_lang="ja", target_lang="zh", translator=TaggingTranslator(), history_s=history_s,
    )  # fmt: skip


def test_the_ring_buffer_keeps_the_latest_audio_in_order():
    pipeline = make_pipeline(history_s=16 / SAMPLE_RATE)  # room for 16 samples
    heard = np.arange(100, dtype=np.float32)
    pos = 0
    for size in [5, 11, 10, 3, 20, 1, 16, 2]:  # lands exactly on the end, wraps, overflows in one chunk
        pipeline._remember(heard[pos : pos + size])
        pos += size
        audio, start, _ = pipeline.recent()
        expected = heard[max(0, pos - 16) : pos]
        assert np.array_equal(audio, expected), (size, audio, expected)
        assert start == pytest.approx(max(0, pos - 16) / SAMPLE_RATE)


def test_recent_returns_the_last_stretch_and_only_its_captions():
    # two lines, 2 s apart; keep 2 s of history, so only the second line is in it
    audio = np.concatenate([speech(0.3, 1.0, 1.5), speech(0.3, 1.0, 1.0)])
    pipeline = make_pipeline(audio, history_s=2.0)
    pipeline.start()
    assert pipeline.wait(20)
    kept, start, captions = pipeline.recent()
    assert np.array_equal(kept, audio[-2 * SAMPLE_RATE :])
    assert start == pytest.approx(len(audio) / SAMPLE_RATE - 2.0)
    lines = {c.id: c for c in captions}  # a caption is logged once per stage, as the same object
    assert len(lines) == 1
    (line,) = lines.values()
    assert line.stage == "translated" and line.translation.startswith("譯") and line.start >= 2.5


class Named(str):
    @property
    def description(self):
        return str(self)


class RecentPipeline:
    separator_error = None

    def __init__(self, audio, start, captions):
        self._recent = audio, start, captions

    def recent(self):
        return self._recent


def test_save_recent_writes_the_audio_and_the_captions(monkeypatch):
    # tempfile rather than pytest's tmp_path, whose shared base folder can be locked on Windows
    with tempfile.TemporaryDirectory() as folder:
        monkeypatch.setattr("tfv.models.feedback_dir", lambda: Path(folder) / "feedback")
        check_save_recent()


def check_save_recent():
    session = Session(SessionConfig("ja", "zh"), lambda c: None, lambda m: None)
    session.asr, session.translator = Named("ReazonSpeech"), Named("TranslateGemma 4B")
    first = Caption(7, 100.5, 102.0, False, "ja", "こんにちは", "zh", "你好", stage="translated")
    second = Caption(8, 103.0, 104.0, False, "ja", "さようなら", "zh", None, stage="untranslated")
    audio = np.sin(np.linspace(0, 2000, 5 * SAMPLE_RATE)).astype(np.float32) * 0.5
    session.pipeline = RecentPipeline(audio, 100.0, [first, second, first])  # first logged twice (text, translated)
    path = session.save_recent()
    with wave.open(str(path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, 16000, len(audio))
        saved = np.frombuffer(w.readframes(w.getnframes()), "<i2") / 32767
    assert np.abs(saved - audio).max() < 1e-3
    text = path.with_suffix(".txt").read_text("utf-8").splitlines()
    assert text[:3] == ["Now: 日本語 → 中文", "Recognition: ReazonSpeech", "Translation: TranslateGemma 4B"]
    assert text[-2:] == ["   0.5-   2.0 | こんにちは | 你好", "   3.0-   4.0 | さようなら | "]
    assert session.save_recent() != path  # saving again in the same second keeps both
    session.pipeline = RecentPipeline(np.zeros(0, np.float32), 0.0, [])
    with pytest.raises(RuntimeError, match="nothing heard yet"):
        session.save_recent()


def test_saving_needs_a_local_engine(monkeypatch):
    from tfv.ui.controller import EngineController
    from tfv.ui.prefs import Prefs

    monkeypatch.setattr(Prefs, "save", lambda self: None)
    controller = EngineController(Prefs())
    statuses = []
    controller.status.connect(statuses.append, Qt.ConnectionType.DirectConnection)
    controller.save_recent()
    controller.session = object()  # a remote engine has no recording to save
    controller.save_recent()
    assert statuses == ["Nothing to save yet", "Saving is not available with a remote engine"]

    saved = threading.Event()

    class Local:
        def save_recent(self):
            saved.set()
            return "D:/x.wav"

    controller.session = Local()
    controller.save_recent()
    assert saved.wait(5)
