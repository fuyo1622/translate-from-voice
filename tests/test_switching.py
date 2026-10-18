"""Switching languages while running: as few reloads as possible, and only the latest choice applied."""

import os
import threading
import time

import pytest
from PySide6.QtCore import Qt

from tfv.hardware import GPU, Hardware
from tfv.resources import Choice, choose_profile
from tfv.session import Session, SessionConfig

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class Named(str):
    """A fake model: compares like its name and has a description like a real backend."""

    @property
    def description(self):
        return str(self)


class FakePipeline:
    separator_error = None

    def __init__(self):
        self.source_lang, self.target_lang = "ja", "zh"
        self.translator = Named("translategemma")
        self.calls = []

    def replace_asr(self, asr, source_lang=None):
        self.calls.append(("asr", asr))
        if source_lang:
            self.source_lang = source_lang
        return Named("old-asr")

    def replace_translator(self, translator, target_lang=None):
        self.calls.append(("translator", translator))
        old, self.translator = self.translator, translator
        if target_lang:
            self.target_lang = target_lang
        return old

    def set_target(self, target_lang):
        self.calls.append(("target", target_lang))
        self.target_lang = target_lang


class FakeLadder:
    def __init__(self, rungs, loads, kind):
        self.rungs, self.loads, self.kind, self.index = rungs, loads, kind, 0

    def load(self, start=0):
        self.loads.append((self.kind, str(self.rungs[0])))
        return Named(f"{self.kind}:{self.rungs[0]}")


@pytest.fixture
def session(monkeypatch):
    """A running gpu-small session (RTX 3050 4 GB), Japanese → Chinese, with fake models."""
    monkeypatch.setattr("tfv.session._close", lambda backend: closed.append(backend))
    closed, loads, statuses = [], [], []
    s = Session(SessionConfig("ja", "zh"), lambda c: None, statuses.append)
    s.hardware = Hardware("windows", "amd64", "x", 4, 8, 16000, 8000, [GPU("RTX 3050", "nvidia", 4096)], cuda=True)
    s.profile = choose_profile(s.hardware)
    s.pipeline = FakePipeline()
    s.asr, s.translator = Named("reazonspeech"), s.pipeline.translator
    s._asr_ladder = lambda: FakeLadder(s.profile.asr_for(s.config.source_lang), loads, "asr")
    s._mt_ladder = lambda: FakeLadder(s.profile.mt_for(s.config.target_lang), loads, "mt")
    s.asr_ladder = s._asr_ladder()
    s.test = {"closed": closed, "loads": loads, "statuses": statuses}
    return s


def test_new_target_retargets_the_loaded_translator(session):
    session.set_languages("ja", "en")
    assert session.test["loads"] == [] and session.test["closed"] == []  # instant: nothing reloaded
    assert session.pipeline.target_lang == "en" and session.translator == "translategemma"
    assert session.summary() == "Now: 日本語 → English"


def test_japanese_chinese_japanese_keeps_the_translator(session):
    session.set_languages("zh", "zh")  # Chinese speech: SenseVoice on the CPU, captions only
    assert session.test["loads"] == [("asr", "sensevoice-small@cpu")]
    assert session.translator == "translategemma" and session.test["closed"] == ["old-asr"]  # translator kept
    assert session.summary() == "Now: 中文, captions only"
    assert session.describe()[1] == "Translation: off"
    session.set_languages("ja", "zh")
    assert session.test["loads"][-1] == ("asr", "reazonspeech-k2-v2@cpu")
    assert [k for k, _ in session.test["loads"]].count("mt") == 0  # translation back without a reload
    assert session.pipeline.target_lang == "zh" and session.summary() == "Now: 日本語 → 中文"
    session.set_languages("ja", "en")
    assert [k for k, _ in session.test["loads"]].count("mt") == 0 and session.pipeline.target_lang == "en"


def test_english_speech_frees_the_gpu_for_whisper_first(session):
    session.set_languages("en", "zh")
    assert session.test["closed"][0] == "translategemma"  # freed before Whisper loads
    assert session.test["loads"] == [("asr", "whisper-large-v3-turbo@cuda"), ("mt", "translategemma-4b@gpu")]
    assert session.translator == "mt:translategemma-4b@gpu" and session.pipeline.translator == session.translator


def test_turning_translation_on_loads_a_translator_once(session):
    session.translator = None
    session.pipeline.translator = None
    session.set_languages("ja", None)
    assert session.test["loads"] == [] and session.summary() == "Now: 日本語, captions only"
    session.set_languages("ja", "en")
    assert session.test["loads"] == [("mt", "qwen3.5-4b@gpu")]
    assert any("Loading the translator for English" in m for m in session.test["statuses"])


def test_rapid_choices_apply_only_the_latest(monkeypatch):
    from tfv.ui.controller import EngineController
    from tfv.ui.prefs import Prefs

    class SlowSession:
        def __init__(self):
            self.applied = []

        def set_languages(self, source, target):
            self.applied.append((source, target))
            time.sleep(0.3)  # loading a model

        def summary(self):
            return f"Now: {self.applied[-1]}"

    monkeypatch.setattr(Prefs, "save", lambda self: None)
    controller = EngineController(Prefs())
    controller.session = SlowSession()
    done = threading.Event()
    statuses = []
    controller.status.connect(lambda m: (statuses.append(m), done.set()), Qt.ConnectionType.DirectConnection)
    for choice in [("zh", "zh"), ("ja", "zh"), ("ja", "en"), ("ja", "ja"), ("ja", "en")]:
        controller.set_languages(*choice)
        time.sleep(0.05)  # clicking through the menu
    assert done.wait(5)
    assert controller.session.applied == [("zh", "zh"), ("ja", "en")]  # the first, then only the latest
    assert statuses == ["Now: ('ja', 'en')"] and not controller.switching
    assert (controller.prefs.source_lang, controller.prefs.target_lang) == ("ja", "en")
