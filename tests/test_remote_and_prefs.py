import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tfv.engine.pipeline import Caption
from tfv.remote import NetworkAudioSource, _is_loopback, caption_from_message, caption_to_message
from tfv.ui.prefs import Prefs


def test_caption_survives_the_wire():
    c = Caption(7, 1.5, 4.0, True, "ja", "こんにちは", target_lang="zh", translation="你好", stage="translated", asr_s=0.4, mt_s=0.2)
    message = json.loads(json.dumps(caption_to_message(c), ensure_ascii=False))
    assert message["type"] == "caption"
    back = caption_from_message(message)
    assert (back.id, back.text, back.translation, back.stage, back.forced) == (7, "こんにちは", "你好", "translated", True)


def test_network_source_decodes_pcm16():
    got = []
    src = NetworkAudioSource()
    src.start(got.append, lambda: None)
    src.feed((np.array([0, 16384, -32768], dtype="<i2")).tobytes() + b"\x01")  # odd trailing byte is ignored
    assert np.allclose(got[0], [0.0, 0.5, -1.0])


def test_loopback_detection():
    assert _is_loopback("127.0.0.1") and _is_loopback("localhost") and _is_loopback("::1")
    assert not _is_loopback("0.0.0.0") and not _is_loopback("192.168.1.20")


@pytest.fixture
def config_dir(monkeypatch):
    # tempfile rather than pytest's tmp_path, whose shared base folder can be locked on Windows
    with tempfile.TemporaryDirectory() as path:
        monkeypatch.setenv("TFV_CONFIG_DIR", path)
        yield Path(path)


def test_prefs_roundtrip(config_dir):
    p = Prefs(source_lang="en", target_lang=None, geometry=[10, 900, 800], locked=True)
    p.save()
    loaded = Prefs.load()
    assert (loaded.source_lang, loaded.target_lang, loaded.geometry, loaded.locked) == ("en", None, [10, 900, 800], True)


def test_prefs_ignore_unknown_and_broken_files(config_dir):
    (config_dir / "settings.json").write_text('{"source_lang": "zh", "from_the_future": 1}', "utf-8")
    assert Prefs.load().source_lang == "zh"
    (config_dir / "settings.json").write_text("{not json", "utf-8")
    assert Prefs.load() == Prefs()
