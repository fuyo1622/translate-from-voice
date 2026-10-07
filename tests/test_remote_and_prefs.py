import contextlib
import json
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from tfv.capture.base import AudioSource
from tfv.engine.pipeline import Caption
from tfv.remote import NetworkAudioSource, RemoteSession, _is_loopback, caption_from_message, caption_to_message
from tfv.session import SessionConfig
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


@contextlib.contextmanager
def stub_server(handler):
    """A protocol-speaking stand-in for `tfv serve`, without models."""
    from websockets.sync.server import serve

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}"
        finally:
            server.shutdown()
            thread.join(5)


class SilentSource(AudioSource):
    description = "nothing"

    def start(self, on_audio, on_end):
        pass

    def stop(self):
        pass


def send(ws, **message):
    ws.send(json.dumps(message, ensure_ascii=False))


def test_remote_client_shows_progress_until_ready(monkeypatch):
    def handler(ws):
        assert json.loads(ws.recv())["source_lang"] == "en"
        send(ws, type="status", message="Loading recognition for en…")  # the server reloads for us first
        send(ws, type="ready", protocol=1, engine=["Recognition: stub"])
        for _ in ws:
            pass

    monkeypatch.setattr("tfv.remote.make_source", lambda cfg, status: SilentSource())
    statuses = []
    with stub_server(handler) as url:
        session = RemoteSession(url, None, SessionConfig("en", "zh"), lambda c: None, statuses.append)
        session.start()
        session.stop()
    assert "Loading recognition for en…" in statuses
    assert session.describe() == [f"Remote engine: {url}", "Recognition: stub"]


def test_remote_client_reports_refusal():
    def handler(ws):
        ws.recv()
        send(ws, type="error", message="wrong or missing token")

    with stub_server(handler) as url:
        with pytest.raises(RuntimeError, match="wrong or missing token"):
            RemoteSession(url, "bad", SessionConfig("ja"), lambda c: None).start()


def test_file_over_remote_is_paced_and_keeps_the_last_translation():
    file_s = 1.0
    arrivals = []

    def handler(ws):
        ws.recv()
        send(ws, type="ready", engine=[])
        received = 0
        for message in ws:
            arrivals.append(time.monotonic())
            received += len(message) // 2
            if received >= (file_s + 1.0) * 16000:  # the file plus the client's closing silence
                base = dict(type="caption", id=1, start=0.0, end=file_s, forced=False, source_lang="ja", text="こんにちは。", target_lang="zh")
                send(ws, **base, stage="text")
                time.sleep(1.0)  # a slow translation: longer than the quiet period below
                send(ws, **base, stage="translated", translation="你好。")

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "tone.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((np.sin(np.arange(int(file_s * 16000)) / 5) * 8000).astype("<i2").tobytes())
        captions = []
        with stub_server(handler) as url:
            session = RemoteSession(url, None, SessionConfig("ja", "zh", file=path), captions.append)
            session.drain_quiet_s = 0.3
            session.start()
            assert session.wait(20)
            session.stop()
    assert arrivals[-2] - arrivals[0] >= 0.7 * file_s  # sent at playback speed, not all at once
    assert [(c.stage, c.translation) for c in captions] == [("text", None), ("translated", "你好。")]


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


def test_remote_client_asks_for_live_text(monkeypatch):
    hellos = []

    def handler(ws):
        hellos.append(json.loads(ws.recv()))
        send(ws, type="ready", engine=[])
        for message in ws:
            hellos.append(json.loads(message))

    monkeypatch.setattr("tfv.remote.make_source", lambda cfg, status: SilentSource())
    with stub_server(handler) as url:
        session = RemoteSession(url, None, SessionConfig("ja", "zh", partials=True), lambda c: None)
        session.start()
        session.set_partials(False)
        time.sleep(0.3)
        session.stop()
    assert hellos[0]["live"] is True and hellos[1] == {"type": "live", "live": False}
