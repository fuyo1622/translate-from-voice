"""The music filter: MDX-Net framing, the helper process, device choice, and its use in the pipeline."""

import time
from pathlib import Path

import numpy as np
import pytest
import soxr

from tfv import SAMPLE_RATE, models
from tfv.engine import separate
from tfv.engine.pipeline import Pipeline
from tfv.engine.segmenter import Segmenter
from tfv.engine.separate import Mdx, StreamingVoices, VocalSeparator

from test_pipeline import GrowingASR, PacedSource, TaggingTranslator, fake_vad, speech

VOC_FT = {"sample_rate": "44100", "n_fft": "6144", "hop_length": "1024", "dim_f": "3072", "dim_t": "256"}


class Identity:
    """Stands in for the model: returns its input, so the framing alone is tested."""

    def __init__(self):
        self.calls = 0

    def run(self, outputs, feeds):
        self.calls += 1
        x = feeds["input"]
        assert x.shape == (1, 4, 3072, 256) and x.dtype == np.float32
        return [x]


def si_snr(ref: np.ndarray, est: np.ndarray) -> float:
    a = np.dot(est, ref) / np.dot(ref, ref)
    return float(10 * np.log10(np.sum((a * ref) ** 2) / np.sum((a * ref - est) ** 2)))


@pytest.mark.parametrize("seconds", [0.3, 5.0, 13.7])
def test_mdx_framing_gives_back_what_the_model_returns(seconds):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    audio = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 1750 * t + 1) + 0.1 * np.sin(2 * np.pi * 5100 * t)).astype(np.float32)
    session = Identity()
    mdx = Mdx(session, VOC_FT)
    out = mdx.vocals(audio)
    assert out.shape == audio.shape and out.dtype == np.float32
    assert si_snr(audio, out) > 35  # windows stitched without gaps or clicks
    assert mdx.step == 1024 * 255 - 6144  # each ~5.8 s window keeps its middle
    assert session.calls == len(soxr.resample(audio, SAMPLE_RATE, 44100)) // mdx.step + 1


def test_candidates_prefer_a_discrete_card_with_room():
    rtx = {"index": 1, "name": "NVIDIA GeForce RTX 3050 Laptop GPU", "discrete": True, "memory_mb": 3964}
    radeon = {"index": 2, "name": "AMD Radeon RX 7600", "discrete": True, "memory_mb": 8176}
    iris = {"index": 0, "name": "Intel(R) Iris(R) Xe Graphics", "discrete": False, "memory_mb": 128}
    devices = [iris, rtx, radeon]
    assert VocalSeparator._candidates(devices, free_mb=2000) == [radeon, rtx, iris]
    assert VocalSeparator._candidates(devices, free_mb=600) == [radeon, iris]  # the translator filled the RTX
    assert VocalSeparator._candidates([rtx, iris], free_mb=None) == [iris]  # unknown use on a small card: not risked
    assert VocalSeparator._candidates([], free_mb=None) == []


def test_helper_process_round_trip():
    separator = VocalSeparator("-", "unused")  # "-": the helper echoes audio back, no GPU needed
    try:
        assert [d["name"] for d in separator.devices] == ["echo"]
        with pytest.raises(RuntimeError, match="no device loaded"):
            separator.vocals(np.zeros(16, np.float32))
        assert separator.choose_device()["name"] == "echo" and separator.description == "UVR MDX-Net on echo"
        audio = np.random.default_rng(1).standard_normal(5 * SAMPLE_RATE).astype(np.float32)
        assert np.array_equal(separator.vocals(audio), audio)
        assert len(separator.vocals(np.zeros(0, np.float32))) == 0
    finally:
        separator.close()
    with pytest.raises(RuntimeError, match="stopped"):
        separator.vocals(np.zeros(16, np.float32))


class FakeDenoiser:
    """Takes `removed_db` out of everything, like GTCRN does from music."""

    def __init__(self, removed_db: float):
        self.gain = 10 ** (-removed_db / 20)

    def push(self, audio):
        return audio, audio * self.gain

    def flush(self):
        return np.zeros(0, np.float32), np.zeros(0, np.float32)

    def reset(self):
        pass


class FakeSeparator:
    def __init__(self, fail: bool = False, rtf: float = 0.1):
        self.calls, self.fail, self.rtf = 0, fail, rtf

    def vocals(self, audio):
        self.calls += 1
        if self.fail:
            raise RuntimeError("out of graphics memory")
        time.sleep(0.01)
        return audio


def run_pipeline(removed_db: float, separator, live: bool = False) -> tuple[Pipeline, list]:
    captions = []
    pipeline = Pipeline(
        PacedSource(np.concatenate([speech(0.3, 1.5, 1.0), speech(0.3, 1.5, 1.0)]), realtime=live), Segmenter(fake_vad),
        GrowingASR(), captions.append, source_lang="ja", target_lang="zh", translator=TaggingTranslator(),
        vad_denoiser=FakeDenoiser(removed_db), separator=separator, partials=live, partial_interval=0.3,
    )  # fmt: skip
    pipeline.start()
    assert pipeline.wait(20)
    return pipeline, list({c.id: c for c in captions if c.stage != "partial"}.values())  # one per line


def test_only_lines_with_music_are_separated():
    separator = FakeSeparator()
    _, lines = run_pipeline(4.4, separator)  # (the fake VAD still hears the speech at -4.4 dB)
    assert len(lines) == 2 and separator.calls == 2 and all(c.sep_s > 0 for c in lines)
    separator = FakeSeparator()
    _, lines = run_pipeline(0.0, separator)  # clean speech: nothing to take out
    assert len(lines) == 2 and separator.calls == 0 and all(c.sep_s == 0 for c in lines)


def test_live_text_skips_a_slow_separator():
    fast, slow = FakeSeparator(rtf=0.1), FakeSeparator(rtf=0.45)
    _, lines = run_pipeline(6.0, fast, live=True)
    assert len(lines) == 2 and fast.calls > 2  # the lines and their live text
    _, lines = run_pipeline(6.0, slow, live=True)
    assert len(lines) == 2 and slow.calls == 2  # only the finished lines


def test_a_failing_separator_is_turned_off_and_lines_still_come():
    separator = FakeSeparator(fail=True)
    pipeline, lines = run_pipeline(6.0, separator)
    assert len(lines) == 2 and separator.calls == 1  # tried once, then off
    assert pipeline.separator is None and "out of graphics memory" in pipeline.separator_error


def test_session_reports_why_there_is_no_music_filter(monkeypatch):
    from tfv.session import Session, SessionConfig

    monkeypatch.setattr(separate, "unavailable_reason", lambda: "it needs Windows for now (DirectML)")
    session = Session(SessionConfig("ja", None), lambda c: None, lambda m: None)
    session.set_separate_vocals(True)
    assert session.separator is None
    assert session.describe()[-1] == "Music filter: not available: it needs Windows for now (DirectML)"
    session.set_separate_vocals(False)
    assert session.describe()[-1] == "Music filter: off"


def test_session_turns_the_filter_off_and_frees_it(monkeypatch):
    from tfv.session import Session, SessionConfig

    class Loaded:
        description = "UVR MDX-Net on test GPU"
        run_s = 0.4  # one model run: fast enough for song mode
        closed = False

        def close(self):
            self.closed = True

    loaded = Loaded()
    monkeypatch.setattr(separate, "unavailable_reason", lambda: None)
    monkeypatch.setattr(separate, "load", lambda progress=None: loaded)
    statuses = []
    session = Session(SessionConfig("ja", None), lambda c: None, statuses.append)
    session.set_separate_vocals(True)
    assert session.separator is loaded and session.describe()[-1] == "Music filter: UVR MDX-Net on test GPU; song mode off"
    assert statuses[-1] == "Music filter: UVR MDX-Net on test GPU; song mode off"
    session.set_song_mode(True)
    assert session.describe()[-1] == "Music filter: UVR MDX-Net on test GPU; song mode on"
    loaded.run_s = 1.5  # e.g. integrated graphics: whole lines only
    assert session.describe()[-1] == "Music filter: UVR MDX-Net on test GPU; song mode needs a faster graphics chip"
    session.set_separate_vocals(False)
    assert loaded.closed and session.separator is None and session.describe()[-1] == "Music filter: off"


def _real_separator_ready() -> bool:
    if separate.unavailable_reason() is not None or not models.is_installed(separate.MODEL_ID):
        return False
    return (separate.runtime_dir() / ".tfv-complete").exists() and Path("samples/ja.wav").exists()


@pytest.mark.skipif(not _real_separator_ready(), reason="needs the DirectML runtime, the model and samples/ja.wav")
def test_the_real_model_takes_music_out_from_under_speech():
    from faster_whisper.audio import decode_audio

    voice = decode_audio("samples/ja.wav", SAMPLE_RATE)[: 10 * SAMPLE_RATE]
    t = np.arange(len(voice)) / SAMPLE_RATE
    chords = sum(np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t)) for f in (110, 220, 277.2, 329.6, 440))
    beat = np.random.default_rng(0).standard_normal(len(t)) * (np.mod(t, 0.5) < 0.05)  # a hi-hat every half second
    music = (chords + 0.5 * beat).astype(np.float32)
    music *= np.sqrt(np.mean(voice**2) / np.mean(music**2))  # as loud as the voice
    mix = (voice + music).astype(np.float32)
    separator = separate.load()
    assert separator is not None, "no graphics chip ran it fast enough"
    try:
        out = separator.vocals(mix)
    finally:
        separator.close()
    assert len(out) == len(mix)
    assert si_snr(voice, out) > si_snr(voice, mix) + 6, (si_snr(voice, mix), si_snr(voice, out))


# -- song mode ----------------------------------------------------------------------------------


class Doubler:
    """Stands in for the separator: returns twice its input, so its output can be recognized."""

    run_s = 0.3

    def __init__(self):
        self.windows = []

    def vocals(self, audio):
        self.windows.append(len(audio))
        return audio * 2


def test_streaming_voices_give_every_sample_once_in_order():
    audio = np.arange(10 * SAMPLE_RATE, dtype=np.float32)
    separator = Doubler()
    stream = StreamingVoices(separator, hop=1.0, lookahead=0.3, window=5.5)
    blocks = []
    for i in range(0, len(audio), 1234):  # uneven chunks, as capture delivers them
        blocks += stream.push(audio[i : i + 1234])
    assert [len(b) for b in blocks] == [SAMPLE_RATE] * 9  # the 10th second waits for its lookahead
    blocks.append(stream.flush())
    assert np.array_equal(np.concatenate(blocks), audio * 2)  # all from the separator, none twice or missing
    assert max(separator.windows) <= 5.5 * SAMPLE_RATE and len(stream.flush()) == 0
    stream = StreamingVoices(Doubler(), hop=1.0, lookahead=0.3)
    assert stream.push(np.zeros(int(1.2 * SAMPLE_RATE), np.float32)) == []  # needs 0.3 s after the second
    assert len(stream.push(np.zeros(int(0.2 * SAMPLE_RATE), np.float32))) == 1


class ScheduledDenoiser(FakeDenoiser):
    """Removes `before_db` until `switch_s` seconds of audio, then `after_db` (the music stops)."""

    def __init__(self, before_db: float, after_db: float, switch_s: float):
        super().__init__(before_db)
        self.after, self.switch, self.heard = 10 ** (-after_db / 20), int(switch_s * SAMPLE_RATE), 0

    def push(self, audio):
        self.heard += len(audio)
        return audio, audio * (self.gain if self.heard <= self.switch else self.after)


class FastSeparator(FakeSeparator):
    run_s = 0.3  # fast enough for song mode


def run_song(audio, denoiser, separator):
    captions = []
    pipeline = Pipeline(
        PacedSource(audio, realtime=False), Segmenter(fake_vad), GrowingASR(), captions.append,
        source_lang="ja", target_lang="zh", translator=TaggingTranslator(), vad_denoiser=denoiser, separator=separator,
    )  # fmt: skip
    pipeline.start()
    assert pipeline.wait(20)
    return pipeline, list({c.id: c for c in captions if c.stage != "partial"}.values())


def test_song_mode_finds_singing_the_denoised_copy_hides():
    song = np.concatenate([speech(0.3, 1.5, 1.0), speech(0.3, 1.5, 1.0)])
    # the denoiser takes 12 dB out of the singing too: the VAD alone misses both lines
    _, lines = run_song(song, FakeDenoiser(12.0), None)
    assert lines == []
    separator = FastSeparator()
    pipeline, lines = run_song(song, FakeDenoiser(12.0), separator)
    assert len(lines) == 2 and all(c.sep_s == 0 for c in lines)  # separated once, in the stream, not again per line
    assert separator.calls >= 5 and pipeline.song_mode_s > 4


def test_song_mode_ends_when_the_music_stops():
    audio = np.concatenate([speech(0.3, 1.5, 1.2), speech(0.3, 5.0, 1.0)])
    pipeline, lines = run_song(audio, ScheduledDenoiser(12.0, 0.0, switch_s=3.0), FastSeparator())
    assert pipeline._song is None and 2.5 < pipeline.song_mode_s < 8  # left a few seconds after the music did
    assert pipeline._segmenter.config.threshold == 0.5  # the VAD is as strict as before song mode
    assert len(lines) >= 2 and lines[0].text and lines[-1].text


def test_a_failing_separator_leaves_song_mode():
    pipeline, _ = run_song(speech(0.3, 2.0, 1.0), FakeDenoiser(12.0), FastSeparator(fail=True))
    assert pipeline.separator is None and pipeline._song is None and "out of graphics memory" in pipeline.separator_error
