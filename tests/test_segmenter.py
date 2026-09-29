import numpy as np
import pytest

from tfv import SAMPLE_RATE
from tfv.engine.segmenter import Segmenter, SegmenterConfig

WIN = 512


def fake_vad(window: np.ndarray) -> float:
    """Treat loud windows as speech, so tests can script speech with amplitudes."""
    return 0.9 if np.abs(window).mean() > 0.5 else 0.05


def audio(*parts: tuple[str, float]) -> np.ndarray:
    """audio(("silence", 1.0), ("speech", 2.0), ...) -> samples."""
    level = {"silence": 0.0, "speech": 1.0}
    return np.concatenate([np.full(int(s * SAMPLE_RATE), level[kind], np.float32) for kind, s in parts])


def run(samples: np.ndarray, chunk: int = 1600, **cfg) -> list:
    seg = Segmenter(fake_vad, SegmenterConfig(**cfg))
    out = []
    for i in range(0, len(samples), chunk):
        out += seg.push(samples[i : i + chunk])
    if (last := seg.flush()) is not None:
        out.append(last)
    return out


def test_silence_produces_nothing():
    assert run(audio(("silence", 5))) == []


def test_single_utterance_with_padding():
    segs = run(audio(("silence", 1), ("speech", 2), ("silence", 2)))
    assert len(segs) == 1
    s = segs[0]
    assert not s.forced
    # 200 ms padding on each side, quantised to 32 ms windows
    assert s.start == pytest.approx(0.8, abs=0.05)
    assert s.duration == pytest.approx(2.4, abs=0.07)


def test_short_blip_is_dropped():
    assert run(audio(("silence", 1), ("speech", 0.1), ("silence", 1))) == []


def test_pause_splits_utterances():
    segs = run(audio(("speech", 1), ("silence", 1), ("speech", 1), ("silence", 1)))
    assert len(segs) == 2
    assert segs[1].start > segs[0].end - 0.01


def test_short_pause_does_not_split():
    segs = run(audio(("speech", 1), ("silence", 0.3), ("speech", 1), ("silence", 1)))
    assert len(segs) == 1


def test_long_speech_is_split_at_max_length():
    segs = run(audio(("speech", 30), ("silence", 1)), max_segment_s=8)
    assert len(segs) >= 4
    assert all(s.duration <= 8.01 for s in segs)
    assert all(s.forced for s in segs[:-1]) and not segs[-1].forced
    total = sum(s.duration for s in segs)
    assert total == pytest.approx(30.2, abs=0.05)  # speech + trailing pad: nothing lost or duplicated


def test_split_prefers_a_quiet_point():
    # 7 s speech, 0.3 s dip (too short to end the utterance), 3 s speech; cap at 9 s
    segs = run(audio(("speech", 7), ("silence", 0.3), ("speech", 3), ("silence", 1)), max_segment_s=9)
    assert segs[0].forced
    assert 7.0 <= segs[0].end <= 7.4


def test_flush_closes_open_utterance():
    seg = Segmenter(fake_vad)
    assert seg.push(audio(("speech", 1))) == []
    assert seg.active
    s = seg.flush()
    assert s is not None and s.duration == pytest.approx(1.0, abs=0.05)
    assert not seg.active


@pytest.mark.parametrize("chunk", [1, 511, 512, 4800, 99999])
def test_chunk_size_does_not_matter(chunk):
    samples = audio(("silence", 0.5), ("speech", 1.5), ("silence", 1), ("speech", 0.8), ("silence", 1))
    reference = [(s.start, len(s.audio)) for s in run(samples, chunk=1600)]
    assert [(s.start, len(s.audio)) for s in run(samples, chunk=chunk)] == reference
