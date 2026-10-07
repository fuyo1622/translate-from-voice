import numpy as np
import pytest

from tfv import SAMPLE_RATE, models
from tfv.engine.segmenter import Segmenter


def loud_vad(window: np.ndarray) -> float:
    return 0.9 if np.abs(window).mean() > 0.5 else 0.05


def test_segmenter_judges_one_copy_and_cuts_the_other():
    original = np.full(4 * SAMPLE_RATE, 0.3, np.float32)  # too quiet for the VAD on its own
    for_vad = np.zeros_like(original)
    for_vad[SAMPLE_RATE : 3 * SAMPLE_RATE] = 1.0  # the cleaned copy shows speech from 1 s to 3 s
    seg = Segmenter(loud_vad)
    done = seg.push(original, for_vad)
    assert len(done) == 1 and done[0].start == pytest.approx(0.8, abs=0.05)
    assert np.allclose(done[0].audio, 0.3)  # cut from the original audio
    assert seg.push(original) == []  # without the copy, the VAD hears nothing


@pytest.mark.skipif(not models.is_installed("gtcrn"), reason="GTCRN not downloaded (python -m tfv models download gtcrn)")
def test_denoiser_keeps_the_two_copies_aligned():
    from tfv.engine.denoise import VadDenoiser

    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(3 * SAMPLE_RATE) * 0.1).astype(np.float32)
    d = VadDenoiser(models.get("gtcrn").main_file)
    originals, cleans, i = [], [], 0
    for size in (1600, 777, 4800, 333) * 9:
        if i >= len(audio):
            break
        original, clean = d.push(audio[i : i + size])
        assert len(original) == len(clean)
        originals.append(original)
        cleans.append(clean)
        i += size
    original, clean = d.flush()
    originals.append(original)
    cleans.append(clean)
    joined = np.concatenate(originals)
    assert np.array_equal(joined, audio[: len(joined)]) and len(audio) - len(joined) < 256  # nothing lost or reordered
    assert len(np.concatenate(cleans)) == len(joined)
