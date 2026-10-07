"""The pipeline's live text, with a paced fake source and fake models (no audio devices or downloads)."""

import threading
import time

import numpy as np

from tfv import SAMPLE_RATE
from tfv.capture.base import AudioSource
from tfv.engine.asr.base import ASRBackend, ASRResult
from tfv.engine.mt.base import Translator
from tfv.engine.pipeline import Pipeline
from tfv.engine.segmenter import Segmenter, SegmenterConfig


def fake_vad(window: np.ndarray) -> float:
    return 0.9 if np.abs(window).mean() > 0.5 else 0.05


def speech(silence_before: float, talk: float, silence_after: float) -> np.ndarray:
    parts = [np.zeros(int(silence_before * SAMPLE_RATE)), np.ones(int(talk * SAMPLE_RATE)), np.zeros(int(silence_after * SAMPLE_RATE))]
    return np.concatenate(parts).astype(np.float32)


class PacedSource(AudioSource):
    description = "test audio"

    def __init__(self, audio: np.ndarray, realtime: bool = True):
        self.audio, self.realtime = audio, realtime

    def start(self, on_audio, on_end) -> None:
        def run():
            block = SAMPLE_RATE // 10
            for i in range(0, len(self.audio), block):
                on_audio(self.audio[i : i + block])
                if self.realtime:
                    time.sleep(0.1)
            on_end()

        threading.Thread(target=run, daemon=True).start()

    def stop(self) -> None:
        pass


class GrowingASR(ASRBackend):
    """One あ per half second of speech heard, so live text grows as the utterance does."""

    def __init__(self):
        self.mute = False

    def transcribe(self, audio: np.ndarray) -> ASRResult:
        heard = int(np.sum(np.abs(audio) > 0.5) / (SAMPLE_RATE / 2))
        return ASRResult("" if self.mute else "あ" * max(1, heard), "ja", 0.001)


class TaggingTranslator(Translator):
    def translate(self, text, source, target, context=()):
        return f"譯{text}"


def run(audio: np.ndarray, partials: bool = True, realtime: bool = True, asr=None, on_caption=None, config=None) -> list:
    captions = []
    asr = asr or GrowingASR()

    def collect(caption):
        captions.append((caption.id, caption.stage, caption.text, caption.translation))
        if on_caption:
            on_caption(caption, asr)

    pipeline = Pipeline(
        PacedSource(audio, realtime), Segmenter(fake_vad, config), asr, collect, source_lang="ja", target_lang="zh",
        translator=TaggingTranslator(), partials=partials, partial_interval=0.3,
    )  # fmt: skip
    pipeline.start()
    assert pipeline.wait(20)
    return captions


def test_live_text_then_the_line_then_its_translation_in_one_caption():
    captions = run(speech(0.3, 2.0, 1.0))
    ids = {cid for cid, *_ in captions}
    stages = [stage for _, stage, *_ in captions]
    assert len(ids) == 1  # one row on screen, updated in place
    assert stages[0] == "partial" and stages[-2:] == ["text", "translated"] and set(stages) == {"partial", "text", "translated"}
    live = [text for _, stage, text, _ in captions if stage == "partial"]
    assert live == sorted(live, key=len)  # grows while the speaker talks
    assert captions[-1][3] == "譯ああああ"


def test_live_text_that_turns_out_to_be_noise_is_withdrawn():
    def silence_after_first(caption, asr):
        if caption.stage == "partial":
            asr.mute = True  # the full utterance recognizes as nothing

    captions = run(speech(0.3, 2.0, 1.0), on_caption=silence_after_first)
    first_id = captions[0][0]
    assert captions[0][1] == "partial" and captions[-1][:2] == (first_id, "dropped")
    assert "text" not in [stage for _, stage, *_ in captions]


def test_no_live_text_when_off_or_for_files():
    assert {stage for _, stage, *_ in run(speech(0.3, 2.0, 1.0), partials=False)} == {"text", "translated"}
    assert {stage for _, stage, *_ in run(speech(0.3, 2.0, 1.0), realtime=False)} == {"text", "translated"}


def test_live_text_of_a_blip_that_was_not_speech_is_withdrawn():
    # 0.5 s of sound shows live text, but is shorter than the 0.6 s an utterance needs
    captions = run(speech(0.3, 0.5, 1.5), config=SegmenterConfig(min_speech_ms=600))
    stages = [stage for _, stage, *_ in captions]
    assert stages[0] == "partial" and stages[-1] == "dropped" and "text" not in stages
    assert len({cid for cid, *_ in captions}) == 1


def test_pausing_withdraws_live_text():
    from tfv.engine.pipeline import Caption  # noqa: F401  (stage names live there)

    captions = []
    pipeline = Pipeline(
        PacedSource(speech(0.3, 3.0, 1.0)), Segmenter(fake_vad), GrowingASR(),
        lambda c: captions.append((c.id, c.stage)), source_lang="ja", target_lang="zh",
        translator=TaggingTranslator(), partials=True, partial_interval=0.3,
    )  # fmt: skip
    pipeline.start()
    deadline = time.monotonic() + 5
    while not any(stage == "partial" for _, stage in captions) and time.monotonic() < deadline:
        time.sleep(0.05)
    pipeline.pause()  # mid-sentence
    assert pipeline.wait(20)
    stages = [stage for _, stage in captions]
    assert "partial" in stages and stages[-1] == "dropped" and "text" not in stages
