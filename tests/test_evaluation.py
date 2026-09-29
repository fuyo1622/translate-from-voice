import numpy as np
import pytest

from tfv.evaluation import add_noise, chrf, edit_distance, error_rate, tokens_for_scoring


def test_edit_distance():
    assert edit_distance(list("kitten"), list("sitting")) == 3
    assert edit_distance([], list("abc")) == 3


def test_tokens_ignore_case_punctuation_and_script():
    assert tokens_for_scoring("Hello, World!", "en") == ["hello", "world"]
    assert tokens_for_scoring("這個軟體。", "zh") == list("这个软件")  # compared in mainland Simplified
    assert tokens_for_scoring("今日は、晴れ。", "ja") == list("今日は晴れ")


def test_error_rate():
    assert error_rate(["這個程式"], ["这个程式"], "zh") == 0.0
    assert error_rate(["the cat sat"], ["the cat sat down"], "en") == pytest.approx(0.25)


def test_chrf():
    assert chrf(["今天天氣很好"], ["今天天氣很好"]) == pytest.approx(100.0)
    assert chrf(["completely different"], ["今天天氣很好"]) == 0.0
    partial = chrf(["the weather is nice today"], ["the weather is very nice today"])
    assert 60 < partial < 100


def test_noise_level():
    rng = np.random.default_rng(1)
    clean = np.sin(np.linspace(0, 2000, 16000)).astype(np.float32)
    noisy = add_noise(clean, 10.0, rng)
    snr = 10 * np.log10(np.mean(clean**2) / np.mean((noisy - clean) ** 2))
    assert snr == pytest.approx(10.0, abs=0.5)
