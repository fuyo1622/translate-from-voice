import pytest

from tfv.engine.textproc import TextPostProcessor, is_hallucination


@pytest.mark.parametrize(
    "text",
    ["Thanks for watching!", "  you ", "ご視聴ありがとうございました", "字幕由Amara.org社区提供", "♪♪", "[Music]", "(音楽)"],
)
def test_hallucinations_are_detected(text):
    assert is_hallucination(text)


@pytest.mark.parametrize("text", ["Thank you for the music.", "今日は音楽の話です。", "You know what I mean?"])
def test_real_speech_is_kept(text):
    assert not is_hallucination(text)


def test_cjk_spacing_and_punctuation():
    post = TextPostProcessor()
    assert post("今日は リアルタイム 翻訳 の 仕組み", "ja") == "今日はリアルタイム翻訳の仕組み"
    assert post("今天天氣很好,我們走吧", "zh") == "今天天氣很好，我們走吧"
    assert post("はい,そうです!", "ja") == "はい、そうです！"
    assert post("我用 Python 寫程式, 很好", "zh") == "我用 Python 寫程式，很好"  # spaces around Latin kept
    assert post("It costs 1,000 yen, right?", "en") == "It costs 1,000 yen, right?"
    assert post("即時翻譯的方法.", "zh") == "即時翻譯的方法。"
    assert post("價格是3.5元,好嗎...", "zh") == "價格是3.5元，好嗎..."
    # seen from SenseVoice and Qwen: Japanese numbers spaced from their counters
    assert post("場合は、50 円のパン", "ja") == "場合は、50円のパン"
    assert post("営業時間は午前 9 時から午後 5 時までです。", "ja") == "営業時間は午前9時から午後5時までです。"
    assert post("Python 3 を使う", "ja") == "Python 3を使う"
    assert post("我有 3 個", "zh") == "我有 3 個"  # Chinese spacing is left alone


def test_chinese_becomes_traditional_taiwan():
    post = TextPostProcessor()
    assert post("这个软件的内存不够", "zh") == "這個軟體的記憶體不夠"
    assert post("这个软件", "ja") == "这个软件"  # only Chinese is converted
    assert post("   ", "en") is None
