import pytest

from tfv.engine.furigana import Furigana, split_okurigana, to_hiragana


def test_to_hiragana():
    assert to_hiragana("チュウガク") == "ちゅうがく"
    assert to_hiragana("ラーメン") == "らーめん"


@pytest.mark.parametrize(
    ("surface", "reading", "expected"),
    [
        ("食べる", "たべる", [("食", "た"), ("べる", None)]),
        ("持ち帰り", "もちかえり", [("持", "も"), ("ち", None), ("帰", "かえ"), ("り", None)]),
        ("お茶", "おちゃ", [("お", None), ("茶", "ちゃ")]),
        ("今日", "きょう", [("今日", "きょう")]),
        ("取手市", "とりでし", [("取手市", "とりでし")]),
        ("大人しい", "おとなしい", [("大人", "おとな"), ("しい", None)]),
        ("聞き間違い", "ききまちがえ", [("聞き間違い", "ききまちがえ")]),  # doesn't fit: over the whole word
    ],
)
def test_reading_goes_over_kanji_only(surface, reading, expected):
    assert split_okurigana(surface, reading) == expected


@pytest.fixture(scope="module")
def furigana():
    return Furigana()


def test_sentence_readings(furigana):
    segments = furigana.segments("持ち帰りの弁当を食べました。")
    assert "".join(text for text, _ in segments) == "持ち帰りの弁当を食べました。"
    assert [(t, r) for t, r in segments if r] == [("持", "も"), ("帰", "かえ"), ("弁当", "べんとう"), ("食", "た")]
    assert furigana.segments("ロボットタコライス") == [("ロボットタコライス", None)]  # no kanji: no lookup
