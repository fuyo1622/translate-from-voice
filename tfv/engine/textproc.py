"""Clean up recognizer output: Traditional Chinese conversion and hallucination filtering."""

from __future__ import annotations

import logging
import re
import unicodedata

log = logging.getLogger(__name__)

# Phrases Whisper tends to invent over music or silence (learned from subtitled
# video). Compared after lower-casing and removing spaces and punctuation.
_HALLUCINATIONS = {
    "you",
    "thanksforwatching",
    "thankyouforwatching",
    "thankyousomuchforwatching",
    "pleasesubscribe",
    "pleasesubscribetomychannel",
    "ご視聴ありがとうございました",
    "ご覧いただきありがとうございます",
    "チャンネル登録お願いします",
    "チャンネル登録をお願いします",
    "謝謝觀看",
    "谢谢观看",
    "感謝觀看",
    "感谢观看",
    "請訂閱我的頻道",
    "请订阅我的频道",
}
_HALLUCINATION_PARTS = ("amara.org", "明鏡與點點", "明镜与点点", "字幕志願者", "字幕志愿者")
_MUSIC_ONLY = re.compile(r"^[\s♪♫♬#*\-~(\[（【]*(music|音楽|音乐|音樂|bgm)?[\s♪♫♬#*\-~)\]）】]*$", re.IGNORECASE)

# CJK punctuation, kana, ideographs and full-width forms
_CJK = "　-〿぀-ヿ㐀-䶿一-鿿豈-﫿＀-￯"
_SPACE_IN_CJK = re.compile(f"(?<=[{_CJK}])\\s+(?=[{_CJK}])")
# Japanese never spaces numbers from their counters ("午前 9 時", "50 円"); Chinese keeps
# the spaces some writers put around Latin text and digits.
_SPACE_AROUND_DIGITS_JA = re.compile(f"(?<=[{_CJK}])\\s+(?=[0-9])|(?<=[0-9])\\s+(?=[{_CJK}])")
# a lone "." (not part of "...") counts too
_ASCII_PUNCT_IN_CJK = re.compile(f"(?<=[{_CJK}])([,?!:;]|\\.(?!\\.))\\s*(?=[{_CJK}]|$)")
_FULLWIDTH = {
    "zh": {",": "，", "?": "？", "!": "！", ":": "：", ";": "；", ".": "。"},
    "ja": {",": "、", "?": "？", "!": "！", ":": "：", ";": "；", ".": "。"},
}


def _squash(text: str) -> str:
    return "".join(ch for ch in text.lower() if unicodedata.category(ch)[0] not in "PZS")


def is_hallucination(text: str) -> bool:
    lowered = text.lower()
    return (
        _squash(text) in _HALLUCINATIONS
        or any(part in lowered for part in _HALLUCINATION_PARTS)
        or bool(_MUSIC_ONLY.match(text))
    )


class TextPostProcessor:
    """Returns display-ready text for a recognized utterance, or None to drop it."""

    def __init__(self, chinese: str = "s2twp"):
        # s2twp: Simplified -> Traditional with Taiwan vocabulary (软件 -> 軟體)
        try:
            import opencc

            self._opencc = opencc.OpenCC(chinese)
        except Exception as exc:
            log.warning("OpenCC unavailable (%s); Chinese will not be converted to Traditional", exc)
            self._opencc = None

    def __call__(self, text: str, language: str | None) -> str | None:
        text = " ".join(text.split())
        if not text or is_hallucination(text):
            return None
        if language == "zh" and self._opencc is not None:
            text = self._opencc.convert(text)
        if language in _FULLWIDTH:
            # recognizers sometimes emit "今日は リアルタイム" or "很好,我們"
            table = _FULLWIDTH[language]
            text = _ASCII_PUNCT_IN_CJK.sub(lambda m: table[m.group(1)], text)
            text = _SPACE_IN_CJK.sub("", text)
            if language == "ja":
                text = _SPACE_AROUND_DIGITS_JA.sub("", text)
        return text
