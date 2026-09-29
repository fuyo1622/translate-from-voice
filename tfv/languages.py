"""Languages the app handles.

Codes are the ISO 639-1 codes Whisper uses. "zh" is always displayed as
Traditional Chinese (Taiwan), whatever script an engine produced.
The spoken language is chosen by the user; there is no auto-detection yet.
"""

SUPPORTED = ("zh", "en", "ja")

NAMES = {"zh": "Traditional Chinese (Taiwan)", "en": "English", "ja": "Japanese"}
"""English names, as used in LLM prompts."""

LABELS = {"zh": "中文", "en": "English", "ja": "日本語"}
"""Names shown in the UI."""
