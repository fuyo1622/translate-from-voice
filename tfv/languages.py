"""Languages the app handles.

Codes are the ISO 639-1 codes Whisper uses. "zh" is always displayed as
Traditional Chinese (Taiwan), whatever script the recognizer produced.
The spoken language is chosen by the user; there is no auto-detection yet.
"""

SUPPORTED = ("zh", "en", "ja")
