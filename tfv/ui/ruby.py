"""A caption line with small readings above its kanji (furigana).

Qt's rich text has no ruby annotation, so this widget lays the text out itself:
centred, wrapped to the width like a word-wrapped QLabel, never splitting a
word from its reading.
"""

from __future__ import annotations

import unicodedata

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget


def _units(segments) -> list[tuple[str, str | None]]:
    """Pieces a line may break between: a word with its reading, one CJK character, or a Latin word."""
    units: list[tuple[str, str | None]] = []
    for text, reading in segments:
        if reading:
            units.append((text, reading))
            continue
        word = ""
        for ch in text:
            wide = unicodedata.east_asian_width(ch) in "WF"
            if wide or ch.isspace():
                if word:
                    units.append((word, None))
                    word = ""
                units.append((ch, None))
            else:
                word += ch
        if word:
            units.append((word, None))
    return units


class RubyLabel(QWidget):
    RUBY_SCALE = 0.5
    """Reading size relative to the text."""
    MIN_RUBY_PX = 11
    """Smaller kana are not readable on a subtitle bar."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._units: list[tuple[str, str | None]] = []
        self._font = QFont()
        self._ruby_font = QFont()
        self._color = QColor(235, 235, 235, 210)
        self._cache: tuple[int, list] | None = None
        policy = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    # -- content ---------------------------------------------------------------------

    def setSegments(self, segments) -> None:
        """(text, reading or None) pairs, e.g. from tfv.engine.furigana."""
        self._units = _units(segments)
        self._changed()

    def text(self) -> str:
        return "".join(text for text, _ in self._units)

    def readings(self) -> list[tuple[str, str]]:
        return [(text, reading) for text, reading in self._units if reading]

    def setFont(self, font: QFont) -> None:
        self._font = QFont(font)
        self._ruby_font = QFont(font)
        self._ruby_font.setBold(False)
        self._ruby_font.setPixelSize(max(self.MIN_RUBY_PX, round(font.pixelSize() * self.RUBY_SCALE)))
        self._changed()

    def setColor(self, color: QColor) -> None:
        self._color = QColor(color)
        self.update()

    # -- layout ------------------------------------------------------------------------

    def _changed(self) -> None:
        self._cache = None
        self.updateGeometry()
        self.update()

    def _line_height(self) -> float:
        return QFontMetricsF(self._ruby_font).height() + QFontMetricsF(self._font).height()

    def _lines(self, width: int) -> list[tuple[float, list[tuple[str, str | None, float]]]]:
        """Greedy wrap: [(line width, [(text, reading, unit width), ...]), ...]."""
        if self._cache and self._cache[0] == width:
            return self._cache[1]
        base, ruby = QFontMetricsF(self._font), QFontMetricsF(self._ruby_font)
        lines: list[tuple[float, list]] = []
        current: list = []
        used = 0.0
        for text, reading in self._units:
            w = max(base.horizontalAdvance(text), ruby.horizontalAdvance(reading) if reading else 0.0)
            if current and used + w > width and not text.isspace():
                lines.append((used, current))
                current, used = [], 0.0
            if not current and text.isspace():
                continue  # no leading space on a wrapped line
            current.append((text, reading, w))
            used += w
        if current:
            lines.append((used, current))
        self._cache = (width, lines)
        return lines

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return round(max(1, len(self._lines(width))) * self._line_height())

    def sizeHint(self) -> QSize:
        base, ruby = QFontMetricsF(self._font), QFontMetricsF(self._ruby_font)
        width = sum(max(base.horizontalAdvance(t), ruby.horizontalAdvance(r) if r else 0.0) for t, r in self._units)
        return QSize(round(width), round(self._line_height()))

    def minimumSizeHint(self) -> QSize:
        return QSize(0, round(self._line_height()))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setPen(self._color)
        ruby_h = QFontMetricsF(self._ruby_font).height()
        base_h = QFontMetricsF(self._font).height()
        y = 0.0
        for used, units in self._lines(self.width()):
            x = (self.width() - used) / 2
            for text, reading, w in units:
                if reading:
                    painter.setFont(self._ruby_font)
                    painter.drawText(QRectF(x, y, w, ruby_h), Qt.AlignmentFlag.AlignCenter, reading)
                painter.setFont(self._font)
                painter.drawText(QRectF(x, y + ruby_h, w, base_h), Qt.AlignmentFlag.AlignCenter, text)
                x += w
            y += ruby_h + base_h
