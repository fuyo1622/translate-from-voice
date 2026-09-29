"""The subtitle overlay: a frameless, always-on-top window near the bottom of the screen.

Locked, it lets clicks pass through to the video or window below; unlocked, it
can be dragged and resized. Captions grow upward from a fixed bottom edge and
fade out after a few seconds without speech.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QLabel, QSizeGrip, QVBoxLayout, QWidget

from tfv.engine.pipeline import Caption
from tfv.ui.prefs import Prefs

# Han characters look different in Chinese and Japanese fonts, so pick per language.
_FONTS = {
    "zh": ["Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC", "Segoe UI"],
    "ja": ["Yu Gothic UI", "Meiryo UI", "Hiragino Sans", "Noto Sans CJK JP", "Segoe UI"],
    "en": ["Segoe UI", "Helvetica Neue", "Noto Sans", "Arial"],
}


def font_for(lang: str | None, size: int, bold: bool = False) -> QFont:
    font = QFont()
    font.setFamilies(_FONTS.get(lang or "en", _FONTS["en"]))
    font.setPixelSize(max(8, size))
    font.setBold(bold)
    return font


class _Row:
    def __init__(self, parent: QWidget):
        self.original = QLabel(parent)
        self.translation = QLabel(parent)
        for label in (self.original, self.translation):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.original.setStyleSheet("color: rgba(235, 235, 235, 210);")
        self.translation.setStyleSheet("color: white;")
        self.updated = time.monotonic()

    def widgets(self) -> list[QLabel]:
        return [self.original, self.translation]

    def delete(self) -> None:
        for w in self.widgets():
            w.deleteLater()


class OverlayWindow(QWidget):
    geometry_changed = Signal()

    def __init__(self, prefs: Prefs):
        super().__init__(None)
        self.prefs = prefs
        self.setWindowTitle("tfv subtitles")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(18, 10, 18, 12)
        self._layout.setSpacing(4)
        self._status = QLabel(self)
        self._status.setTextFormat(Qt.TextFormat.PlainText)
        self._status.setWordWrap(True)
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color: rgba(255, 220, 120, 230);")
        self._layout.addWidget(self._status)
        self._rows: dict[int, _Row] = {}
        self._status_until = 0.0
        self._grip = QSizeGrip(self)
        self._drag_offset: QPoint | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._expire)
        self._timer.start(500)
        self._apply_flags()
        self._restore_geometry()
        self.apply_style()

    # -- public ------------------------------------------------------------------

    def show_caption(self, caption: Caption) -> None:
        row = self._rows.get(caption.id)
        if row is None:
            row = _Row(self)
            self._rows[caption.id] = row
            for w in row.widgets():
                self._layout.addWidget(w)
            while len(self._rows) > max(1, self.prefs.lines):
                oldest = next(iter(self._rows))
                self._rows.pop(oldest).delete()
        translating = caption.target_lang is not None
        row.original.setText(caption.text)
        row.original.setFont(font_for(caption.source_lang, self.prefs.font_size * (7 if translating else 10) // 10))
        row.original.setVisible(self.prefs.show_original or not translating or caption.stage == "untranslated")
        if translating and caption.stage != "untranslated":
            row.translation.setText(caption.translation or "…")
            row.translation.setFont(font_for(caption.target_lang, self.prefs.font_size, bold=True))
            row.translation.setVisible(True)
        else:
            row.translation.setVisible(False)
        row.updated = time.monotonic()
        self._relayout()

    def show_status(self, text: str, seconds: float = 6.0) -> None:
        """A small notice above the captions (loading, errors); empty text hides it."""
        self._status.setText(text)
        self._status.setFont(font_for("en", max(12, self.prefs.font_size * 6 // 10)))
        self._status_until = time.monotonic() + seconds if text else 0.0
        self._relayout()

    def clear(self) -> None:
        for row in self._rows.values():
            row.delete()
        self._rows.clear()
        self._relayout()

    def set_locked(self, locked: bool) -> None:
        self.prefs.locked = locked
        self._apply_flags()
        self.show_status("" if locked else "Drag to move, drag the corner to resize. Lock again from the tray menu or Ctrl+Alt+L.", 8)

    def apply_style(self) -> None:
        """Re-read font size, line count and visibility options from prefs."""
        while len(self._rows) > max(1, self.prefs.lines):
            self._rows.pop(next(iter(self._rows))).delete()
        self._relayout()
        self.update()

    # -- layout and painting ---------------------------------------------------------

    def _apply_flags(self) -> None:
        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        if self.prefs.locked:
            flags |= Qt.WindowType.WindowTransparentForInput
        was_visible = self.isVisible()
        self.setWindowFlags(flags)
        self._grip.setVisible(not self.prefs.locked)
        if was_visible:
            self.show()
        self.update()

    def _restore_geometry(self) -> None:
        screen = QGuiApplication.primaryScreen().availableGeometry()
        geo = self.prefs.geometry
        if len(geo) == 3 and screen.adjusted(-200, -200, 200, 200).contains(QPoint(geo[0] + 20, geo[1] - 10)):
            self._x, self._bottom, self._width = geo
        else:
            self._width = int(screen.width() * 0.6)
            self._x = screen.x() + (screen.width() - self._width) // 2
            self._bottom = screen.y() + screen.height() - 60
        self._relayout()

    def _relayout(self) -> None:
        self._status.setVisible(bool(self._status.text()) and time.monotonic() < self._status_until)
        self._layout.activate()
        height = max(40, self._layout.totalHeightForWidth(self._width))
        self.setGeometry(QRect(self._x, self._bottom - height, self._width, height))
        self._grip.move(self.width() - self._grip.width(), self.height() - self._grip.height())
        self.update()

    def _has_content(self) -> bool:
        return bool(self._rows) or self._status.isVisible()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._has_content() or not self.prefs.locked:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, int(255 * self.prefs.opacity)))
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 12, 12)
        if not self.prefs.locked:
            pen = QPen(QColor(255, 255, 255, 160), 1.5, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 12, 12)

    def _expire(self) -> None:
        now = time.monotonic()
        stale = [cid for cid, row in self._rows.items() if now - row.updated > self.prefs.clear_after_s]
        for cid in stale:
            self._rows.pop(cid).delete()
        if stale or (self._status.isVisible() and now >= self._status_until):
            self._relayout()

    # -- moving and resizing when unlocked --------------------------------------------

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self.prefs.locked:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_offset is not None:
            top_left = event.globalPosition().toPoint() - self._drag_offset
            self._x, self._bottom = top_left.x(), top_left.y() + self.height()
            self.move(top_left)

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_offset is not None:
            self._drag_offset = None
            self._save_geometry()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self.prefs.locked and self.width() != self._width and self._drag_offset is None:
            # the size grip changed the width; keep the bottom edge where it was
            self._width = max(240, self.width())
            self._save_geometry()
            QTimer.singleShot(0, self._relayout)

    def _save_geometry(self) -> None:
        self.prefs.geometry = [self._x, self._bottom, self._width]
        self.geometry_changed.emit()
