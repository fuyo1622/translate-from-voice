"""The subtitle overlay: a frameless, always-on-top window near the bottom of the screen.

Locked, it lets clicks pass through to the video or window below; unlocked, it
can be dragged, resized by its side edges, and right-clicked for the menu.
Captions grow upward from a fixed bottom edge, update while someone is still
talking (live text), and fade out after a few seconds without speech.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QPoint, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QPainter, QPen
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from tfv.engine.pipeline import Caption
from tfv.ui.controls import ToolBar, UnlockButton
from tfv.ui.prefs import Prefs
from tfv.ui.ruby import RubyLabel

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


UNLOCKED_HINT = "Drag to move · drag a side edge to resize · buttons above, or right-click for the menu · 🔒 locks it"

_ORIGINAL_RGBA = (240, 240, 240, 235)
_LIVE_RGBA = (205, 205, 205, 190)
"""Text still being spoken: dimmer, since it may change."""


class _Row:
    def __init__(self, parent: QWidget):
        self.original = QLabel(parent)
        self.ruby = RubyLabel(parent)  # the original with furigana, shown instead when there are readings
        self.ruby.hide()
        self.translation = QLabel(parent)
        for label in (self.original, self.translation):
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setWordWrap(True)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.translation.setStyleSheet("color: white;")
        self.updated = time.monotonic()
        self.caption: Caption | None = None
        """The latest state shown, so a style change can redraw it."""

    def show_original(self, text: str, ruby: list | None, font: QFont, live: bool, visible: bool) -> None:
        rgba = _LIVE_RGBA if live else _ORIGINAL_RGBA
        if ruby:
            self.ruby.setSegments(ruby)
            self.ruby.setFont(font)
            self.ruby.setColor(QColor(*rgba))
        else:
            self.original.setText(text)
            self.original.setFont(font)
            self.original.setStyleSheet("color: rgba({}, {}, {}, {});".format(*rgba))
        self.ruby.setVisible(visible and bool(ruby))
        self.original.setVisible(visible and not ruby)

    def widgets(self) -> list[QWidget]:
        return [self.original, self.ruby, self.translation]

    def delete(self) -> None:
        for w in self.widgets():
            w.deleteLater()


class OverlayWindow(QWidget):
    geometry_changed = Signal()
    menu_requested = Signal(QPoint)
    """Right-click while unlocked: the app shows its menu there (the tray icon can be hard to find)."""
    action_requested = Signal(str, QPoint)
    """A toolbar button or the unlock button was clicked: its name, and where a menu should open."""

    EDGE = 12
    """Pixels at the left and right edge that resize instead of moving."""
    MIN_WIDTH = 240

    def __init__(self, prefs: Prefs):
        super().__init__(None)
        self.prefs = prefs
        self.setWindowTitle("tfv subtitles")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMouseTracking(True)  # resize cursors at the edges
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(18, 10, 18, 12)
        self._layout.setSpacing(4)
        self.toolbar = ToolBar(self)  # shown while unlocked
        self.toolbar.triggered.connect(self.action_requested)
        self._layout.addWidget(self.toolbar)
        self._unlock = UnlockButton()  # shown at the corner while locked
        self._unlock.clicked.connect(lambda: self.action_requested.emit("unlock", QPoint()))
        self._status = QLabel(self)
        self._status.setTextFormat(Qt.TextFormat.PlainText)
        self._status.setWordWrap(True)
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status.setStyleSheet("color: rgba(255, 220, 120, 230);")
        self._layout.addWidget(self._status)
        self._rows: dict[int, _Row] = {}
        self._status_until = 0.0
        self._press: tuple[str, QPoint, QPoint, int] | None = None
        """While the left button is down: (what it does, where it was pressed, top-left then, width then)."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._expire)
        self._timer.start(500)
        self._restore_geometry()
        self._apply_flags()
        self.apply_style()

    # -- public ------------------------------------------------------------------

    def show_caption(self, caption: Caption) -> None:
        if caption.stage == "dropped":  # the live text was noise after all
            if (gone := self._rows.pop(caption.id, None)) is not None:
                gone.delete()
                self._relayout()
            return
        translating = caption.target_lang is not None
        row = self._rows.get(caption.id)
        if row is None:
            if self.prefs.show_together and translating and caption.stage == "text":
                return  # shown together with its translation (or alone if translation fails)
            row = _Row(self)
            self._rows[caption.id] = row
            for w in row.widgets():
                self._layout.addWidget(w)
            while len(self._rows) > max(1, self.prefs.lines):
                oldest = next(iter(self._rows))
                self._rows.pop(oldest).delete()
        self._render(row, caption)
        row.updated = time.monotonic()
        self._relayout()

    def _render(self, row: _Row, caption: Caption) -> None:
        row.caption = caption
        translating = caption.target_lang is not None
        live = caption.stage == "partial"
        ruby = caption.ruby if self.prefs.furigana else None
        size = self.prefs.font_size
        if translating:  # smaller beside its translation, but not so small that its readings can't be read
            size = round(size * (0.85 if ruby else 0.7))
        row.show_original(
            caption.text,
            ruby,
            font_for(caption.source_lang, size),
            live,
            visible=self.prefs.show_original or not translating or live or caption.stage == "untranslated",
        )
        if live:
            row.translation.setVisible(False)  # translated once the phrase is complete
        elif translating and caption.stage != "untranslated":
            row.translation.setText(caption.translation or "…")
            row.translation.setFont(font_for(caption.target_lang, self.prefs.font_size, bold=True))
            row.translation.setVisible(True)
        else:
            row.translation.setVisible(False)

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
        self.show_status("" if locked else UNLOCKED_HINT, 10)

    def apply_style(self) -> None:
        """Re-read font size, line count and visibility options from prefs (also for captions on screen)."""
        while len(self._rows) > max(1, self.prefs.lines):
            self._rows.pop(next(iter(self._rows))).delete()
        for row in self._rows.values():
            self._render(row, row.caption)
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
        self.unsetCursor()
        self.toolbar.setVisible(not self.prefs.locked)
        if was_visible:
            self.show()
        self._relayout()

    def min_width(self) -> int:
        """Narrowest the bar may be dragged to: the toolbar must fit."""
        margins = self._layout.contentsMargins()
        return max(self.MIN_WIDTH, self.toolbar.sizeHint().width() + margins.left() + margins.right())

    def _sync_unlock_button(self) -> None:
        show = self.prefs.locked and self.prefs.unlock_button and self.isVisible()
        if show:
            g = self.geometry()
            self._unlock.move(g.right() - self._unlock.width() - 4, g.top() + 4)
        self._unlock.setVisible(show)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._sync_unlock_button()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._unlock.hide()

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
        width = self._width if self.prefs.locked else max(self._width, self.min_width())
        # -1 when nothing visible wraps (e.g. only the toolbar): then the minimum size decides
        height = max(40, self._layout.totalHeightForWidth(width), self._layout.totalMinimumSize().height())
        self.setGeometry(QRect(self._x, self._bottom - height, width, height))
        self._sync_unlock_button()
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
            # handles on the side edges, which resize
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 200))
            handle = min(28, self.height() - 16)
            top = (self.height() - handle) // 2
            for x in (4, self.width() - 8):
                painter.drawRoundedRect(x, top, 4, handle, 2, 2)

    def _expire(self) -> None:
        now = time.monotonic()
        stale = [cid for cid, row in self._rows.items() if now - row.updated > self.prefs.clear_after_s]
        for cid in stale:
            self._rows.pop(cid).delete()
        if stale or (self._status.isVisible() and now >= self._status_until):
            self._relayout()

    # -- moving, resizing and the menu when unlocked ------------------------------------

    def _zone(self, x: float) -> str:
        if x < self.EDGE:
            return "left"
        if x >= self.width() - self.EDGE:
            return "right"
        return "move"

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self.prefs.locked:
            zone = self._zone(event.position().x())
            self._press = (zone, event.globalPosition().toPoint(), self.geometry().topLeft(), self._width)

    def mouseMoveEvent(self, event) -> None:
        if self._press is None:
            if not self.prefs.locked:
                move = self._zone(event.position().x()) == "move"
                self.setCursor(Qt.CursorShape.SizeAllCursor if move else Qt.CursorShape.SizeHorCursor)
            return
        zone, start, top_left, width = self._press
        delta = event.globalPosition().toPoint() - start
        if zone == "move":
            self._x, self._bottom = top_left.x() + delta.x(), top_left.y() + delta.y() + self.height()
        elif zone == "right":
            self._width = max(self.min_width(), width + delta.x())
        else:  # the left edge moves; the right edge stays put
            self._width = max(self.min_width(), width - delta.x())
            self._x = top_left.x() + width - self._width
        self._relayout()

    def mouseReleaseEvent(self, event) -> None:
        if self._press is not None:
            self._press = None
            self._save_geometry()

    def contextMenuEvent(self, event) -> None:
        self.menu_requested.emit(event.globalPos())

    def _save_geometry(self) -> None:
        self.prefs.geometry = [self._x, self._bottom, self._width]
        self.geometry_changed.emit()
