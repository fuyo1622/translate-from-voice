"""Controls on the subtitle bar itself, so the common things don't need the tray icon or hotkeys:
a toolbar while the bar is unlocked, and a small unlock button while it is locked."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QToolButton, QWidget

# name, label, tooltip, checkable: the things changed most while watching, left to right
TOOLBAR = [
    ("smaller", "A−", "Smaller text (Ctrl+Alt+-)", False),
    ("bigger", "A+", "Bigger text (Ctrl+Alt+=)", False),
    ("furigana", "ふりがな", "Readings above kanji", True),
    ("original", "原文", "Show the original text (Ctrl+Alt+O)", True),
    ("languages", "日→中 ▾", "Spoken language and translation", False),
    ("pause", "❚❚", "Pause (Ctrl+Alt+P)", False),
    ("lock", "🔒", "Lock: clicks go through to the video (Ctrl+Alt+L)", False),
    ("settings", "⚙", "Settings (Ctrl+Alt+S)", False),
    ("quit", "✕", "Quit tfv", False),
]

_STYLE = """
QToolButton { color: white; background: rgba(255, 255, 255, 28); border: 1px solid rgba(255, 255, 255, 70);
              border-radius: 6px; padding: 2px 8px; font-size: 13px; }
QToolButton:hover { background: rgba(255, 255, 255, 75); }
QToolButton:checked { background: rgba(90, 160, 255, 120); border-color: rgba(140, 190, 255, 200); }
"""


class ToolBar(QWidget):
    triggered = Signal(str, QPoint)
    """An action's name, and where a menu for it should open (screen coordinates)."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setStyleSheet(_STYLE)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 2)
        layout.setSpacing(4)
        layout.addStretch()
        self.buttons: dict[str, QToolButton] = {}
        for name, label, tip, checkable in TOOLBAR:
            button = QToolButton(self)
            button.setText(label)
            button.setToolTip(tip)
            button.setCheckable(checkable)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda _=False, n=name, b=button: self.triggered.emit(n, b.mapToGlobal(QPoint(0, b.height()))))
            layout.addWidget(button)
            self.buttons[name] = button
        layout.addStretch()

    def set_state(self, *, furigana: bool, original: bool, paused: bool, languages: str) -> None:
        self.buttons["furigana"].setChecked(furigana)
        self.buttons["original"].setChecked(original)
        self.buttons["pause"].setText("▶" if paused else "❚❚")
        self.buttons["pause"].setToolTip(f"{'Resume' if paused else 'Pause'} (Ctrl+Alt+P)")
        self.buttons["languages"].setText(f"{languages} ▾")


class UnlockButton(QWidget):
    """A small lock at the corner of the locked bar: one click unlocks it (and shows the toolbar)."""

    clicked = Signal()
    SIZE = 26

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setToolTip("Unlock the subtitles to move them or change settings (Ctrl+Alt+L)")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hover = False

    def enterEvent(self, event) -> None:
        self._hover = True
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover = False
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(0, 0, 0, 170 if self._hover else 70))  # faint until pointed at
        p.drawEllipse(QRectF(1, 1, self.SIZE - 2, self.SIZE - 2))
        ink = QColor(255, 255, 255, 240 if self._hover else 140)
        p.setPen(QPen(ink, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawArc(QRectF(9, 6, 8, 9), 0, 180 * 16)  # shackle
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(ink)
        p.drawRoundedRect(QRectF(7.5, 11, 11, 8.5), 1.5, 1.5)  # body
