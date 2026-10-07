"""Desktop app entry point: overlay + tray menu + settings + hotkeys."""

from __future__ import annotations

import logging
import logging.handlers
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from tfv import languages
from tfv.ui.controller import EngineController
from tfv.ui.hotkeys import GlobalHotkeys
from tfv.ui.overlay import UNLOCKED_HINT, OverlayWindow
from tfv.ui.prefs import Prefs, prefs_path

log = logging.getLogger("tfv")

_STATE_COLORS = {"running": "#2e9d57", "loading": "#d18b00", "paused": "#7a7a7a", "error": "#c0392b", "stopped": "#7a7a7a"}
_STATE_TEXT = {"running": "Listening", "loading": "Loading models…", "paused": "Paused", "error": "Error", "stopped": "Stopped"}


def make_icon(state: str) -> QIcon:
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(_STATE_COLORS.get(state, "#7a7a7a")))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(2, 2, 60, 60, 14, 14)
    font = QFont()
    font.setFamilies(["Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC"])
    font.setPixelSize(40)
    font.setBold(True)
    p.setFont(font)
    p.setPen(QColor("white"))
    p.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "譯")
    p.end()
    return QIcon(pix)


class TrayApp:
    def __init__(self, app: QApplication):
        self.app = app
        self.prefs = Prefs.load()
        self.overlay = OverlayWindow(self.prefs)
        self.overlay.geometry_changed.connect(self.prefs.save)
        self.controller = EngineController(self.prefs)
        self.controller.caption.connect(self.overlay.show_caption)
        self.controller.status.connect(self._on_status)
        self.controller.state_changed.connect(self._on_state)
        self.settings = None
        self.menu = self._build_menu()
        self.overlay.menu_requested.connect(self.menu.popup)
        self.overlay.action_requested.connect(self._on_overlay_action)
        self.tray = QSystemTrayIcon(make_icon("loading"))
        self.tray.setToolTip("tfv subtitles")
        self.tray.setContextMenu(self.menu)
        self.tray.activated.connect(self._on_tray_click)
        self.hotkeys = GlobalHotkeys()
        self.hotkeys.register(
            {
                "P": self.toggle_pause, "L": self.toggle_lock, "O": self.toggle_original, "H": self.toggle_overlay,
                "S": self.open_settings, "=": lambda: self.change_text_size(+2), "-": lambda: self.change_text_size(-2),
            }
        )  # fmt: skip
        self._sync_toolbar()

    def start(self) -> None:
        self.tray.show()
        self.overlay.show()
        self.overlay.show_status("Loading models… (the first start downloads them)", 3600)
        self.controller.start()
        if not self.prefs.tip_shown and QSystemTrayIcon.supportsMessages():
            # Windows 11 hides new tray icons under ^, so say where the controls are once
            self.tray.showMessage(
                "tfv is running",
                "The subtitle bar has buttons for text size, furigana, languages, settings and Quit. "
                "Click the small lock at its corner (or press Ctrl+Alt+L) to unlock it. The 譯 tray icon may be hidden under ^.",
                make_icon("running"),
                15000,
            )
            self.prefs.tip_shown = True
            self.prefs.save()

    def shutdown(self) -> None:
        self.hotkeys.unregister()
        self.controller.stop()
        self.prefs.save()
        self.tray.hide()

    # -- menu -----------------------------------------------------------------------

    def _build_menu(self) -> QMenu:
        menu = QMenu()
        self.state_action = menu.addAction("Loading…")
        self.state_action.setEnabled(False)
        menu.addSeparator()
        self.pause_action = _check(menu, "Pause", False, lambda on: self.controller.set_paused(on))
        self.source_group = self._language_menu(menu, "Spoken language", self.prefs.source_lang, include_off=False)
        self.target_group = self._language_menu(menu, "Translate into", self.prefs.target_lang, include_off=True)
        self.languages_popup = QMenu()  # the toolbar's language button opens just these two
        for sub in self.language_menus:
            self.languages_popup.addMenu(sub)
        menu.addSeparator()
        self.original_action = _check(menu, "Show original text", self.prefs.show_original, self._set_original)
        self.furigana_action = _check(menu, "Furigana (readings above kanji)", self.prefs.furigana, self._set_furigana)
        menu.addAction("Bigger text\tCtrl+Alt+=").triggered.connect(lambda: self.change_text_size(+2))
        menu.addAction("Smaller text\tCtrl+Alt+-").triggered.connect(lambda: self.change_text_size(-2))
        self.lock_action = _check(menu, "Lock overlay (click-through)", self.prefs.locked, self._set_locked)
        self.hide_action = _check(menu, "Hide overlay", False, lambda on: self.overlay.setVisible(not on))
        menu.addSeparator()
        menu.addAction("Settings…").triggered.connect(self.open_settings)
        menu.addAction("Quit").triggered.connect(self.app.quit)
        return menu

    def _language_menu(self, menu: QMenu, title: str, current: str | None, include_off: bool) -> QActionGroup:
        sub = menu.addMenu(title)
        self.language_menus = [*getattr(self, "language_menus", []), sub]
        group = QActionGroup(sub)
        options = ([(None, "Off (captions only)")] if include_off else []) + [(c, languages.LABELS[c]) for c in languages.SUPPORTED]
        for code, label in options:
            action = sub.addAction(label)
            action.setCheckable(True)
            action.setData(code)
            action.setChecked(code == current)
            group.addAction(action)
        group.triggered.connect(lambda _a: self._languages_from_menu())
        return group

    def _languages_from_menu(self) -> None:
        source = self.source_group.checkedAction().data()
        target = self.target_group.checkedAction().data()
        self.overlay.clear()
        self.overlay.show_status(f"Switching to {languages.LABELS[source]} → {languages.LABELS.get(target, 'captions only')}…")
        self.controller.set_languages(source, target)
        self.prefs.save()
        self._sync_toolbar()

    # -- toggles (menu and hotkeys) ----------------------------------------------------

    def toggle_pause(self) -> None:
        self.pause_action.setChecked(not self.pause_action.isChecked())  # triggers set_paused via toggled

    def toggle_lock(self) -> None:
        self.lock_action.setChecked(not self.lock_action.isChecked())

    def toggle_original(self) -> None:
        self.original_action.setChecked(not self.original_action.isChecked())

    def toggle_overlay(self) -> None:
        self.hide_action.setChecked(not self.hide_action.isChecked())

    def change_text_size(self, step: int) -> None:
        self.prefs.font_size = min(72, max(12, self.prefs.font_size + step))
        self.overlay.apply_style()  # also redraws the captions on screen
        self.overlay.show_status(f"Text size {self.prefs.font_size} px", 2)
        self.prefs.save()

    def _set_original(self, on: bool) -> None:
        self.prefs.show_original = on
        self.overlay.apply_style()
        self.prefs.save()
        self._sync_toolbar()

    def _set_furigana(self, on: bool) -> None:
        self.prefs.furigana = on
        self.overlay.apply_style()
        self.prefs.save()
        self._sync_toolbar()

    def _set_locked(self, on: bool) -> None:
        self.overlay.set_locked(on)
        self.prefs.save()

    def _on_overlay_action(self, name: str, where) -> None:
        """A button on the subtitle bar (its toolbar, or the unlock button)."""
        if name in ("smaller", "bigger"):
            self.change_text_size(-2 if name == "smaller" else +2)
        elif name == "furigana":
            self.furigana_action.setChecked(not self.prefs.furigana)
        elif name == "original":
            self.toggle_original()
        elif name == "languages":
            self.languages_popup.popup(where)
        elif name == "pause":
            self.toggle_pause()
        elif name in ("lock", "unlock"):
            self.toggle_lock()
        elif name == "settings":
            self.open_settings()
        elif name == "quit":
            self.app.quit()
        self._sync_toolbar()

    def _sync_toolbar(self) -> None:
        short = {"zh": "中", "en": "EN", "ja": "日"}
        self.overlay.toolbar.set_state(
            furigana=self.prefs.furigana,
            original=self.prefs.show_original,
            paused=self.controller.state == "paused",
            languages=f"{short[self.prefs.source_lang]}→{short.get(self.prefs.target_lang, '–')}",
        )

    def open_settings(self) -> None:
        from tfv.ui.settings_dialog import SettingsDialog

        if self.settings is None:
            self.settings = SettingsDialog(self.prefs, self.controller)
            self.settings.applied.connect(self._after_settings)
            self.settings.finished.connect(lambda _r: setattr(self, "settings", None))
        self.settings.show()
        self.settings.raise_()
        self.settings.activateWindow()

    def _after_settings(self) -> None:
        self.overlay.set_locked(self.prefs.locked)
        self.overlay.apply_style()
        for group, value in ((self.source_group, self.prefs.source_lang), (self.target_group, self.prefs.target_lang)):
            for action in group.actions():
                action.setChecked(action.data() == value)
        self.original_action.setChecked(self.prefs.show_original)
        self.furigana_action.setChecked(self.prefs.furigana)
        self.lock_action.setChecked(self.prefs.locked)
        self._sync_toolbar()

    # -- engine events ------------------------------------------------------------------

    def _on_status(self, message: str) -> None:
        # while loading, keep each message up until the next one replaces it
        self.overlay.show_status(message, 3600 if self.controller.state == "loading" else 6)
        self.tray.setToolTip(f"tfv — {message}"[:120])

    def _on_state(self, state: str) -> None:
        self.tray.setIcon(make_icon(state))
        text = _STATE_TEXT.get(state, state)
        if state == "error":
            text += f": {self.controller.error}"[:80]
            self.overlay.show_status(f"Could not start: {self.controller.error}. Open Settings (⚙ on the bar, or Ctrl+Alt+S).", 30)
        elif state == "running":
            hint = "" if self.prefs.locked else f"\n{UNLOCKED_HINT}"
            self.overlay.show_status(f"Listening. Captions appear when speech plays.{hint}", 5 if self.prefs.locked else 15)
        self.state_action.setText(text)
        self.pause_action.setChecked(state == "paused")
        self._sync_toolbar()

    def _on_tray_click(self, reason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.open_settings()


def _check(menu: QMenu, title: str, checked: bool, on_toggle) -> QAction:
    action = menu.addAction(title)
    action.setCheckable(True)
    action.setChecked(checked)
    action.toggled.connect(on_toggle)
    return action


def _setup_logging() -> None:
    log_file = prefs_path().with_name("tfv.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    if sys.stderr is not None:  # pythonw has no console
        root.addHandler(logging.StreamHandler())
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "faster_whisper", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    app = QApplication(sys.argv[:1] + list(argv or []))
    app.setApplicationName("tfv")
    app.setQuitOnLastWindowClosed(False)  # closing the settings window must not quit
    if not QSystemTrayIcon.isSystemTrayAvailable():
        log.warning("No system tray; use the hotkeys, or quit with Ctrl+C in the terminal")
    tray_app = TrayApp(app)
    app.aboutToQuit.connect(tray_app.shutdown)
    tray_app.start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
