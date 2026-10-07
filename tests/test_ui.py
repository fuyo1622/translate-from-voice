"""Overlay and settings window, rendered offscreen (no visible windows, no engine)."""

import os
import tempfile
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, Qt  # noqa: E402
from PySide6.QtGui import QContextMenuEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from tfv.engine.pipeline import Caption  # noqa: E402
from tfv.hardware import Hardware  # noqa: E402
from tfv.ui.overlay import OverlayWindow  # noqa: E402
from tfv.ui.prefs import Prefs  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def config_dir(monkeypatch):
    with tempfile.TemporaryDirectory() as path:
        monkeypatch.setenv("TFV_CONFIG_DIR", path)
        yield path


def caption(id, stage="translated", text="こんにちは。", translation="你好。", target="zh"):
    return Caption(id, 0.0, 1.0, False, "ja", text, target_lang=target, translation=translation, stage=stage)


def test_overlay_shows_original_with_its_translation(qt_app):
    ov = OverlayWindow(Prefs())
    ov.show_caption(caption(1, stage="text", translation=None))
    assert not ov._rows  # held back until the translation arrives
    ov.show_caption(caption(1))
    row = ov._rows[1]
    assert (row.original.text(), row.translation.text()) == ("こんにちは。", "你好。")
    assert row.original.isVisibleTo(ov) and row.translation.isVisibleTo(ov)
    assert row.translation.font().pixelSize() == 22 and row.original.font().pixelSize() == 15


def test_overlay_options(qt_app):
    prefs = Prefs(show_original=False, lines=2)
    ov = OverlayWindow(prefs)
    ov.show_caption(caption(1))
    assert not ov._rows[1].original.isVisibleTo(ov)
    ov.show_caption(caption(2, stage="untranslated", translation=None))
    assert ov._rows[2].original.isVisibleTo(ov) and not ov._rows[2].translation.isVisibleTo(ov)  # nothing else to show
    ov.show_caption(caption(3, target=None, translation=None, stage="text"))  # captions only
    assert list(ov._rows) == [2, 3]  # at most `lines` captions
    assert ov._rows[3].original.font().pixelSize() == 22


def test_overlay_clears_after_silence(qt_app):
    ov = OverlayWindow(Prefs(clear_after_s=8))
    ov.show_caption(caption(1))
    ov._expire()
    assert 1 in ov._rows
    ov._rows[1].updated = time.monotonic() - 9
    ov._expire()
    assert not ov._rows


def test_overlay_lock_makes_it_click_through(qt_app):
    ov = OverlayWindow(Prefs(locked=False))
    ov.show()
    assert not ov.windowFlags() & Qt.WindowType.WindowTransparentForInput
    ov.set_locked(True)
    assert ov.windowFlags() & Qt.WindowType.WindowTransparentForInput
    assert ov.windowFlags() & Qt.WindowType.WindowStaysOnTopHint and ov.isVisible()
    ov.close()


def drag(widget, start: QPoint, end: QPoint) -> None:
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(widget, end)
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)


def test_overlay_resizes_by_its_side_edges(qt_app):
    prefs = Prefs(locked=False, geometry=[100, 500, 1000])
    ov = OverlayWindow(prefs)
    ov.show()
    h = ov.height()
    drag(ov, QPoint(ov.width() - 3, h // 2), QPoint(ov.width() - 3 + 50, h // 2))  # right edge
    assert (ov.x(), ov.width()) == (100, 1050)
    drag(ov, QPoint(3, h // 2), QPoint(3 - 80, h // 2))  # left edge: the right edge stays put
    assert (ov.x(), ov.width()) == (20, 1130)
    assert prefs.geometry == [20, 500, 1130]  # saved for the next start
    drag(ov, QPoint(ov.width() - 3, h // 2), QPoint(-500, h // 2))
    assert ov.width() == ov.min_width()
    drag(ov, QPoint(ov.width() // 2, h // 2), QPoint(ov.width() // 2 + 30, h // 2 - 40))  # the middle moves it
    assert (ov.x(), ov.geometry().bottom() + 1, ov.width()) == (50, 460, ov.min_width())
    ov.close()


def test_overlay_right_click_asks_for_the_menu(qt_app):
    ov = OverlayWindow(Prefs(locked=False))
    ov.show()
    asked = []
    ov.menu_requested.connect(asked.append)
    QApplication.sendEvent(ov, QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(10, 10), QPoint(110, 510)))
    assert asked == [QPoint(110, 510)]
    ov.close()


def test_overlay_shows_live_text_until_the_phrase_is_done(qt_app):
    ov = OverlayWindow(Prefs())
    ov.show_caption(caption(1, stage="partial", text="こんにち", translation=None))
    row = ov._rows[1]
    assert row.original.text() == "こんにち" and row.original.isVisibleTo(ov) and not row.translation.isVisibleTo(ov)
    ov.show_caption(caption(1, stage="text", translation=None))  # the phrase ended: original fixed, translation coming
    assert row.original.text() == "こんにちは。" and row.translation.text() == "…"
    ov.show_caption(caption(1))
    assert row.translation.text() == "你好。" and list(ov._rows) == [1]


class FakeController:
    session = None
    state = "running"
    error = ""

    def __init__(self):
        self.restarts, self.switches = 0, []

    def stats(self):
        return None

    def start(self):
        self.restarts += 1

    def set_languages(self, source, target):
        self.switches.append((source, target))


OUTPUTS = [
    {"index": 25, "name": "Speakers (Realtek(R) Audio)", "channels": 2, "rate": 48000, "default": True},
    {"index": 29, "name": "喇叭 (Steam Streaming Speakers)", "channels": 2, "rate": 48000, "default": False},
]


@pytest.mark.parametrize(
    ("saved", "shown"),
    [
        (None, "Default output (follows speaker / earphones)"),
        ("Speakers (Realtek(R) Audio)", "Speakers (Realtek(R) Audio)"),
        ("Steam Streaming", "Steam Streaming"),  # part of a name, as --audio-device accepts
        ("USB Headset", "USB Headset (not connected)"),  # unplugged since it was chosen
    ],
)
def test_settings_apply_keeps_the_saved_device(qt_app, config_dir, monkeypatch, saved, shown):
    from tfv.capture import loopback
    from tfv.ui.settings_dialog import SettingsDialog

    monkeypatch.setattr(loopback, "list_output_devices", lambda: OUTPUTS)
    hw = Hardware(os="windows", arch="amd64", cpu="test", physical_cores=4, logical_cores=8, ram_mb=16000, ram_free_mb=8000)
    monkeypatch.setattr("tfv.hardware.probe", lambda: hw)
    prefs = Prefs(audio_device=saved, source_lang="ja", target_lang="zh")
    controller = FakeController()
    dialog = SettingsDialog(prefs, controller)
    assert (dialog.device.currentData(), dialog.device.currentText()) == (saved, shown)
    dialog.font_size.setValue(30)
    dialog.target.setCurrentIndex(dialog.target.findData("en"))
    dialog.apply()
    assert prefs.audio_device == saved and prefs.font_size == 30
    assert controller.restarts == 0 and controller.switches == [("ja", "en")]  # no reload for a style change
    assert Prefs.load().audio_device == saved
    dialog.close()


RUBY = [("持", "も"), ("ち", None), ("帰", "かえ"), ("りの", None), ("弁当", "べんとう"), ("を食べました。", None)]


def test_ruby_label_wraps_without_splitting_a_word_from_its_reading(qt_app):
    from tfv.ui.overlay import font_for
    from tfv.ui.ruby import RubyLabel

    label = RubyLabel()
    label.setFont(font_for("ja", 20))
    label.setSegments(RUBY)
    assert label.text() == "持ち帰りの弁当を食べました。"
    assert label.readings() == [("持", "も"), ("帰", "かえ"), ("弁当", "べんとう")]
    one_line = label.heightForWidth(2000)
    assert label.heightForWidth(60) > 2 * one_line  # wraps when narrow
    for _, units in label._lines(60):
        assert all(reading is None or text in ("持", "帰", "弁当") for text, reading, _ in units)
    label.resize(300, label.heightForWidth(300))
    assert not label.grab().isNull()  # paints without errors


def test_overlay_shows_furigana_and_follows_the_setting(qt_app):
    prefs = Prefs(furigana=True)
    ov = OverlayWindow(prefs)
    c = caption(1, text="持ち帰りの弁当を食べました。")
    c.ruby = RUBY
    ov.show_caption(c)
    row = ov._rows[1]
    assert row.ruby.isVisibleTo(ov) and not row.original.isVisibleTo(ov) and row.ruby.text() == c.text
    prefs.furigana = False
    ov.apply_style()  # redraws what is on screen
    assert row.original.isVisibleTo(ov) and not row.ruby.isVisibleTo(ov) and row.original.text() == c.text


def test_overlay_removes_live_text_that_was_noise(qt_app):
    ov = OverlayWindow(Prefs())
    ov.show_caption(caption(4, stage="partial", text="えー", translation=None))
    assert 4 in ov._rows
    ov.show_caption(caption(4, stage="dropped", text="", translation=None))
    assert not ov._rows


def test_bar_controls_follow_the_lock(qt_app):
    prefs = Prefs(locked=False)
    ov = OverlayWindow(prefs)
    ov.show()
    assert ov.toolbar.isVisibleTo(ov) and not ov._unlock.isVisible()
    ov.set_locked(True)  # click-through: the toolbar can't be used, the corner button can
    assert not ov.toolbar.isVisibleTo(ov) and ov._unlock.isVisible()
    assert ov._unlock.geometry().right() <= ov.geometry().right()
    prefs.unlock_button = False
    ov.apply_style()
    assert not ov._unlock.isVisible()
    ov.close()


def test_bar_buttons_ask_for_their_action(qt_app):
    ov = OverlayWindow(Prefs(locked=False))
    ov.show()
    asked = []
    ov.action_requested.connect(lambda name, where: asked.append(name))
    for name in ("smaller", "bigger", "furigana", "languages", "pause", "settings"):
        QTest.mouseClick(ov.toolbar.buttons[name], Qt.MouseButton.LeftButton)
    ov.set_locked(True)
    QTest.mouseClick(ov._unlock, Qt.MouseButton.LeftButton)
    assert asked == ["smaller", "bigger", "furigana", "languages", "pause", "settings", "unlock"]
    ov.close()


def test_app_applies_bar_actions_and_saves_them(qt_app, config_dir, monkeypatch):
    from tfv.ui import app as ui_app

    monkeypatch.setattr(ui_app.GlobalHotkeys, "register", lambda self, bindings: None)  # no system-wide keys in tests
    tray = ui_app.TrayApp(qt_app)
    size = tray.prefs.font_size
    tray._on_overlay_action("bigger", QPoint())
    assert tray.prefs.font_size == size + 2
    for _ in range(2):
        tray._on_overlay_action("smaller", QPoint())
    tray._on_overlay_action("furigana", QPoint())
    assert tray.prefs.furigana is False and not tray.overlay.toolbar.buttons["furigana"].isChecked()
    tray._on_overlay_action("original", QPoint())
    assert tray.prefs.show_original is False and not tray.overlay.toolbar.buttons["original"].isChecked()
    tray._on_overlay_action("lock", QPoint())
    assert tray.prefs.locked and not tray.overlay.toolbar.isVisibleTo(tray.overlay)
    tray._on_overlay_action("unlock", QPoint())
    assert not tray.prefs.locked
    saved = Prefs.load()
    assert (saved.font_size, saved.furigana, saved.show_original) == (size - 2, False, False)
    assert tray.overlay.toolbar.buttons["languages"].text() == "日→中 ▾"
    tray.overlay.close()
    tray.tray.hide()
