"""Settings window: languages, audio device, models, paid-service keys and overlay style."""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from tfv import keys, languages, usage
from tfv.ui.hotkeys import HOTKEYS
from tfv.ui.prefs import Prefs

log = logging.getLogger(__name__)

_CLOUD_ASR = ["openai:gpt-4o-mini-transcribe", "openai:gpt-4o-transcribe", "deepgram:nova-3"]
_CLOUD_MT = ["claude", "claude:claude-sonnet-5", "claude:claude-haiku-4-5", "deepl", "openai", "openai-compatible:MODEL"]


def _language_combo(include_off: bool) -> QComboBox:
    combo = QComboBox()
    if include_off:
        combo.addItem("Off (captions only)", None)
    for code in languages.SUPPORTED:
        combo.addItem(f"{languages.LABELS[code]} ({code})", code)
    return combo


def _select(combo: QComboBox, value) -> None:
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)
    elif combo.isEditable():
        combo.setEditText(str(value))


class SettingsDialog(QDialog):
    applied = Signal()

    def __init__(self, prefs: Prefs, controller, parent=None):
        super().__init__(parent)
        self.prefs = prefs
        self.controller = controller
        self.setWindowTitle("tfv settings")
        self.setMinimumWidth(560)
        tabs = QTabWidget()
        tabs.addTab(self._general_tab(), "General")
        tabs.addTab(self._models_tab(), "Models")
        tabs.addTab(self._paid_tab(), "Paid services")
        tabs.addTab(self._overlay_tab(), "Overlay")
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply)
        buttons.rejected.connect(self.close)
        layout = QVBoxLayout(self)
        layout.addWidget(tabs)
        layout.addWidget(buttons)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_live)
        self._timer.start(2000)
        self._refresh_live()

    # -- tabs ----------------------------------------------------------------------

    def _general_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.source = _language_combo(include_off=False)
        _select(self.source, self.prefs.source_lang)
        self.target = _language_combo(include_off=True)
        _select(self.target, self.prefs.target_lang)
        self.capture = QComboBox()
        self.capture.addItem("What this computer plays (speaker / earphones)", "output")
        self.capture.addItem("A microphone or other input device", "input")
        _select(self.capture, self.prefs.capture)
        self.device = QComboBox()
        self.capture.currentIndexChanged.connect(lambda _i: self._fill_devices(None))
        self._fill_devices(self.prefs.audio_device)
        form.addRow("Spoken language:", self.source)
        form.addRow("Translate into:", self.target)
        form.addRow("Capture:", self.capture)
        form.addRow("Device:", self.device)
        hotkeys = "\n".join(f"Ctrl+Alt+{k}: {what}" for k, what in HOTKEYS.items())
        form.addRow("Hotkeys:", QLabel(hotkeys))
        return w

    def _models_tab(self) -> QWidget:
        from tfv import hardware, resources

        w = QWidget()
        form = QFormLayout(w)
        hw = hardware.probe()
        auto_profile = resources.choose_profile(hw)
        form.addRow("This computer:", _wrap_label(hw.summary()))
        profiles = resources.load_profiles()
        self.profile = QComboBox()
        self.profile.addItem(f"auto ({auto_profile.name})", "auto")
        for p in profiles:
            self.profile.addItem(f"{p.name} — {p.summary}", p.name)
        _select(self.profile, self.prefs.profile)
        asr_ladders = [r for p in profiles for r in (p.asr, *p.asr_by_lang.values())]
        mt_ladders = [r for p in profiles for r in (p.mt, *p.mt_by_lang.values())]
        asr_options = dict.fromkeys(str(c) for ladder in asr_ladders for c in ladder)
        mt_options = dict.fromkeys(str(c) for ladder in mt_ladders for c in ladder)
        self.asr = _choice_combo(["auto", *asr_options, *_CLOUD_ASR], self.prefs.asr)
        self.mt = _choice_combo(["auto", "none", *mt_options, *_CLOUD_MT], self.prefs.mt)
        self.mt_url = QLineEdit(self.prefs.mt_url)
        self.mt_url.setPlaceholderText("only for openai-compatible, e.g. http://localhost:11434 (Ollama)")
        self.adapt = QCheckBox("Switch to lighter models automatically if captions fall behind")
        self.adapt.setChecked(self.prefs.adapt)
        form.addRow("Hardware profile:", self.profile)
        form.addRow("Speech recognition:", self.asr)
        form.addRow("Translation:", self.mt)
        form.addRow("Server URL:", self.mt_url)
        form.addRow("", self.adapt)
        form.addRow("", _wrap_label("“auto” uses the profile's first option that fits this computer; see `python -m tfv hardware`."))
        self.remote_url = QLineEdit(self.prefs.remote_url)
        self.remote_url.setPlaceholderText("empty = run here; or ws://PC-address:8765 from `tfv serve` on a stronger PC")
        form.addRow("Remote engine:", self.remote_url)
        form.addRow("", _wrap_label("With a remote engine, set its token under Paid services → remote."))
        self.live = _wrap_label("")
        form.addRow("Now running:", self.live)
        return w

    def _paid_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(
            _wrap_label(
                "Paid services are only used when you pick them (the “cloud” profile, or a service under Models). "
                "Audio or text is then sent to that provider. Keys are kept in the system credential store."
            )
        )
        box = QGroupBox("API keys")
        grid = QGridLayout(box)
        self._key_labels: dict[str, QLabel] = {}
        for row, service in enumerate(keys.SERVICES):
            status = QLabel()
            self._key_labels[service] = status
            set_btn = QPushButton("Set key…")
            set_btn.clicked.connect(lambda _=False, s=service: self._set_key(s))
            del_btn = QPushButton("Remove")
            del_btn.clicked.connect(lambda _=False, s=service: self._delete_key(s))
            grid.addWidget(QLabel(service), row, 0)
            grid.addWidget(status, row, 1)
            grid.addWidget(set_btn, row, 2)
            grid.addWidget(del_btn, row, 3)
        layout.addWidget(box)
        self.usage = _wrap_label("")
        usage_box = QGroupBox("Usage this session (estimate)")
        QVBoxLayout(usage_box).addWidget(self.usage)
        layout.addWidget(usage_box)
        layout.addStretch()
        self._refresh_keys()
        return w

    def _overlay_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.font_size = _spin(12, 72, self.prefs.font_size, " px")
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(int(self.prefs.opacity * 100))
        self.lines = _spin(1, 4, self.prefs.lines)
        self.clear_after = _spin(2, 60, self.prefs.clear_after_s, " s")
        self.show_original = QCheckBox("Show the original text above the translation")
        self.show_original.setChecked(self.prefs.show_original)
        self.show_together = QCheckBox("Show the original and its translation at the same time")
        self.show_together.setChecked(self.prefs.show_together)
        self.locked = QCheckBox("Lock the overlay (clicks pass through to the window below)")
        self.locked.setChecked(self.prefs.locked)
        form.addRow("Text size:", self.font_size)
        form.addRow("Background opacity:", self.opacity)
        form.addRow("Captions on screen:", self.lines)
        form.addRow("Clear after silence:", self.clear_after)
        form.addRow("", self.show_original)
        form.addRow("", self.show_together)
        form.addRow("", self.locked)
        return w

    def _fill_devices(self, selected: str | None) -> None:
        self.device.clear()
        try:
            if self.capture.currentData() == "input":
                from tfv.capture.input_device import list_input_devices as list_devices

                self.device.addItem("Default input", None)
            else:
                from tfv.capture.loopback import list_output_devices as list_devices

                self.device.addItem("Default output (follows speaker / earphones)", None)
            for dev in list_devices():
                self.device.addItem(dev["name"], dev["name"])
        except Exception as exc:
            log.warning("Could not list devices: %s", exc)
        _select(self.device, selected)

    # -- actions ---------------------------------------------------------------------

    def apply(self) -> None:
        p = self.prefs
        before_engine, before_langs = p.engine_key(), (p.source_lang, p.target_lang)
        p.capture = self.capture.currentData()
        p.audio_device = self.device.currentData()
        p.remote_url = self.remote_url.text().strip()
        p.profile = self.profile.currentData()
        p.asr = self.asr.currentText().strip() or "auto"
        p.mt = self.mt.currentText().strip() or "auto"
        p.mt_url = self.mt_url.text().strip()
        p.adapt = self.adapt.isChecked()
        p.font_size = self.font_size.value()
        p.opacity = self.opacity.value() / 100
        p.lines = self.lines.value()
        p.clear_after_s = self.clear_after.value()
        p.show_original = self.show_original.isChecked()
        p.show_together = self.show_together.isChecked()
        p.locked = self.locked.isChecked()
        source, target = self.source.currentData(), self.target.currentData()
        if p.engine_key() != before_engine:
            p.source_lang, p.target_lang = source, target
            self.controller.start()  # reload everything with the new models or device
        elif (source, target) != before_langs:
            self.controller.set_languages(source, target)
        p.save()
        self.applied.emit()

    def _set_key(self, service: str) -> None:
        key, ok = QInputDialog.getText(self, f"{service} API key", f"Paste your {service} API key:", QLineEdit.EchoMode.Password)
        if ok and key.strip():
            try:
                keys.store(service, key)
            except Exception as exc:
                QMessageBox.warning(self, "tfv", f"Could not store the key: {exc}")
        self._refresh_keys()

    def _delete_key(self, service: str) -> None:
        keys.delete(service)
        self._refresh_keys()

    def _refresh_keys(self) -> None:
        for service, label in self._key_labels.items():
            env = keys.SERVICES[service]
            label.setText("set" if keys.has(service) else f"not set (or set {env})")

    def _refresh_live(self) -> None:
        session = self.controller.session
        if session is None:
            text = {"loading": "loading models…", "error": f"error: {self.controller.error}"}.get(self.controller.state, "stopped")
        else:
            parts = session.describe()
            if stats := self.controller.stats():
                parts.append(
                    f"Recognition speed {stats['asr_rtf']:.2f}x real time, translation {stats['mt_s']:.2f}s per line, "
                    f"skipped {stats['dropped_utterances']} utterances / {stats['dropped_translations']} translations"
                )
            text = "\n".join(parts)
        self.live.setText(text)
        lines = usage.summary()
        self.usage.setText("\n".join(lines) + (f"\nTotal ≈ ${usage.total_cost():.3f}" if lines else "") or "No paid services used.")


def _choice_combo(options: list[str], current: str) -> QComboBox:
    combo = QComboBox()
    combo.setEditable(True)
    for option in dict.fromkeys(options):
        combo.addItem(option, option)
    _select(combo, current)
    return combo


def _spin(low: int, high: int, value: int, suffix: str = "") -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(low, high)
    spin.setValue(value)
    spin.setSuffix(suffix)
    return spin


def _wrap_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label
