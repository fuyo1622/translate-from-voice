"""Runs the engine off the GUI thread and reports back through Qt signals."""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, Signal

from tfv.session import Session, SessionConfig
from tfv.ui.prefs import Prefs

# Session and remote.RemoteSession share: start, stop, pause, resume, set_languages, set_partials, describe,
# summary, stats.

log = logging.getLogger(__name__)


class EngineController(QObject):
    caption = Signal(object)
    """A Caption was recognized or translated (emitted from engine threads, delivered on the GUI thread)."""
    status = Signal(str)
    state_changed = Signal(str)
    """"stopped", "loading", "running", "paused" or "error"."""

    def __init__(self, prefs: Prefs):
        super().__init__()
        self.prefs = prefs
        self.session = None
        self.state = "stopped"
        self.error = ""
        self._lock = threading.Lock()
        self._generation = 0  # ignores callbacks from sessions that were replaced
        self.switching = False
        """A language switch is loading; choices made meanwhile wait in _wanted (only the latest)."""
        self._wanted: tuple[str, str | None] | None = None

    # -- lifecycle ---------------------------------------------------------------

    def start(self) -> None:
        """(Re)load the engine with the current prefs in the background."""
        with self._lock:
            self._generation += 1
            generation = self._generation
            old, self.session = self.session, None
        self._set_state("loading")
        threading.Thread(target=self._run, args=(old, generation), name="tfv-engine-start", daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            old, self.session = self.session, None
        if old:
            old.stop()
        self._set_state("stopped")

    def _run(self, old, generation: int) -> None:
        if old:
            old.stop()  # free the GPU before loading again
        p = self.prefs
        config = SessionConfig(
            source_lang=p.source_lang, target_lang=p.target_lang, capture=p.capture, audio_device=p.audio_device,
            profile=p.profile, asr=p.asr, mt=p.mt, mt_url=p.mt_url or None, adapt=p.adapt, partials=p.live_text,
            vad_denoise=p.vad_denoise, separate_vocals=p.separate_vocals, song_mode=p.song_mode,
        )  # fmt: skip
        def on_caption(caption) -> None:
            self._on_caption(caption, generation)

        def on_status(message: str) -> None:
            self._on_status(message, generation)

        if p.remote_url:
            from tfv import keys
            from tfv.remote import RemoteSession

            session = RemoteSession(p.remote_url, keys.get("remote"), config, on_caption, on_status)
        else:
            session = Session(config, on_caption, on_status)
        try:
            session.start()
        except Exception as exc:
            log.exception("Engine failed to start")
            session.stop()
            if generation == self._generation:
                self.error = str(exc)
                self._set_state("error")
                self.status.emit(f"Could not start: {exc}")
            return
        with self._lock:
            if generation != self._generation:  # replaced or stopped while loading
                stale = True
            else:
                stale = False
                self.session = session
        if stale:
            session.stop()
            return
        self._set_state("running")
        if (p.source_lang, p.target_lang) != (config.source_lang, config.target_lang):
            self.set_languages(p.source_lang, p.target_lang)  # changed in the menu while loading

    # -- live changes ------------------------------------------------------------------

    def set_paused(self, paused: bool) -> None:
        session = self.session
        if not session:
            return
        if paused:
            session.pause()
            self._set_state("paused")
        else:
            session.resume()
            self._set_state("running")

    def set_languages(self, source: str, target: str | None) -> None:
        """Switch languages; reloads only what the change needs. Choices made while a switch is
        still loading don't queue up: only the latest one is applied when it finishes."""
        self.prefs.source_lang, self.prefs.target_lang = source, target
        session = self.session
        if session is None:
            if self.state in ("stopped", "error"):
                self.start()
            return  # a session that is loading picks the prefs up on the next start
        with self._lock:
            self._wanted = (source, target)
            if self.switching:
                return
            self.switching = True
        threading.Thread(target=self._switch, args=(session, self._generation), name="tfv-languages", daemon=True).start()

    def _switch(self, session, generation: int) -> None:
        done = None
        while True:
            with self._lock:
                wanted = self._wanted
                if wanted == done or generation != self._generation:
                    self.switching = False
                    break
            try:
                session.set_languages(*wanted)
            except Exception as exc:
                log.exception("Language switch failed")
                if generation == self._generation:
                    self.status.emit(f"Could not switch languages: {exc}")
            done = wanted
        if generation == self._generation:
            self.status.emit(session.summary())

    def set_live(self, on: bool) -> None:
        """Live text on or off (a session that is still loading reads prefs.live_text when it starts)."""
        self.prefs.live_text = on
        if self.session:
            try:
                self.session.set_partials(on)
            except Exception as exc:  # a remote engine that went away
                log.warning("Could not switch live text: %s", exc)

    def set_separation(self, on: bool) -> None:
        """Music filter on or off; loads or frees it in the background and reports what happened."""
        self.prefs.separate_vocals = on
        self.prefs.save()
        session = self.session
        if session is None or not hasattr(session, "set_separate_vocals"):
            return  # a session that is loading reads the prefs; a remote engine decides for itself

        def work():
            try:
                session.set_separate_vocals(on)
                self.status.emit(next((line for line in session.describe() if line.startswith("Music filter:")), ""))
            except Exception as exc:
                log.exception("Switching the music filter failed")
                self.status.emit(f"Could not switch the music filter: {exc}")

        threading.Thread(target=work, name="tfv-separation", daemon=True).start()

    def set_song_mode(self, on: bool) -> None:
        """Song mode on or off (takes effect with the next music)."""
        self.prefs.song_mode = on
        self.prefs.save()
        session = self.session
        if session is not None and hasattr(session, "set_song_mode"):
            session.set_song_mode(on)
            self.status.emit(next((line for line in session.describe() if line.startswith("Music filter:")), ""))

    def save_recent(self) -> None:
        """Save the last minute of audio and captions for feedback (local engine only)."""
        session = self.session
        if session is None or not hasattr(session, "save_recent"):
            self.status.emit("Nothing to save yet" if session is None else "Saving is not available with a remote engine")
            return

        def work():
            try:
                path = session.save_recent()
                self.status.emit(f"Saved the last minute: {path}")
            except Exception as exc:
                log.exception("Saving the last minute failed")
                self.status.emit(f"Could not save the last minute: {exc}")

        threading.Thread(target=work, name="tfv-save", daemon=True).start()

    def stats(self) -> dict | None:
        return self.session.stats() if self.session else None

    # -- callbacks from engine threads ---------------------------------------------------

    def _on_caption(self, caption, generation: int) -> None:
        if generation == self._generation:
            self.caption.emit(caption)

    def _on_status(self, message: str, generation: int) -> None:
        if generation == self._generation:
            log.info("%s", message)
            self.status.emit(message)

    def _set_state(self, state: str) -> None:
        self.state = state
        self.state_changed.emit(state)
