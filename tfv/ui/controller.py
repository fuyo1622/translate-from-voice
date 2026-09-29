"""Runs the engine off the GUI thread and reports back through Qt signals."""

from __future__ import annotations

import logging
import threading

from PySide6.QtCore import QObject, Signal

from tfv.session import Session, SessionConfig
from tfv.ui.prefs import Prefs

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
        self.session: Session | None = None
        self.state = "stopped"
        self.error = ""
        self._lock = threading.Lock()
        self._generation = 0  # ignores callbacks from sessions that were replaced

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

    def _run(self, old: Session | None, generation: int) -> None:
        if old:
            old.stop()  # free the GPU before loading again
        p = self.prefs
        config = SessionConfig(
            source_lang=p.source_lang, target_lang=p.target_lang, audio_device=p.audio_device,
            profile=p.profile, asr=p.asr, mt=p.mt, mt_url=p.mt_url or None, adapt=p.adapt,
        )  # fmt: skip
        session = Session(config, lambda c: self._on_caption(c, generation), lambda m: self._on_status(m, generation))
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

    # -- live changes ------------------------------------------------------------------

    def set_paused(self, paused: bool) -> None:
        session = self.session
        if not session or not session.pipeline:
            return
        if paused:
            session.pipeline.pause()
            self._set_state("paused")
        else:
            session.pipeline.resume()
            self._set_state("running")

    def set_languages(self, source: str, target: str | None) -> None:
        """Switch languages; reloads only what the change needs."""
        self.prefs.source_lang, self.prefs.target_lang = source, target
        session = self.session
        if session is None:
            if self.state in ("stopped", "error"):
                self.start()
            return  # a session that is loading picks the prefs up on the next start
        generation = self._generation

        def work():
            try:
                session.set_languages(source, target)
            except Exception as exc:
                log.exception("Language switch failed")
                if generation == self._generation:
                    self.status.emit(f"Could not switch languages: {exc}")

        threading.Thread(target=work, name="tfv-languages", daemon=True).start()

    def stats(self) -> dict | None:
        session = self.session
        return session.pipeline.stats() if session and session.pipeline else None

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
