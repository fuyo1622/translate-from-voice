"""Remote engine: run the models on a stronger PC and caption a weaker one over the LAN.

    laptop (RemoteSession)                         desktop (serve)
    capture ─ 16 kHz PCM16 over WebSocket ───────► VAD → ASR → MT
    overlay ◄──────────── captions as JSON ──────── (models loaded once)

Protocol (version 1): the client first sends {"type": "hello", "token",
"source_lang", "target_lang"}, then binary audio frames, and may send
{"type": "languages", ...} later. The server answers "ready" (or "error"),
then streams "caption" and "status" messages. One audio client at a time.
Traffic is not encrypted: use it on a trusted network only.
"""

from __future__ import annotations

import dataclasses
import hmac
import ipaddress
import json
import logging
import queue
import secrets
import threading
from typing import Callable

import numpy as np

from tfv.capture.base import AudioCallback, AudioSource
from tfv.engine.pipeline import Caption
from tfv.session import Session, SessionConfig, make_source

log = logging.getLogger(__name__)

PROTOCOL = 1
DEFAULT_PORT = 8765
_CAPTION_FIELDS = [f.name for f in dataclasses.fields(Caption) if f.name != "speech_ended_at"]


def caption_to_message(caption: Caption) -> dict:
    return {"type": "caption", **{name: getattr(caption, name) for name in _CAPTION_FIELDS}}


def caption_from_message(message: dict) -> Caption:
    return Caption(**{name: message[name] for name in _CAPTION_FIELDS if name in message})


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


# -- server ----------------------------------------------------------------------------


class NetworkAudioSource(AudioSource):
    """Audio arriving from the connected client."""

    realtime = True
    description = "network client"

    def __init__(self):
        self._on_audio: AudioCallback | None = None

    def start(self, on_audio: AudioCallback, on_end: Callable[[], None]) -> None:
        self._on_audio = on_audio

    def stop(self) -> None:
        self._on_audio = None

    def feed(self, data: bytes) -> None:
        if self._on_audio and len(data) >= 2:
            samples = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768
            self._on_audio(samples)


class EngineServer:
    def __init__(self, config: SessionConfig, host: str = "127.0.0.1", port: int = DEFAULT_PORT, token: str | None = None):
        if not token and not _is_loopback(host):
            token = secrets.token_urlsafe(12)
            log.warning("Reachable from the network, so a token is required: %s", token)
        self.host, self.port, self.token = host, port, token
        self._source = NetworkAudioSource()
        self._client = None
        self._lock = threading.Lock()
        self.session = Session(config, lambda c: self._send(caption_to_message(c)), self._status, source=self._source)

    def serve_forever(self) -> None:
        from websockets.sync.server import serve

        self.session.start()  # models load once, before any client connects
        with serve(self._handle, self.host, self.port, max_size=1 << 20) as server:
            log.info("Engine ready on ws://%s:%d%s", self.host, self.port, " (token required)" if self.token else "")
            try:
                server.serve_forever()
            finally:
                self.session.stop()

    def _status(self, message: str) -> None:
        log.info("%s", message)
        self._send({"type": "status", "message": message})

    def _send(self, message: dict) -> None:
        client = self._client
        if client is not None:
            try:
                client.send(json.dumps(message, ensure_ascii=False))
            except Exception as exc:  # the client went away; its handler cleans up
                log.debug("Send failed: %s", exc)

    def _handle(self, ws) -> None:
        peer = ws.remote_address
        try:
            hello = json.loads(ws.recv(timeout=15))
        except Exception:
            ws.close(1002, "expected hello")
            return
        if hello.get("type") != "hello" or (self.token and not hmac.compare_digest(str(hello.get("token") or ""), self.token)):
            log.warning("Rejected %s: bad token", peer)
            ws.send(json.dumps({"type": "error", "message": "wrong or missing token"}))
            ws.close(1008, "unauthorized")
            return
        with self._lock:
            if self._client is not None:
                ws.send(json.dumps({"type": "error", "message": "another client is already connected"}))
                ws.close(1013, "busy")
                return
            self._client = ws
        log.info("Client %s connected", peer)
        try:
            self._apply_languages(hello)
            ws.send(json.dumps({"type": "ready", "protocol": PROTOCOL, "engine": self.session.describe()}))
            for message in ws:
                if isinstance(message, bytes):
                    self._source.feed(message)
                else:
                    msg = json.loads(message)
                    if msg.get("type") == "languages":
                        self._apply_languages(msg)
        except Exception as exc:
            log.info("Client %s: %s", peer, exc)
        finally:
            with self._lock:
                self._client = None
            log.info("Client %s disconnected", peer)

    def _apply_languages(self, msg: dict) -> None:
        cfg = self.session.config
        source, target = msg.get("source_lang") or cfg.source_lang, msg.get("target_lang")
        if (source, target) != (cfg.source_lang, cfg.target_lang):
            self.session.set_languages(source, target)


# -- client ----------------------------------------------------------------------------


class RemoteSession:
    """Same interface as Session, but the models run on a server."""

    pipeline = None

    def __init__(self, url: str, token: str | None, config: SessionConfig, on_caption, on_status=None):
        self.url, self.token, self.config = url, token, config
        self._on_caption = on_caption
        self._status = on_status or (lambda msg: log.info("%s", msg))
        self._ws = None
        self._source = None
        self._engine: list[str] = []
        self._outbox: queue.Queue = queue.Queue(maxsize=100)
        self._paused = threading.Event()
        self._done = threading.Event()

    def start(self) -> None:
        from websockets.sync.client import connect

        self._status(f"Connecting to {self.url}…")
        self._ws = connect(self.url, open_timeout=10, max_size=1 << 20)
        cfg = self.config
        self._ws.send(json.dumps({"type": "hello", "protocol": PROTOCOL, "token": self.token, "source_lang": cfg.source_lang, "target_lang": cfg.target_lang}))
        reply = json.loads(self._ws.recv(timeout=600))  # the server may reload models for other languages
        if reply.get("type") == "error":
            self._ws.close()
            raise RuntimeError(f"server refused: {reply.get('message')}")
        self._engine = reply.get("engine", [])
        threading.Thread(target=self._receive, name="tfv-remote-recv", daemon=True).start()
        threading.Thread(target=self._send_audio, name="tfv-remote-send", daemon=True).start()
        self._source = make_source(cfg, self._status)
        self._source.start(self._queue_audio, self._done.set)
        self._status(f"Listening to {self._source.description}, models on {self.url}")

    def stop(self) -> None:
        self._done.set()  # first, so the receiver knows the disconnect is intended
        if self._source:
            self._source.stop()
        if self._ws:
            self._ws.close()

    def wait(self, timeout: float | None = None) -> bool:
        return self._done.wait(timeout)

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def describe(self) -> list[str]:
        return [f"Remote engine: {self.url}", *self._engine]

    def stats(self) -> None:
        return None

    def set_languages(self, source_lang: str, target_lang: str | None) -> None:
        self.config.source_lang, self.config.target_lang = source_lang, target_lang
        self._ws.send(json.dumps({"type": "languages", "source_lang": source_lang, "target_lang": target_lang}))

    def _queue_audio(self, chunk: np.ndarray) -> None:
        if self._paused.is_set():
            return
        try:  # never block the audio callback on the network
            self._outbox.put_nowait((np.clip(chunk, -1, 1) * 32767).astype("<i2").tobytes())
        except queue.Full:
            log.warning("Network too slow; dropping audio")

    def _send_audio(self) -> None:
        while not self._done.is_set():
            try:
                data = self._outbox.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._ws.send(data)
            except Exception:
                break

    def _receive(self) -> None:
        try:
            for message in self._ws:
                msg = json.loads(message)
                if msg.get("type") == "caption":
                    self._on_caption(caption_from_message(msg))
                elif msg.get("type") in ("status", "error"):
                    self._status(msg.get("message", ""))
        except Exception as exc:
            log.info("Connection closed: %s", exc)
        finally:
            if not self._done.is_set():
                self._status("Disconnected from the engine server")
            self._done.set()
