"""API keys for paid services: environment variables first, then the OS credential store.

Keys are never written to config files. `python -m tfv keys set <service>`
stores one in Windows Credential Manager / macOS Keychain / Secret Service.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

SERVICES = {
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "deepl": "DEEPL_AUTH_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
}
_KEYRING_NAME = "tfv"


def get(service: str) -> str | None:
    if service not in SERVICES:
        return None
    if value := os.environ.get(SERVICES[service]):
        return value
    try:
        import keyring

        return keyring.get_password(_KEYRING_NAME, service)
    except Exception as exc:  # no usable credential store (e.g. headless Linux)
        log.debug("Credential store unavailable: %s", exc)
        return None


def has(service: str) -> bool:
    return bool(get(service))


def store(service: str, key: str) -> None:
    import keyring

    if service not in SERVICES:
        raise ValueError(f"unknown service {service!r}; choose from {', '.join(SERVICES)}")
    keyring.set_password(_KEYRING_NAME, service, key.strip())


def delete(service: str) -> bool:
    import keyring
    from keyring.errors import PasswordDeleteError

    try:
        keyring.delete_password(_KEYRING_NAME, service)
        return True
    except PasswordDeleteError:
        return False
