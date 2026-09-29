"""Audio sources. Each one delivers 16 kHz mono float32 chunks."""

from tfv.capture.base import AudioSource
from tfv.capture.file import FileSource

__all__ = ["AudioSource", "FileSource"]
