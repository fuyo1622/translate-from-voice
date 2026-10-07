"""Capture sources against fake audio libraries: mixing, resampling, following the default output."""

import sys
import time
import types

import numpy as np
import pytest

from tfv import SAMPLE_RATE


class FakeInputStream:
    opened = []

    def __init__(self, device, samplerate, channels, dtype, blocksize, callback):
        self.device, self.rate, self.channels, self.callback = device, samplerate, channels, callback
        self.running = False
        FakeInputStream.opened.append(self)

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def close(self):
        pass


@pytest.fixture
def fake_sounddevice(monkeypatch):
    devices = [
        {"name": "Microphone (MME)", "max_input_channels": 2, "default_samplerate": 44100, "hostapi": 0},
        {"name": "Microphone Array (Realtek)", "max_input_channels": 2, "default_samplerate": 48000, "hostapi": 1},
        {"name": "Speakers (Realtek)", "max_input_channels": 0, "default_samplerate": 48000, "hostapi": 1},
    ]
    sd = types.SimpleNamespace(
        InputStream=FakeInputStream,
        query_hostapis=lambda: [{"name": "MME", "default_input_device": 0}, {"name": "Windows WASAPI", "default_input_device": 1}],
        query_devices=lambda index=None: devices if index is None else devices[index],
        default=types.SimpleNamespace(device=(1, 2)),
    )
    FakeInputStream.opened = []
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setattr(sys, "platform", "win32")  # lists the WASAPI devices, as on Windows
    return sd


def test_input_devices_are_listed_once(fake_sounddevice):
    from tfv.capture.input_device import list_input_devices

    assert list_input_devices() == [{"index": 1, "name": "Microphone Array (Realtek)", "channels": 2, "rate": 48000, "default": True}]


def test_input_source_mixes_to_mono_16k(fake_sounddevice):
    from tfv.capture.input_device import InputDeviceSource

    got = []
    source = InputDeviceSource("array")
    source.start(got.append, lambda: None)
    stream = FakeInputStream.opened[-1]
    assert (stream.device, stream.rate, stream.channels, stream.running) == (1, 48000, 2, True)
    t = np.arange(48000) / 48000
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([left, np.zeros_like(left)], axis=1).astype(np.float32)  # sound on the left only
    for i in range(0, 48000, 4800):
        stream.callback(stereo[i : i + 4800], 4800, None, None)
    out = np.concatenate(got)
    assert abs(len(out) - SAMPLE_RATE) < 400  # resampled to 16 kHz (the resampler holds back a little)
    assert np.sqrt(np.mean(out[1000:-1000] ** 2)) == pytest.approx(0.25 / np.sqrt(2), rel=0.05)  # mixed down
    source.stop()
    assert not stream.running


def test_unknown_input_device_is_reported(fake_sounddevice):
    from tfv.capture.input_device import InputDeviceSource

    with pytest.raises(ValueError, match="no input device matches 'usb'"):
        InputDeviceSource("usb").start(lambda a: None, lambda: None)


class FakePyAudio:
    """Just enough of pyaudiowpatch: two outputs, one default, streams that record their callback."""

    default = 1
    streams = []
    devices = {
        1: {"index": 1, "name": "Speakers (Realtek) [Loopback]", "defaultSampleRate": 48000, "maxInputChannels": 2},
        2: {"index": 2, "name": "Headphones (USB) [Loopback]", "defaultSampleRate": 44100, "maxInputChannels": 2},
    }

    def get_default_wasapi_loopback(self):
        return self.devices[FakePyAudio.default]

    def get_loopback_device_info_generator(self):
        return iter(self.devices.values())

    def open(self, **kw):
        stream = types.SimpleNamespace(kw=kw, open=True)
        stream.stop_stream = lambda: None
        stream.close = lambda: setattr(stream, "open", False)
        FakePyAudio.streams.append(stream)
        return stream

    def terminate(self):
        pass


def test_loopback_follows_the_default_output(monkeypatch):
    from tfv.capture import loopback, win_audio

    fake = types.SimpleNamespace(PyAudio=FakePyAudio, paFloat32=1, paContinue=0)
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", fake)
    monkeypatch.setattr(win_audio, "default_output_id", lambda: f"endpoint-{FakePyAudio.default}")
    monkeypatch.setattr(win_audio, "com_thread", lambda: __import__("contextlib").nullcontext())
    FakePyAudio.default, FakePyAudio.streams = 1, []

    got, changes = [], []
    source = loopback.LoopbackSource(None)
    source.on_device_change = changes.append
    source.start(got.append, lambda: None)
    first = FakePyAudio.streams[-1]
    assert first.kw["input_device_index"] == 1 and source.description.startswith("Speakers (Realtek)")
    first.kw["stream_callback"](np.full(9600, 0.1, np.float32).tobytes(), 4800, None, 0)  # 0.1 s stereo
    assert sum(map(len, got)) > 0

    time.sleep(0.5)  # let the watcher read the current default first
    FakePyAudio.default = 2  # earphones plugged in: Windows switches the default output
    deadline = time.monotonic() + 6
    while not changes and time.monotonic() < deadline:
        time.sleep(0.1)
    source.stop()
    assert changes and changes[0].startswith("Headphones (USB) (44100 Hz")
    second = FakePyAudio.streams[-1]
    assert not first.open and second.kw["input_device_index"] == 2 and second.kw["rate"] == 44100
    assert not second.open  # closed by stop()
