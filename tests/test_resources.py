import pytest

from tfv.hardware import GPU, Hardware
from tfv.resources import Choice, Ladder, Monitor, choose_profile, load_profiles


def machine(gpu: GPU | None = None, cuda: bool = False, ram_mb: int = 16000, cores: int = 4, os: str = "windows") -> Hardware:
    return Hardware(
        os=os, arch="arm64" if os == "macos" else "amd64", cpu="test", physical_cores=cores, logical_cores=cores * 2,
        ram_mb=ram_mb, ram_free_mb=ram_mb // 2, gpus=[gpu] if gpu else [], cuda=cuda,
    )  # fmt: skip


def test_choice_parsing():
    assert Choice.parse("whisper-large-v3-turbo@cuda") == Choice("faster-whisper", "whisper-large-v3-turbo", "cuda")
    assert Choice.parse("sensevoice-small") == Choice("sensevoice", "sensevoice-small", "cpu")
    assert Choice.parse("qwen3.5-2b") == Choice("llama", "qwen3.5-2b", "gpu")
    assert Choice.parse("claude:claude-haiku-4-5") == Choice("claude", "claude-haiku-4-5", "cloud")
    assert Choice.parse("deepl") == Choice("deepl", None, "cloud")
    assert str(Choice.parse("nllb-600m")) == "nllb-600m@cpu"
    with pytest.raises(ValueError):
        Choice.parse("no-such-model")


def test_every_profile_rung_is_valid():
    profiles = load_profiles()  # parses every rung
    assert profiles[-2].name == "cpu-low" and profiles[-2].when == {}


@pytest.mark.parametrize(
    ("hw", "expected"),
    [
        (machine(GPU("RTX 3050 Laptop", "nvidia", 4096), cuda=True), "gpu-small"),
        (machine(GPU("RTX 4070", "nvidia", 12288), cuda=True), "gpu-mid"),
        (machine(GPU("RTX 4090", "nvidia", 24564), cuda=True), "gpu-high"),
        (machine(GPU("RTX 3050 Laptop", "nvidia", 4096), cuda=False), "vulkan-gpu"),
        (machine(GPU("Radeon RX 7600", "amd", 8192)), "vulkan-gpu"),
        (machine(GPU("Intel Iris Xe", "intel", 1024)), "cpu-low"),
        (machine(cores=8, ram_mb=32000), "cpu-high"),
        (machine(ram_mb=8000), "cpu-low"),
        (machine(GPU("Apple Silicon GPU", "apple", 10000), os="macos"), "apple"),
    ],
)
def test_profile_selection(hw, expected):
    assert choose_profile(hw).name == expected


def test_per_language_ladders():
    small = choose_profile(machine(GPU("RTX 3050 Laptop", "nvidia", 4096), cuda=True))
    assert small.asr_for("en")[0] == Choice.parse("whisper-large-v3-turbo@cuda")
    assert small.asr_for("ja")[0] == Choice.parse("sensevoice-small@cpu")  # frees the GPU for a 4B translator
    assert small.mt_for("zh")[0] == Choice.parse("translategemma-4b@gpu")
    assert small.mt_for("en")[0] == Choice.parse("qwen3.5-4b@gpu")
    assert small.mt_for(None) == small.mt
    cpu = choose_profile(machine(ram_mb=8000))
    assert cpu.asr_for("ja") == cpu.asr and cpu.mt_for("zh") == cpu.mt  # no overrides: the defaults


def test_cloud_profile_is_never_automatic():
    assert choose_profile(machine(), "cloud").name == "cloud"
    assert all(p.name != "cloud" for p in load_profiles() if p.auto)


class Fake:
    def __init__(self, name, speed=0.1):
        self.name, self.speed, self.closed = name, speed, False

    def close(self):
        self.closed = True


def ladder(**kw):
    rungs = [Choice.parse(c) for c in ("qwen3.5-4b@gpu", "qwen3.5-2b@gpu", "nllb-600m@cpu")]
    defaults = dict(build=lambda c: Fake(c.model), unfit=lambda c: None, bench=lambda b: (b.speed, ""), limit=1.0)
    return Ladder("translation", rungs, **{**defaults, **kw})


def test_ladder_takes_first_rung_that_works():
    lad = ladder(unfit=lambda c: "too big" if c.model == "qwen3.5-4b" else None)
    assert lad.load().name == "qwen3.5-2b"
    assert lad.current.model == "qwen3.5-2b"
    assert lad.skipped == ["qwen3.5-4b@gpu: too big"]


def test_ladder_skips_failures_and_slow_rungs():
    built = []

    def build(c):
        if c.model == "qwen3.5-4b":
            raise RuntimeError("out of memory")
        built.append(Fake(c.model, speed=5.0 if c.model == "qwen3.5-2b" else 0.1))
        return built[-1]

    lad = ladder(build=build)
    assert lad.load().name == "nllb-600m"
    assert built[0].closed  # the slow one was released


def test_last_rung_is_used_even_if_slow_or_unfit():
    lad = ladder(unfit=lambda c: "no", bench=lambda b: (99.0, ""))
    assert lad.load().name == "nllb-600m"


def test_step_down():
    lad = ladder()
    assert lad.load().name == "qwen3.5-4b"
    assert lad.step_down().name == "qwen3.5-2b"
    assert lad.step_down().name == "nllb-600m"
    assert not lad.can_step_down and lad.step_down() is None


class FakePipeline:
    def __init__(self, **stats):
        self.s = dict(
            captions=0, translated=0, asr_rtf=0.1, mt_s=0.5, dropped_utterances=0, dropped_translations=0, failed_translations=0
        )
        self.s.update(stats)
        self.resets = 0

    def stats(self):
        return dict(self.s)

    def reset_stats(self):
        self.resets += 1


def test_monitor_steps_down_slow_translation_once_per_cooldown():
    calls = []
    p = FakePipeline(translated=5, mt_s=6.0)
    m = Monitor(p, lambda stage, reason: calls.append(stage) or True, cooldown=0)
    m.check()
    assert calls == ["mt"] and p.resets == 1
    m._cooldown = 60
    m.check()
    assert calls == ["mt"]


def test_monitor_steps_down_a_failing_translator():
    calls = []
    p = FakePipeline()
    m = Monitor(p, lambda stage, reason: calls.append((stage, reason)) or True, cooldown=0)
    p.s["failed_translations"] = 2
    m.check()
    assert calls == []  # a couple of failures can be a restart
    p.s["failed_translations"] = 6
    m.check()
    assert calls == [("mt", "4 translations failed")]


def test_monitor_reacts_to_dropped_utterances():
    calls = []
    p = FakePipeline()
    m = Monitor(p, lambda stage, reason: calls.append(stage) or True, cooldown=0)
    m.check()
    assert calls == []
    p.s["dropped_utterances"] = 2
    m.check()
    assert calls == ["asr"]
