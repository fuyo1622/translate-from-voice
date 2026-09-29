# tfv — translate from voice

Live subtitles for whatever your computer is playing (speaker or earphones):
it listens to the audio output, recognizes the speech, and translates it between
**Traditional Chinese, English and Japanese**, shown in an overlay on top of
your video, game or meeting. Everything runs locally with free models; paid
services (Claude, DeepL, OpenAI, Deepgram) are optional fallbacks.

It adapts to the machine: a hardware profile picks the models that fit
(NVIDIA GPU, AMD/Intel GPU, Apple Silicon, or CPU only), skips what doesn't
fit in memory, and switches to lighter models if it can't keep up.
See [PLAN.md](PLAN.md) for the design and [EVAL.md](EVAL.md) for measured quality.

## Setup (Windows)

Requires Python 3.10+.

```bash
python -m venv .venv
```

```bash
.venv\Scripts\pip install -e ".[nvidia]"
```

Leave out `[nvidia]` on computers without an NVIDIA GPU; it adds the CUDA
libraries (~1.3 GB) that Whisper needs on the GPU. With the CUDA 12 toolkit
already installed, `pip install -e . nvidia-cudnn-cu12==9.10.2.21` is enough.

Models download on first use to `%LOCALAPPDATA%\tfv\models`
(`TFV_MODELS_DIR` changes it). On a machine like the development laptop
(RTX 3050 4 GB) the first start downloads about 3 GB.

## The desktop app

```bash
.venv\Scripts\python -m tfv
```

A subtitle bar appears near the bottom of the screen and a 譯 icon in the
system tray (green = listening, orange = loading, grey = paused). The first
start takes a few minutes while models download; the bar shows progress.

- **Languages**: tray menu → *Spoken language* and *Translate into* (or *Off*
  for captions only). The spoken language must be chosen; there is no
  automatic detection yet.
- **Move / resize**: unlock the overlay (tray menu or Ctrl+Alt+L), drag it,
  drag its corner; lock it again so clicks pass through to the video.
- **Hotkeys**: Ctrl+Alt+P pause · Ctrl+Alt+L lock/unlock · Ctrl+Alt+O original
  text on/off · Ctrl+Alt+H hide/show.
- **Settings** (tray menu or double-click the icon): capture device (the
  default follows speaker/earphones automatically), microphone input, models,
  API keys, overlay size/opacity, and live speed statistics.

Settings are saved in `%APPDATA%\tfv\settings.json`; the log is
`%APPDATA%\tfv\tfv.log`.

## Command line

```bash
.venv\Scripts\python -m tfv listen --from ja --to zh
```

Prints captions and translations in the terminal. Useful options:

| Option | Meaning |
|---|---|
| `--from zh\|en\|ja` / `--to zh\|en\|ja` | spoken language (required) / translate into |
| `--capture input` | use a microphone instead of the computer's output |
| `--audio-device NAME` | device index or part of its name (`tfv devices`, `tfv devices --inputs`) |
| `--file PATH [--realtime]` | a recording instead of live audio |
| `--profile NAME` | force a hardware profile (see below) |
| `--asr CHOICE` / `--mt CHOICE` | force a model, e.g. `--asr sensevoice-small`, `--mt nllb-600m`, `--mt claude`, `--mt none` |
| `--remote ws://HOST:8765` | run the models on another PC (see *Remote engine*) |
| `--no-adapt` | never switch to lighter models automatically |
| `-v` | per-line timings |

Other commands: `tfv hardware` (profile and which models fit), `tfv models`
(list/download/remove), `tfv keys` (API keys), `tfv serve` (remote engine),
`tfv eval` (quality measurements), `tfv devices`.

## How models are chosen

`python -m tfv hardware` shows the decision for the current machine. Profiles
live in [tfv/profiles.yaml](tfv/profiles.yaml); each stage is a list of options,
best first. The app uses the first option that fits in free memory, loads,
and benchmarks fast enough for live audio; while running it steps down when
recognition or translation falls behind.

| Profile | When | Recognition | Translation |
|---|---|---|---|
| gpu-high | NVIDIA ≥ 16 GB | Whisper large-v3 | TranslateGemma 12B → Qwen3.5 9B → … |
| gpu-mid | NVIDIA 8–12 GB | Whisper large-v3-turbo | Qwen3.5 9B → 4B → 2B |
| gpu-small | NVIDIA 4–6 GB | Whisper large-v3-turbo | Qwen3.5 4B → 2B (GPU) → 2B (CPU) → NLLB |
| vulkan-gpu | AMD / Intel Arc / NVIDIA without CUDA | SenseVoice (CPU) | Qwen3.5 4B/2B via Vulkan → NLLB |
| apple | Apple Silicon | SenseVoice (CPU) | Qwen3.5 via Metal → NLLB |
| cpu-high / cpu-low | no usable GPU | SenseVoice (CPU) | Qwen3.5 2B (CPU) / NLLB |
| cloud | only when chosen | OpenAI / Deepgram | DeepL / Claude / OpenAI |

Local LLM translation runs in llama.cpp's `llama-server` (downloaded
automatically, Vulkan build on Windows), started on the discrete GPU and sized
to the VRAM left after speech recognition.

## Paid services (optional)

```bash
.venv\Scripts\python -m tfv keys set claude
```

Stores a key in Windows Credential Manager (or set `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `DEEPL_AUTH_KEY`, `DEEPGRAM_API_KEY`). Then choose it:
`--mt claude` (default model `claude-opus-5`; e.g. `--mt claude:claude-haiku-4-5`
is cheaper and faster), `--mt deepl`, `--mt openai`, `--asr openai`,
`--asr deepgram`, or `--profile cloud`. Paid services are never used unless
chosen; audio or text is then sent to that provider. The settings window shows
an estimated cost for the session.

## Remote engine

A weak laptop can use a stronger PC's GPU on the same network.
On the strong PC:

```bash
.venv\Scripts\python -m tfv serve --from ja --to zh --host 0.0.0.0
```

It prints a token. On the laptop: `python -m tfv listen --from ja --to zh
--remote ws://PC-ADDRESS:8765 --token TOKEN`, or in the app enter the address
under Settings → Models → Remote engine and the token under Paid services →
remote. Traffic is not encrypted: use it on a trusted network only.

## Other platforms

The engine is cross-platform; only audio capture differs, and macOS/Linux
have not been tested yet.

- **macOS (Apple Silicon)**: install a loopback driver such as BlackHole, route
  output through it (Multi-Output Device), then `tfv listen --capture input
  --audio-device BlackHole`.
- **Linux**: capture a PulseAudio/PipeWire monitor source as an input device.

## Evaluation

```bash
.venv\Scripts\python -m tfv eval download
```

```bash
.venv\Scripts\python -m tfv eval run
```

Downloads 20 FLEURS clips per language and writes [EVAL.md](EVAL.md) with
recognition error rates (clean and noisy), translation chrF, speed and VRAM.

## Packaging

```bash
.venv\Scripts\pyinstaller --noconfirm packaging\tfv.spec
```

Builds `dist\tfv\` with `tfv-app.exe` (desktop app) and `tfv.exe` (command
line). Set `TFV_NVIDIA=1` first to bundle the CUDA libraries. Models are not
bundled.

## Troubleshooting

- **No captions**: something must be playing on the captured device;
  `tfv devices` shows which output is the default. WASAPI sends no audio while
  nothing plays.
- **Wrong or garbled text**: check the spoken language (`--from`). Speech in
  another language is transcribed badly.
- **Slow or skipped lines**: `tfv hardware` shows what fits; the app steps down
  automatically, or force a lighter option (`--asr sensevoice-small`, `--mt nllb-600m`).
- **CUDA errors**: install the `[nvidia]` extra; the cuDNN version must match
  the one bundled with ctranslate2 (see `pyproject.toml`).
- **A hotkey does nothing**: another program owns it; the log says which.

## Tests

```bash
.venv\Scripts\python -m pytest
```

## Model licenses

Whisper (MIT), SenseVoice (FunASR model license), Silero VAD (MIT),
Qwen3.5 (Apache-2.0), TranslateGemma (Gemma Terms of Use), Hunyuan-MT (Tencent
Hunyuan Community License, excludes EU/UK/South Korea), NLLB-200
(CC-BY-NC-4.0, non-commercial). `tfv models` lists the license of each.
