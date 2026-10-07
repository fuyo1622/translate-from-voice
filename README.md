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

Models download on first use to the `models` folder inside the project
(next to `tfv.exe` in a packaged build)
(`TFV_MODELS_DIR` changes it). On a machine like the development laptop
(RTX 3050 4 GB) the first start downloads about 3 GB.

## The desktop app

```bash
.venv\Scripts\python -m tfv
```

A subtitle bar appears near the bottom of the screen and a 譯 icon in the
system tray (green = listening, orange = loading, grey = paused). Windows 11
hides new tray icons under **^**, so the bar has the same menu: unlock it
and right-click it. The first start takes a few minutes while models
download; the bar shows progress.

- **Live text**: the words appear while they are still being spoken (dimmer
  until the phrase is complete) and the translation follows each phrase.
- **Furigana**: Japanese lines show small hiragana readings above the kanji.
- **Background music**: the speech detector hears a denoised copy of the audio,
  so speech under music or noise isn't missed (Settings → Models turns it off).
- **Buttons on the bar**: while it is unlocked, a toolbar above the captions has
  A− / A+ (text size), ふりがな, 原文 (original text), the languages (e.g. 日→中),
  pause, lock, settings and quit. While locked, clicks go through to the video
  and a faint lock at the bar's corner unlocks it.
- **Move / resize**: unlock the bar, drag it to move it, drag its left or right
  edge to resize it, then lock it again.
- **Menu** (right-click the unlocked bar, or the tray icon): the same functions,
  plus hide. The spoken language must be chosen; there is no automatic detection yet.
- **Hotkeys**: Ctrl+Alt+P pause · Ctrl+Alt+L lock/unlock · Ctrl+Alt+O original
  text on/off · Ctrl+Alt+H hide/show · Ctrl+Alt+S settings · Ctrl+Alt+= / Ctrl+Alt+-
  text size.
- **Settings** (⚙ on the bar, the menu, Ctrl+Alt+S or double-click the icon):
  capture device (the default follows speaker/earphones automatically),
  microphone input, models, API keys, overlay size/opacity, live text, furigana,
  and live speed statistics.

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
| gpu-high | NVIDIA ≥ 16 GB | Japanese: ReazonSpeech (CPU); others: Whisper large-v3 | Qwen3.5 9B / TranslateGemma 12B → … |
| gpu-mid | NVIDIA 8–12 GB | Japanese: ReazonSpeech (CPU); others: Whisper large-v3-turbo | Qwen3.5 9B / TranslateGemma 12B → 4B → 2B |
| gpu-small | NVIDIA 4–6 GB | Japanese: ReazonSpeech (CPU); Chinese: SenseVoice (CPU); English: Whisper turbo (GPU) | into Chinese: TranslateGemma 4B; into English/Japanese: Qwen3.5 4B; → Qwen3.5 2B → NLLB |
| vulkan-gpu | AMD / Intel Arc / NVIDIA without CUDA | Japanese: ReazonSpeech; others: SenseVoice (CPU) | same 4B choices via Vulkan → 2B → NLLB |
| apple | Apple Silicon | Japanese: ReazonSpeech; others: SenseVoice (CPU) | same 4B choices via Metal → 2B → NLLB |
| cpu-high / cpu-low | no usable GPU | Japanese: ReazonSpeech; others: SenseVoice (CPU) | Qwen3.5 2B (CPU) / NLLB |
| cloud | only when chosen | OpenAI / Deepgram | DeepL / Claude / OpenAI |

Japanese is recognized by ReazonSpeech (trained on 35,000 hours of Japanese
TV) on the CPU: on TV speech it makes far fewer errors than Whisper, and it
leaves the GPU to the translator. On a 4 GB GPU, Whisper and a 4B translator
don't fit together, so Chinese speech is recognized on the CPU too (nearly as
accurate); for English speech, where Whisper is much better in noise, the app
keeps Whisper and a 2B translator. The measurements are in [EVAL.md](EVAL.md).
A 4B model runs only if it fits in the VRAM actually free, so a GPU-heavy game
or video player can push it down to 2B.

Local LLM translation runs in llama.cpp's `llama-server` (downloaded
automatically, Vulkan build on Windows), started on the discrete GPU and sized
to the VRAM left after speech recognition.

## Paid services (optional)

```bash
.venv\Scripts\python -m tfv keys set claude
```

Stores a key in Windows Credential Manager (or set `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `DEEPL_AUTH_KEY`, `DEEPGRAM_API_KEY`). Then choose it:
`--mt claude` (default model `claude-opus-5-5`; e.g. `--mt claude:claude-haiku-4-5`
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
FLEURS is read speech; for real Japanese TV speech (news, dramas, variety),
add `--set reazonspeech` to both commands (200 clips of the ReazonSpeech test
set, Japanese recognition only, report in `EVAL-reazonspeech.md`). The clips
are TV recordings: keep them on your computer.

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
- **`pip install` fails with "No such file or directory … PySide6\qml\…"**: the
  folder path is too long for Windows; use a shorter one (e.g. `D:\tfv`) or
  enable Windows long paths.
- **A hotkey does nothing**: another program owns it; the log says which.

## Tests

```bash
.venv\Scripts\python -m pytest
```

## Model licenses

Whisper (MIT), Kotoba-Whisper (MIT), ReazonSpeech k2 (Apache-2.0), SenseVoice
(FunASR model license), Silero VAD (MIT), Qwen3.5 (Apache-2.0), TranslateGemma
(Gemma Terms of Use), Hunyuan-MT (Tencent Hunyuan Community License, excludes
EU/UK/South Korea), NLLB-200 (CC-BY-NC-4.0, non-commercial). `tfv models` lists
the license of each. Furigana come from SudachiPy and SudachiDict (Apache-2.0).
