# tfv — translate from voice

Live captions for whatever your computer is playing (speaker or earphones),
for Traditional Chinese, English and Japanese. See [PLAN.md](PLAN.md) for the
full design and roadmap.

**Current status: Phase 1.** System audio is captured, cut into sentences and
transcribed in the terminal. Translation (Phase 2) and the subtitle overlay
(Phase 4) are next. You choose the spoken language with `--from`; automatic
language detection is postponed.

## Setup (Windows)

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

For GPU recognition on an NVIDIA card, also install the CUDA libraries:

```bash
pip install -e ".[nvidia]"
```

This installs cuDNN 9 and cuBLAS 12 as pip packages (about 1.3 GB). If the
CUDA 12 toolkit is already installed, only cuDNN is needed:
`pip install nvidia-cudnn-cu12==9.10.2.21`. The cuDNN version must match the
one bundled with `ctranslate2` (see `pyproject.toml`).

Models are downloaded on first use to `%LOCALAPPDATA%\tfv\models`. Set
`TFV_MODELS_DIR` to use another folder.

| Model | Size | Used for |
|---|---|---|
| Silero VAD v6.2 | 2 MB | finding speech in the audio |
| SenseVoice-Small (int8) | 240 MB | recognition on CPU |
| Whisper large-v3-turbo | 1.6 GB | recognition on NVIDIA GPU |

## Usage

```bash
python -m tfv devices
```

Lists the output devices that can be captured; `*` marks the Windows default.

```bash
python -m tfv listen --from ja
```

Transcribes Japanese audio playing on the default output. Stop with Ctrl+C.

| Option | Meaning |
|---|---|
| `--from zh\|en\|ja` | language being spoken (required) |
| `--audio-device NAME` | capture another output, by index or part of its name |
| `--asr auto\|faster-whisper\|sensevoice` | recognizer; `auto` = Whisper on an NVIDIA GPU, else SenseVoice on CPU |
| `--compute auto\|cuda\|cpu` | where Whisper runs |
| `--model SIZE\|DIR` | Whisper model, e.g. `small`, `large-v3-turbo` |
| `--file PATH [--realtime]` | transcribe a file instead of live audio |
| `--duration SECONDS` | stop automatically |
| `--log PATH` | also append transcripts to a file |
| `-v` | show per-sentence timings |

A line ending in `…` was cut at the 12 s maximum length (`--max-segment-s`);
the sentence continues on the next line.

## Tests

```bash
python -m pytest
```

## Known limitations

- The capture device is chosen at start. If you switch between speaker and
  earphones while running, restart.
- Speech in a language other than `--from` is transcribed badly (Whisper
  often translates it into the `--from` language).
- Whisper can invent text over music. Common phrases such as "Thanks for
  watching" are filtered out.
