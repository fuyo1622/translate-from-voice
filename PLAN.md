# Translate-from-Voice: Survey & Plan

Goal: live-translate whatever the computer is playing (speaker / earphone output) between
**Traditional Chinese (zh-TW), English, Japanese**, shown as an on-screen subtitle overlay.
v1 uses free/open-source models locally; a paid backend is a drop-in fallback.

**Deployment goal:** the final product runs on *many* devices — from a CPU-only laptop to a
gaming PC with a large GPU — and **picks models/engines automatically based on the device's
resources**, adapting at runtime if it can't keep up.

Primary platform: **Windows 10/11** (x64). macOS/Linux are designed-for but come later
(only the audio-capture layer and GPU engine differ).

Dev machine: Windows 11, Python 3.10, RTX 3050 Laptop (4 GB VRAM) — this is the "GPU-small"
profile below, not the only target.

---

## 1. Survey of existing approaches

### 1.1 Ready-made tools (worth trying before/while building)
| Tool | How it works | Limits |
|---|---|---|
| Windows 11 **Live Captions** | Built-in, captions system audio. Live *translation* only on Copilot+ PCs, and mainly → English | Hardware-gated; no zh-TW/ja targets |
| **LiveCaptions-Translator** (GitHub, open source) | Reads Windows Live Captions text, sends to a translator (Google/DeepL/Ollama/OpenAI…) | Recognition quality depends on Live Captions' language packs; ASR not controllable |
| **Speech Translate** (GitHub, Dadangdut33) | Whisper/faster-whisper on loopback audio + translation APIs, Tkinter UI | Close to what we want; good reference implementation |
| **WhisperLive / whisper_streaming / SimulStreaming** | Streaming Whisper servers with LocalAgreement policy for low-latency partials | ASR only; WhisperLive's client/server design is a good reference for remote mode |
| **Buzz**, **RealtimeSTT** | Whisper desktop app / Python realtime STT library | Mostly transcription; translation limited |
| **sherpa-onnx** (k2-fsa) | Portable ONNX runtime for ASR/VAD (Whisper, SenseVoice, Paraformer…) on Windows/macOS/Linux/Android, CPU-first | Great for low-end/portable profiles |
| Meta **SeamlessStreaming** | End-to-end streaming speech→text translation | Heavy, CC-BY-NC license, zh-TW output needs post-conversion |
| Chrome Live Caption | Browser audio only | Translation limited, browser-only |

Conclusion: building our own pipeline is justified — it gives control over language restriction,
Traditional Chinese output, per-device model selection, and swapping backends.

### 1.2 Pipeline building blocks

**A. Capturing system audio (loopback)** — OS-specific, isolated behind an adapter
| OS | Method | Library |
|---|---|---|
| Windows | **WASAPI loopback** of the default output device (speaker or earphone); no drivers | `PyAudioWPatch` (or `soundcard`) |
| macOS 13+ | ScreenCaptureKit audio capture (or BlackHole virtual device) | small native helper / `pyobjc` |
| Linux | PulseAudio/PipeWire "monitor" source | `soundcard` / `sounddevice` |

**B. Voice activity detection** — **Silero VAD** (MIT, tiny, CPU, ONNX) on every profile.

**C. Speech recognition (ASR), open source**
| Model | zh / en / ja | Cost | Best engine per hardware |
|---|---|---|---|
| **SenseVoice-Small** | excellent zh, good ja/en | ~230M params, very fast even on CPU | sherpa-onnx (CPU, any OS) |
| Whisper **small / medium** | ok / good / ok | 0.5–1.5 GB | faster-whisper int8 (CPU), whisper.cpp |
| Whisper **large-v3-turbo** | good / excellent / good | ~1.6 GB int8 | faster-whisper (NVIDIA), whisper.cpp Vulkan (AMD/Intel), Metal/MLX (Apple) |
| Whisper **large-v3** | best | ~3 GB int8 | same as above, needs ≥6 GB VRAM comfortably |
| kotoba-whisper | ja-specialized | ~1.5 GB | optional add-on |

Whisper often outputs Simplified Chinese → **OpenCC `s2twp`** post-processing.
**v1: the user chooses the spoken language** (`--from`); automatic detection is postponed
(notes for re-adding it are under Phase 1 in §5).

**D. Translation (MT), open source**
| Model | Quality | Cost | Engine |
|---|---|---|---|
| **NLLB-200 distilled 600M** | decent, literal | ~0.6 GB int8, CPU OK | CTranslate2 |
| NLLB-200 1.3B / 3.3B | better | 1.3–3.3 GB | CTranslate2 |
| **Qwen3-1.7B / 4B** Instruct (Q4) | natural, context-aware | 1.2 / 2.5 GB | llama.cpp / Ollama |
| **Qwen3-8B, Hunyuan-MT-7B** (Q4) | best open quality | ~5 GB | llama.cpp / Ollama |
| Qwen3-14B+ (Q4) | top-end | ~9 GB+ | llama.cpp / Ollama |

NLLB is CC-BY-NC (personal use OK; replace or go paid if this ever becomes commercial).
LLM translators get the **previous 2–3 sentences as context** — fixes pronouns and fragmented
segments.

**E. Display** — always-on-top, click-through, semi-transparent subtitle window (PySide6 —
works on Windows/macOS/Linux).

### 1.3 Paid options (backup, and the "thin device" profile)
| Stage | Options | Notes |
|---|---|---|
| ASR | OpenAI `gpt-4o-transcribe` / `gpt-4o-mini-transcribe`, Deepgram Nova-3, Azure Speech, Google Chirp | Better on noisy/music/overlapping speech |
| MT | **DeepL API** (free tier ~500k chars/month; `ZH-HANT`, `JA`, `EN`), Google Translate, LLMs (Claude Haiku 4.5, GPT-4o-mini class) | DeepL cheapest good; LLM most natural |
| End-to-end streaming | **Soniox** real-time translation, Azure Speech Translation, OpenAI Realtime API | Best latency/quality; audio leaves the device; cost scales with hours |

Verify current pricing/features at implementation time (this survey reflects knowledge as of mid-2026).

---

## 2. Architecture

```
┌───────────────┐   ┌──────────────────────────── Engine ───────────────────────────┐   ┌───────────┐
│ Capture       │   │ resample 16k → Silero VAD → ASR backend → OpenCC → Translator │   │ Overlay UI│
│ adapter (OS)  │──►│            ▲                    ▲                  ▲          │──►│ (PySide6) │
└───────────────┘   │            └──── Resource Manager / Profile selector ─────────┘   └───────────┘
                    └────────────────────────────────────────────────────────────────┘
```

Three layers with clean seams, so each can vary per device:

1. **Capture adapter** (per OS) → yields 16 kHz mono float32 frames.
2. **Engine** — VAD, ASR, translation. Backends behind two interfaces:
   - `ASRBackend.transcribe(audio) -> (text, language)` (language is fixed by the user in v1)
     impls: `faster_whisper`, `whisper_cpp`, `sherpa_onnx` (SenseVoice), `mlx_whisper`, `openai`, `deepgram`
   - `Translator.translate(text, src, tgt, context) -> str`
     impls: `ctranslate2_nllb`, `llama_cpp`/`ollama`, `deepl`, `claude`, `openai`
   - Optional `EndToEndBackend` (Soniox/Azure) that replaces both.
3. **UI** — overlay + settings window (shows detected hardware, chosen profile, live latency).

Each stage runs in its own thread with bounded queues (capture never blocks; if ASR falls
behind, the queue depth is the signal to downgrade — see §3.3).

### 2.1 Deployment modes
| Mode | Where models run | Use case |
|---|---|---|
| **Local** (default) | on this device | Most PCs |
| **Remote engine** | device streams audio (Opus over WebSocket) to an engine on a stronger PC in the LAN, gets text back | Weak laptop + home desktop with GPU |
| **Cloud** | paid APIs | Very weak devices, or best quality |

Local mode is just the engine and client in one process; remote mode reuses the same engine as
a server (`python -m tfv.server`). Design the engine API as messages from day one so this split
is cheap later.

---

## 3. Resource-adaptive model selection

### 3.1 Hardware probe (at startup, cached)
- **GPU**: vendor + VRAM free/total — NVIDIA via `pynvml`/CUDA; AMD/Intel via DXGI (Windows) or
  Vulkan enumeration; Apple Silicon via `platform` + unified memory size.
- **CPU**: physical cores, AVX2/AVX-512 support; **RAM** total/available.
- **Power state**: on battery → prefer a lower profile (optional setting).
- **Network/API keys** available → whether cloud profiles are allowed.

### 3.2 Profiles (tiers)
| Profile | Detected when | ASR | Translation | Expected latency* |
|---|---|---|---|---|
| **cloud** | < 4 GB RAM, or user choice | paid ASR | DeepL / Claude | ~1 s |
| **cpu-low** | no usable GPU, < 12 GB RAM or ≤ 4 cores | SenseVoice-Small (sherpa-onnx) | NLLB-600M int8 (CPU) | 1–3 s |
| **cpu-high** | no usable GPU, ≥ 16 GB RAM, ≥ 6 cores | SenseVoice-Small or Whisper small int8 | Qwen3-1.7B Q4 (CPU) or NLLB-1.3B | 2–4 s |
| **gpu-small** | 4–6 GB VRAM (e.g. RTX 3050 4 GB) | Whisper large-v3-turbo int8 (GPU) | Qwen3-4B Q4 (GPU/partial CPU) or NLLB-600M CPU | 1–2 s |
| **gpu-mid** | 8–12 GB VRAM | Whisper large-v3 (GPU) | Qwen3-8B / Hunyuan-MT-7B Q4 (GPU) | ~1 s |
| **gpu-high** | ≥ 16 GB VRAM | Whisper large-v3 (GPU) | Qwen3-14B Q4 (GPU) | ~1 s |
| **apple** | Apple Silicon | mlx-whisper / whisper.cpp Metal (size by unified memory) | llama.cpp Metal Qwen3-4B/8B | ~1–2 s |

\* after end of utterance; to be validated by the benchmark in Phase 5.

GPU engine choice by vendor:
- **NVIDIA** → faster-whisper/CTranslate2 (CUDA) + llama.cpp CUDA.
- **AMD / Intel GPU** → whisper.cpp + llama.cpp with **Vulkan** backend (CTranslate2 is CUDA-only).
- **None** → sherpa-onnx / CTranslate2 CPU + llama.cpp CPU.

Profiles live in a data file (`profiles.yaml`), not code, so adding models or tuning thresholds
needs no code change. The user can always override profile or individual components.

### 3.3 Runtime adaptation
- **Startup micro-benchmark** (~5 s on a bundled clip): measure ASR real-time factor (RTF) and
  translation tokens/s; if RTF > 0.5, step down one profile before starting.
- **Live monitoring**: track queue depth and end-to-end latency; if latency > target (e.g. 4 s)
  for N segments, downgrade (smaller ASR → move translator to CPU → switch to NLLB → offer cloud).
  Upgrade back only on user action (avoid oscillation).
- **OOM guard**: catch CUDA/Vulkan out-of-memory on model load → retry next-lower option.
- **Budget VRAM jointly**: ASR + translator must fit in free VRAM minus a safety margin
  (~0.5 GB; other apps like games/browsers use VRAM too).

### 3.4 Model manager
- Models are **not bundled**; downloaded on first use of a profile (Hugging Face / GitHub
  release URLs), with SHA-256 check, resumable download, progress in UI, cache in
  `%LOCALAPPDATA%\tfv\models` (per-OS equivalent). Show disk size before downloading.
- Offline mode: import a pre-downloaded model folder.

---

## 4. Packaging & distribution
- Python app; package per OS with **PyInstaller** (Windows installer via Inno Setup later).
- Two builds to keep size down: **CPU build** (sherpa-onnx, CTranslate2 CPU, llama.cpp CPU/Vulkan)
  and **NVIDIA build** (adds CUDA runtime libs, ~1 GB larger). Or single build that downloads CUDA
  libs on demand if an NVIDIA GPU is found.
- Settings/profile in a user config file; API keys in the OS credential store (Windows Credential
  Manager via `keyring`), never in plain config.

---

## 5. Implementation phases

**Phase 0 — Setup ✅ done**
- Project skeleton (`tfv/capture`, `tfv/engine`, `tfv/engine/asr`), venv, deps; `tfv/engine/mt`,
  the hardware profiles and `tfv/ui` arrive with their phases.

**Phase 1 — Capture + ASR CLI ✅ done**
- WASAPI loopback → Silero VAD v6 (ONNX, no PyTorch) → ASR → terminal. Two ASR backends:
  faster-whisper (GPU) and SenseVoice via sherpa-onnx (CPU); `--asr auto` picks by CUDA availability.
- Spoken language chosen by the user (`--from zh|en|ja`); OpenCC `s2twp` for Chinese; filter for
  known Whisper hallucinations; CJK spacing and full-width punctuation cleanup.
- Findings on the dev machine (RTX 3050 Laptop 4 GB, i5-11300H, 16 GB):
  - Whisper large-v3-turbo (int8_float16, GPU): 0.45–0.65 s per sentence. The encoder always
    processes a 30 s window (~400 ms), so short sentences cost about the same as long ones.
  - SenseVoice-Small (int8, 4 CPU threads): 0.1–0.4 s per sentence. Close to Whisper on clean
    speech, but more homophone errors in Chinese (城市/程式).
  - Live runs were ~0.3 s slower per sentence than file runs, probably because the laptop GPU
    clocks down between sentences. Phase 3's benchmark should measure the live case.
  - ctranslate2 4.8.2 bundles the cuDNN 9.10.2 dispatcher DLL, and cuDNN rejects sub-libraries of
    another version, so the `nvidia` extra pins `nvidia-cudnn-cu12==9.10.2.21` with it.
- Notes for re-adding auto language detection later: restrict Whisper's detection to the three
  languages (argmax over their probabilities, counting `yue` as `zh`), and reuse the detection's
  encoder pass for the transcription. faster-whisper otherwise encodes twice, which doubled
  per-sentence time (0.9 s vs 0.5 s). SenseVoice reports no language probabilities, so its
  detection cannot be restricted.

**Phase 2 — Translation ✅ done**
- Model manager with a catalog (sizes, licenses), resumable checksum-verified downloads.
- NLLB-200 600M (CTranslate2, CPU); local LLMs as GGUF in a managed llama.cpp `llama-server`
  (prebuilt b11255, Vulkan build on Windows, picks the discrete GPU, `--fit` sizes it to free
  VRAM, per-run API key, killed with the parent via a Windows job object); any
  OpenAI-compatible server (Ollama, LM Studio). Newer models than the original plan were
  available by now: Qwen3.5 (0.8B–9B), TranslateGemma (4B/12B), Hunyuan-MT-7B.
- Findings: NLLB is poor for short Japanese lines (こんにちは → "您的位置："); Qwen3.5-2B is natural
  and fits beside Whisper turbo on 4 GB (both together ~2.5 GB VRAM), 0.2–1.9 s per line.

**Phase 3 — Resource manager ✅ done**
- Hardware probe, `profiles.yaml` with ladders of `<model>@<device>` rungs, fit checks against
  free VRAM/RAM, startup benchmark on bundled per-language clips, load-failure fallback, and a
  runtime monitor that steps down a rung when a stage falls behind (verified live: swap without
  a gap in captions).

**Phase 4 — Overlay UI ✅ done**
- PySide6 overlay (click-through when locked), tray menu, settings window, global hotkeys, and
  capture that follows the Windows default output when earphones are plugged in or out.

**Phase 5 — Evaluation ✅ done** — see [EVAL.md](EVAL.md)
- `tfv eval` on a FLEURS subset (same sentences in zh/en/ja, CC-BY): CER/WER clean and with
  10 dB pink noise, chrF in all six directions, speed and VRAM. Used to order the ladders.

**Phase 6 — Paid backends ✅ done**
- Claude (Anthropic SDK, `claude-opus-5-5` at low effort with server-side refusal fallbacks,
  model selectable), DeepL, OpenAI chat; OpenAI transcription and Deepgram for recognition;
  keys in the OS credential store; usage and cost estimate in the settings window.
  Soniox end-to-end streaming was not implemented (no key to verify the protocol against).

**Phase 7 — Portability ✅ done (partly untested)**
- Remote engine over WebSocket (`tfv serve` / `--remote`, token, one client), input-device
  capture for microphones / macOS BlackHole / Linux monitor sources, PyInstaller spec.
- Not verified on real hardware: AMD/Intel GPUs (the Vulkan runtime was tested on NVIDIA),
  macOS and Linux.

**Phase 8 — After the first real-video test ✅ done** (user feedback: Japanese recognition weak on
unclear or casual speech, lines too late, overlay couldn't be resized, tray icon not found; wants furigana)
- Japanese recognition: ReazonSpeech k2 (Zipformer, CPU) on every profile; measured on Japanese TV
  speech with `tfv eval run --set reazonspeech` (EVAL.md). The user prefers Japanese accuracy over
  translation quality when hardware forces a choice; here both improve, since the GPU stays free.
- Live text: the utterance being spoken is re-recognized every ~0.6 s (finished phrases first) and
  shown as "partial" captions with the id of the final one; phrases are cut at 8 s; Silero VAD is
  reset after each utterance (its state hid new voices after short pauses).
- Furigana (SudachiPy + SudachiDict core, readings over the kanji only) drawn by a custom widget.
- Overlay: resize by the side edges, right-click menu, Ctrl+Alt+S, a first-start notice (Windows 11
  hides new tray icons).

**Phase 9 — Second real-video test ✅ done** (feedback: speech with background music or singing
still missed; wants text size and the common functions directly on the bar)
- The VAD judges a GTCRN-denoised copy of the audio (sherpa-onnx streaming denoiser, 0.5 MB); the
  recognizer gets the original. Measured with FMA music under TV speech (EVAL.md); Spleeter vocal
  separation and denoising the recognizer's input were both worse.
- A toolbar on the unlocked bar (text size, furigana, original, languages, pause, lock, settings,
  quit), a faint unlock button at the corner of the locked bar, Ctrl+Alt+= / Ctrl+Alt+- for text size.
- Model choice still follows the hardware (profiles, fit checks, benchmarks, step-down while running);
  Japanese recognition is ReazonSpeech on the CPU everywhere, the translator scales with the GPU.

**Phase 10 — Third real-video test ✅ done** (feedback: talk over music is recognized but
inaccurate; sung lyrics wanted; switching ja→zh→ja→en stopped translating)
- Language switching: a new target retargets the loaded translator (instant), the translator stays
  loaded while translation is off, quick successive choices apply only the latest, and the bar keeps
  a "Loading…" notice up until it shows what is running ("Now: 日本語 → English").
- Music filter: on lines with music, UVR MDX-Net (Voc_FT) takes the music out before recognition.
  Measured on GTSinger lyrics and TV speech under FMA music (EVAL.md). It runs on a GPU through
  DirectML, in a helper process with a private ONNX Runtime DirectML build downloaded on first use
  (it can't be installed beside the regular onnxruntime package).
- "Save the last minute" (menu, Ctrl+Alt+K): the last 60 s of audio and the captions shown for it,
  saved locally, so failures in real videos can be measured.

**Phase 11 — Videos and songs with the music filter ✅ done** (feedback: talk over music better but
still often wrong; songs: the delay is not OK; no to 6 s more delay for complete lyrics)
- The filter runs in half precision (converted once on first use): it shares the 4 GB RTX 3050 with
  the translator without slowing it, so a line with music shows ~1.3 s after it ends instead of ~3 s
  on the integrated graphics, and live text uses the separated voices too.
- Song mode (a switch, off by default): continuous separation in 0.75 s steps feeds the VAD while
  music plays; 81% of the singing found instead of 46%, captions ~2.1 s after a line. On the user's
  own clip it was no clear gain, so the user compares it on real songs.

**Later (v2+)**: per-app capture (Windows process loopback API), automatic spoken-language
detection, the music filter on macOS/Linux (CoreML/CUDA), diarization, more languages, glossary UI;
for known songs, official lyrics looked up and lined up with what is heard (needs the user's OK to
query a lyrics service).

---

## 6. Risks
- **Hardware diversity** → combinatorial testing. Mitigate: few well-defined profiles, force-profile
  flags, and a test matrix of at least: CPU-only, NVIDIA 4 GB, NVIDIA 8 GB+, one AMD/Intel GPU.
- **CUDA/driver mismatch** on user machines → catch load errors and fall back to CPU/Vulkan engine
  automatically rather than crash.
- **Music/SFX** → Whisper hallucinations. Mitigate: VAD, `no_speech_threshold`,
  `condition_on_previous_text=False`, hallucination blocklist.
- **Wrong spoken language** (no detection in v1) → speech in another language is transcribed
  badly; Whisper tends to translate it into the chosen language. When detection returns,
  mixed-language content (ja with English words) will need smoothing over recent sentences.
- **Install size / first-run download** (models 0.5–10 GB) → download per profile only, show size.
- **Privacy** with cloud/remote modes → explicit opt-in, remote mode LAN-only by default.
- **Licenses**: NLLB and SeamlessStreaming are non-commercial; Qwen (Apache 2.0), Whisper (MIT),
  SenseVoice (model license — check) — revisit if distribution becomes commercial.
