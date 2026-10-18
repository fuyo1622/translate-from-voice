# Evaluation results

**What changed because of these numbers** (see `tfv/profiles.yaml`):
- Japanese is recognized by ReazonSpeech k2 on the CPU on every profile: on Japanese TV speech it
  made far fewer errors than Whisper turbo, Kotoba-Whisper and SenseVoice (see the last section),
  and the GPU stays free for the 4B translator.
- Live text: the utterance being spoken is recognized again every ~0.6 s; finished phrases are cut
  at 8 s (was 12 s), and Silero VAD's state is reset after each utterance, which halved the errors
  on a stream of TV clips (a new voice after a short pause went unheard for up to 2.5 s).
- Background music: the VAD judges a GTCRN-denoised copy of the audio (the recognizer still gets the
  original); with music as loud as the speech, missed lines fell from 8 of 30 to 1 (last section).
- Songs and loud music: on lines with music, the recognizer hears the voices with the music taken out
  (UVR MDX-Net in half precision on a GPU, via DirectML). Sung lyrics under music as loud as the
  voice: 30.5% → 21.6% errors (19.6% for the singing alone); music 5 dB louder: 51.4% → 23.2%. Song
  mode (a switch) also lets the VAD hear the separated voices: 81% of the singing found instead of 46%.
- SenseVoice (CPU) beat Whisper small (GPU) in every language, so it replaced it as the fallback.
- On 4 GB GPUs, Chinese/Japanese speech uses SenseVoice on the CPU so the GPU can hold a 4B
  translator (ja→zh chrF 24 vs 14 for 2B); English speech keeps Whisper turbo (much better in noise) + Qwen3.5-2B.
- Into Chinese: TranslateGemma-4B first; into English/Japanese: Qwen3.5-4B first.
- Found and fixed while measuring: llama-server crashed under memory pressure (RAM prompt cache,
  now disabled; the translator also restarts a dead server), and VRAM fit checks now use
  nvidia-smi because Vulkan's free-memory figure ignored Whisper's CUDA memory.

Caveats: 20 clips / 30 sentences per language, so differences under ~2 chrF or ~1% error are noise.
Translation speed was measured with the GPU otherwise idle.

Machine: 11th Gen Intel(R) Core(TM) i5-11300H @ 3.10GHz (4 cores), 16 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), CUDA ok  
Data: FLEURS test set (CC-BY-4.0), 20 clips per language; noisy = pink noise at 10 dB SNR.  
Recognition: CER for Chinese/Japanese, WER for English (lower is better). Translation: chrF against the FLEURS reference in the target language (higher is better), translating the reference transcripts so recognition errors don't count.

## Speech recognition

| Model | zh CER | zh noisy | en WER | en noisy | ja CER | ja noisy | s / utterance | × real time | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|
| whisper-large-v3-turbo@cuda | 9.1% | 10.3% | 5.3% | 5.5% | 4.2% | 3.7% | 0.71 | 0.06 | 980 |
| whisper-small@cuda | 16.7% | 15.5% | 7.3% | 9.6% | 11.1% | 14.8% | 0.65 | 0.06 | 296 |
| sensevoice-small@cpu | 9.7% | 10.4% | 7.3% | 11.2% | 6.7% | 6.6% | 0.78 | 0.07 | 0 |

## Translation

| Model | zh→en | zh→ja | en→zh | en→ja | ja→zh | ja→en | mean | s / line | VRAM MB |
|---|---|---|---|---|---|---|---|---|---|
| nllb-600m@cpu | 55.7 | 24.2 | 19.6 | 33.3 | 12.9 | 54.2 | **33.3** | 2.06 | 0 |
| qwen3.5-2b@gpu | 56.8 | 26.3 | 29.8 | 33.5 | 14.1 | 53.3 | **35.6** | 0.61 | 1344 |
| qwen3.5-4b@gpu | 60.3 | 30.4 | 29.0 | 36.4 | 21.5 | 58.7 | **39.4** | 1.71 | 2805 |
| translategemma-4b@gpu | 55.0 | 30.2 | 32.2 | 35.6 | 24.2 | 54.2 | **38.6** | 1.45 | 2722 |

## Japanese TV speech (ReazonSpeech)

`tfv eval run --set reazonspeech`: 200 clips spread over the ReazonSpeech test set (news, dramas,
variety, sports), so casual, fast and unclear speech is included; noisy = pink noise at 10 dB SNR.
CER, lower is better. Same machine as above.

| Model | CER | CER noisy | s / utterance | × real time | VRAM MB |
|---|---|---|---|---|---|
| reazonspeech-k2-v2@cpu | **10.1%** | **10.3%** | 0.63 | 0.11 | 0 (CPU) |
| whisper-large-v3-turbo@cuda | 12.3% | 17.3% | 0.68 | 0.12 | 981 |
| kotoba-whisper-v2.0@cuda | 13.0% | 15.6% | 0.64 | 0.11 | 798 |
| sensevoice-small@cpu | 24.6% | 24.2% | 0.38 | 0.07 | 0 (CPU) |

By clip length (clean), ReazonSpeech is best or tied everywhere; Whisper's extra errors on short
lines are mostly invented text (そうですね → なんでしょうかね), not formatting:

| Clips | ReazonSpeech | Whisper turbo | SenseVoice |
|---|---|---|---|
| short, < 3 s (69) | **17.3%** | 25.6% | 36.2% |
| 3–6 s (67) | 16.5% | 16.1% | 36.1% |
| long, > 6 s: narration, news (64) | **5.8%** | 7.6% | 17.2% |

Caveat: ReazonSpeech k2 was trained on other parts of this corpus, and on FLEURS read speech
(above) it is weaker than Whisper (ja CER 7.4% / noisy 8.1%; Kotoba-Whisper 5.2% / 6.6%).

### Live text latency

A minute of these clips played in real time through the engine (12 clips, 0.3–0.9 s pauses;
TranslateGemma-4B on the GPU for Chinese). Medians:

| | before (SenseVoice, whole lines) | now (ReazonSpeech, live text) |
|---|---|---|
| first text after a phrase starts | 4.1 s | **1.0 s** |
| live text behind the speech | – | 0.24 s |
| finished line after the phrase ends | 0.65 s | 0.68 s |
| translation after the phrase ends | 1.2 s | 1.25 s |
| CER over the whole stream | 39.0% | **11.2%** |

With no pauses between clips (nonstop talk), cutting phrases at 8 s instead of 12 s gave
14.7% vs 17.9% CER and the translation 1.5 s instead of 1.6 s after each phrase.

### Speech under background music

30 of the TV clips with pauses, music under all of it (FMA tracks, CC-BY: 2 instrumental, 4 with
singing), at a given speech-to-music level; the real segmenter and ReazonSpeech, offline. CER over
the stream, and how many of the 30 lines never became a caption:

| Music | none | GTCRN for the VAD only (default) | GTCRN for VAD and recognizer | Spleeter vocals |
|---|---|---|---|---|
| no music | 25.6% (1) | **24.7% (1)** | 24.9% (1) | – |
| instrumental, 10 dB | 31.4% (1) | **20.2% (0)** | 20.5% (0) | 29.5% (2) |
| instrumental, 5 dB | 30.7% (1) | **22.6% (0)** | 28.1% (0) | 33.5% (2) |
| instrumental, 0 dB | 49.3% (8) | **20.9% (1)** | 33.3% (1) | 38.1% (3) |
| with singing, 10 dB | 25.8% (1) | 28.8% (0) | 25.8% (0) | 30.2% (0) |
| with singing, 5 dB | 32.1% (1) | **25.8% (0)** | 29.3% (0) | 30.0% (1) |
| with singing, 0 dB | 41.2% (5) | **25.1% (0)** | 33.7% (0) | 34.9% (0) |

Silero VAD misses speech when music is loud; ReazonSpeech copes with the music once it gets the
utterance. So the VAD now judges a copy cleaned by GTCRN (0.5 MB, streaming, ~15% of one core, 20 ms
delay) and the recognizer gets the original. Confirmed live through the whole engine (20 clips, music as
loud as the speech): instrumental 24.5% → 14.2% CER, with singing 24.8% → 16.1% and no lines lost
(no music: 15.8%), latency unchanged.

Music louder than the speech (−5 dB), and two alternatives: DPDFNet (a 10 MB denoiser) instead of
GTCRN, and giving the recognizer a 50/50 blend of the denoised and the original audio. Same 30 lines:

| Music | none | GTCRN, VAD only (default) | GTCRN, recognizer hears a blend | DPDFNet, VAD only | DPDFNet, blend |
|---|---|---|---|---|---|
| instrumental, 0 dB | 49.3% (8) | **20.9% (1)** | 23.5% (1) | 23.3% (1) | 23.3% (1) |
| instrumental, −5 dB | 100% (30) | **31.6% (2)** | 31.9% (2) | 31.9% (1) | 33.0% (1) |
| with singing, 0 dB | 41.2% (5) | 25.1% (0) | **24.0% (0)** | 34.2% (2) | 31.6% (2) |
| with singing, −5 dB | 75.8% (18) | 42.3% (2) | 38.4% (2) | 40.7% (2) | **34.2% (2)** |

None is better across the board, and the differences are within what 30 lines can tell apart,
so the default stays GTCRN for the VAD only.

### Sung lyrics, and the music filter

48 lines of Japanese songs from GTSinger (CC BY-NC-SA 4.0; a soprano and a tenor, about 25 songs),
each also spoken by the same singer. Each line goes whole to the recognizer, with instrumental music
(FMA) under the singing at a given level. Error rate on the kana readings, since lyrics often use
unusual kanji (蒼い and 青い read the same):

| Recognizer | spoken | sung | sung, music 0 dB | sung, music −5 dB |
|---|---|---|---|---|
| ReazonSpeech k2 (default) | **16.9%** | 19.6% | 30.5% | 51.4% |
| SenseVoice-Small | 19.1% | 20.8% | 30.8% | 45.9% |
| Whisper large-v3 turbo | 20.3% | **19.0%** | 35.2% (7 lines empty) | 55.5% (14 empty) |
| Kotoba-Whisper v2.0 | 18.6% | 20.0% | **28.9%** | **42.2%** |

Singing alone is recognized nearly as well as speech; the accompaniment is what hurts. With the
music taken out first by UVR MDX-Net Voc_FT (64 MB; the smaller UVR_MDXNET_9482 in brackets):

| Recognizer | sung, music 0 dB | sung, music −5 dB |
|---|---|---|
| ReazonSpeech k2 | **21.6%** (27.7%) | **23.2%** (33.1%) |
| SenseVoice-Small | 23.9% (28.9%) | 25.3% (38.5%) |
| Whisper large-v3 turbo | 21.0% (22.5%) | 22.2% (33.1%) |
| Kotoba-Whisper v2.0 | 21.9% (30.0%) | 25.1% (34.2%) |

On the 30 TV clips (character error rate): music at the speech's level 23.7% → 20.2%, 5 dB louder
29.5% → 24.9%. Clean audio is unaffected: spoken lyrics 16.9% → 16.8%, sung 19.6% → 18.9%, clean TV
speech 19.8% → 20.9% (one or two characters in 30 lines). Spleeter (above) was worse than nothing.

So tfv runs Voc_FT on the lines in which the VAD's denoiser removed at least 4 dB (it removed at
least 5 dB from every line under music above, and 0–2 dB from most clean speech). It needs a
graphics chip: through DirectML one model run (5.8 s of audio) takes 0.44 s on the RTX 3050 and
1.5 s on Intel Iris Xe graphics, but 6–10 s on the 4-core CPU, so it stays off without a GPU.
It runs in half precision, converted once on first use: the same output (vocals 20.4 dB cleaner),
half the graphics memory. That is what lets it share the 4 GB card with TranslateGemma 4B. In full
precision, translation slowed from 0.35 s to 9.6 s per line while it separated, and to 6.6 s even
when the two took turns (memory was being swapped); in half precision: 0.36 s, and 0.56 s while it
separated nonstop. In the app, a line with music is shown about 1.3 s after it ends.

### Song mode

The denoised copy still hides much of the singing from the VAD under loud music. In song mode (a
switch, off by default) the VAD and the recognizer hear the voices separated continuously while the
denoiser removes 8 dB or more: a model run every 0.75 s over the last 5.5 s, keeping the newest
0.75 s, with 0.25 s of audio after it. The GTSinger song (10 lines, music 5 dB louder), offline:

| VAD hears | singing found | errors (kana) |
|---|---|---|
| the denoised mix (default) | 46% | 55% |
| song mode: 0.75 s steps, 0.25 s ahead, VAD threshold 0.3 | **81%** | **33%** |
| the same with threshold 0.5 | 64% | 41% |
| 1 s steps, 0.3 s ahead, threshold 0.3 | 85% | 35% |
| 0.6 s steps, 0.1 s ahead, threshold 0.3 | 79% | 35% |

The same song played through the speakers into the real app: filter off, 3 of 10 lines found (79%
errors); the filter on whole lines, 6 of 10 (65%); song mode, 10 of 10 (33%), with the end of each
line shown about 2.1 s after it was sung. Song mode keeps the GPU busy ~60% of the time while music
plays; translation stayed at about 0.5 s per line.

On the user's own recording (2 minutes of a live performance, vocals mixed loud) song mode was no
clear gain: of six sung lines, two came out better, two worse and two the same, with fewer junk
captions (3 instead of 5). Finding the singing matters less there; recognizing sung words is the
harder part. So song mode stays a switch, off by default.
