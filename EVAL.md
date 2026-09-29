# Evaluation results

**What changed because of these numbers** (see `tfv/profiles.yaml`):
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
