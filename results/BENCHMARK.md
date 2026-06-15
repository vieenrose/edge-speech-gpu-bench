# Edge speech engines — GPU/CPU benchmark on NVIDIA GB10

**Host:** DGX Spark — NVIDIA **GB10** (Grace-Blackwell, aarch64, sm_121, integrated GPU, UMA),
CUDA 13.0, 128 GB unified memory, driver 580.159.03. Research comparison box, **not** the
deployment target (Jetson Nano gen1, sm_53, CUDA 10.2).

**Engines × backends**

| Engine | Backends measured | Models |
|---|---|---|
| sherpa-onnx (ONNX Runtime) | CPU (CUDA = blocked, see below) | SenseVoice, X-ASR, melo8k, silero-VAD, TEN-VAD |
| RapidSpeech.cpp (ggml) | **CPU / CUDA / Vulkan** | SenseVoice, melo8k, silero-VAD |

**Method.** Warm RTF = steady-state in a **persistent context** (model + backend init once, median
of iters ≥2). This matters enormously: RapidSpeech's ggml backend is **launch-bound** — the first
inference compiles the GPU graph/pipeline (CUDA ~0.2 s, Vulkan up to ~1.7 s for melo8k), then warm
calls are 6–250× faster. A per-process CLI re-pays that every call. GPU memory: `N/A` (GB10 UMA;
`nvidia-smi` reports N/A).

## RTF — warm steady-state (lower is better)

| Model (task) | sherpa-onnx CPU | RapidSpeech CPU | RapidSpeech CUDA | RapidSpeech Vulkan |
|---|---|---|---|---|
| **SenseVoice** STT | **0.0102** | 0.0739 | **0.0031** | **0.0031** |
| **X-ASR-480ms** STT | 0.0666 | — | — | — |
| **melo8k** TTS | 0.0209 | 0.0657 | **0.0107** | 0.0245 |
| **silero-VAD** | **0.0023** | 0.0055 | 0.0057 | 0.0413 |
| **TEN-VAD** | 0.0029 | — | — | — |

*(cold RTF and per-row notes in `full_rtf.csv`. X-ASR is sherpa-only.)*

## RapidSpeech.cpp — CUDA vs Vulkan (same ggml engine, same GB10 GPU)

| | CPU | CUDA | Vulkan |
|---|---|---|---|
| SenseVoice STT | 0.0739 | **0.0031** | **0.0031** |
| melo8k TTS | 0.0657 | **0.0107** | 0.0245 |
| silero-VAD | 0.0055 | 0.0057 | 0.0413 |

Vulkan **ties CUDA** on the bigger SenseVoice workload, lags ~2.3× on melo8k TTS, and *loses to
CPU* on tiny silero-VAD — Vulkan's per-dispatch overhead dominates sub-millisecond models.

## Matcha-TTS — same-utterance (ggml vs sherpa-onnx cuDNN-free CUDA)

Matcha-TTS ([Luigi/matcha-zh-tw-en-8k](https://huggingface.co/Luigi/matcha-zh-tw-en-8k), CFM acoustic +
Vocos vocoder, 8 kHz) ported from scratch to RapidSpeech.cpp ggml. Benchmarked **apples-to-apples**: the
identical phoneme tokens are fed to both engines. Sentence `這個星期的研究進度。` → sherpa's matcha frontend
emits `[2069, 614, 1886, 1397, 420, 1927, 829, 814, 489, 5]` (dumped with `--debug=1`), injected into the
ggml e2e harness (`MATCHA_IDS=…`) so it synthesizes the exact same tokens. Warm = best-of-3 in a persistent
ggml context / sherpa's per-call generation timer (model already loaded). sherpa on the cuDNN-free ORT 1.11
CUDA EP (`--provider=cuda`, `GraphOptimizationLevel=1`, `--tts-silence-scale=1`).

| Backend (same tokens) | warm synth | output audio | RTF | peak RSS |
|---|---|---|---|---|
| RapidSpeech ggml **CPU** | 87 ms | 2.40 s | 0.036 | **154 MB** |
| RapidSpeech ggml **CUDA** | **26 ms** | 2.40 s | **0.011** | 579 MB |
| sherpa-onnx **CPU** | 64 ms | 1.88 s | 0.034 | 201 MB |
| sherpa-onnx **cuDNN-free CUDA** | 55 ms | 1.88 s | 0.029 | 667 MB |

- For identical token input the **ggml CUDA path (26 ms) is ~2.1× faster** than sherpa-onnx's cuDNN-free
  CUDA (55 ms), at **~1.2× lower RSS** (579 vs 667 MB). ggml CPU is leanest at **154 MB**. On **CPU** the
  usual order holds (sherpa 64 ms vs ggml 87 ms — ORT's CPU kernels win), so it's specifically **CUDA**
  where ggml *flips* the small-model verdict: the deep 3-step-ODE CFM decoder favors ggml's hand-written
  graph.
- **Output-length caveat (honest):** the two runtimes disagree on synthesized length for the same tokens —
  ggml's duration regulator matches host ONNX Runtime (~150 mel frames → 2.40 s); the deployment ORT 1.11
  runtime sherpa links yields ~118 frames → 1.88 s. Both vocoders read `n_fft=512 hop=128`, so this is an
  ORT-version / duration-regulator divergence, not a vocoder bug. Each RTF is vs its own output; the
  directly-comparable figure is the **warm-synth wall-clock for identical input**.
- ggml CUDA pays a one-time ~57 s sm_53→sm_121 PTX-JIT (cached in `CUDA_CACHE_PATH`), excluded from warm.

## Accuracy

| Metric | Result |
|---|---|
| SenseVoice — sherpa↔RapidSpeech agreement (CER) | **0.015** (one quant homophone) |
| melo8k TTS — round-trip ASR CER | sherpa 0.083, **RapidSpeech 0.000** |
| silero-VAD / TEN-VAD frame-agreement vs TEN labels | 0.799 / 0.704 |
| X-ASR (zh-en streaming) | clean on zh/en; ja/ko out-of-domain |

## Verdict

- **CPU:** sherpa-onnx wins decisively (SenseVoice 6.7×, melo8k 3.9×, VAD 2.4× vs RapidSpeech) —
  ONNX Runtime's CPU kernels are simply faster for these small speech models.
- **CUDA flips the verdict for RapidSpeech.** SenseVoice 0.0031 and melo8k 0.0107 both beat
  sherpa-**CPU** (0.0102 / 0.0209). The ggml-CUDA thesis lands — *if* you keep a persistent
  process (launch-bound).
- **CUDA vs Vulkan (ggml, same GPU):** identical on the bigger model (SenseVoice 0.0031 both);
  CUDA ~2.3× better on melo8k TTS (0.0107 vs 0.0245); Vulkan *loses to CPU* on tiny silero-VAD
  (0.041 vs 0.0055) — Vulkan's per-dispatch overhead dominates sub-millisecond models.

## Blockers (see `BLOCKERS.md`)
- **sherpa-onnx CUDA:** no prebuilt path on GB10/CUDA-13; from-source onnxruntime build (CUDA-13
  CUTLASS/cccl fixes applied) — the CUDA provider was still compiling when this was written.
