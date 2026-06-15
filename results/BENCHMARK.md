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

## RTF — warm steady-state (lower is better) — **all measured on GB10**

> Every RTF in this table (and the two below) is from the **GB10** (sm_121 native for CPU/Vulkan;
> sm_53 SASS PTX-JIT'd for the CUDA columns). For **real Jetson Nano gen1** numbers see the
> [Real Jetson Nano gen1 section](#real-jetson-nano-gen1-sm_53-maxwell--the-device-not-the-gb10-proxy-2026-06-15) below.

| Model (task, all **GB10**) | sherpa-onnx CPU | RapidSpeech CPU | RapidSpeech CUDA | RapidSpeech Vulkan |
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

> ⚠️ The synth-time (ms) and peak-RSS (MB) below are **GB10** (sm_53 SASS PTX-JIT'd onto Blackwell),
> not real Nano. Real-device equivalents are in the Nano section at the bottom of this file.

| Backend (same tokens, all **GB10**) | warm synth | output audio | RTF | peak RSS |
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

---

## Real Jetson Nano gen1 (sm_53 Maxwell) — the device, not the GB10 proxy (2026-06-15)

All numbers above are **GB10** (sm_53 SASS JIT'd onto a Blackwell GPU) — relative only. The deployment
device is now measured. Engines **x86-cross-compiled** for CUDA-10.2 / gcc-8.3, run on real sm_53 under
the live product stack. TTS numbers are **single-shot / per-call** (RapidSpeech is **launch-bound** —
~1.5 s wall ≈ model-load + first-graph compile); the SenseVoice STT numbers are **warm steady-state**
(load amortized over a multi-segment clip). The headline: the GPU verdict is **model-size dependent** —
ggml-CUDA wins the heavy STT and loses the small TTS.

### RTF (lower is better) — real Nano gen1

| Model (task) | sherpa CPU | sherpa cuDNN-free CUDA | RapidSpeech ggml-CPU | RapidSpeech ggml-CUDA |
|---|---|---|---|---|
| **melo8k** TTS (small vocoder) | **0.457** | 0.701 | 3.60 (cold) | 0.9–1.2 (cold) |
| **matcha8k** TTS (CFM) | 0.347 @t4 | 0.502 | 0.507 (warm) | **0.184 (warm) — fastest matcha** |
| **SenseVoice** STT (heavy attn) | 0.589 int8 | OOM / empty† (int8) | 0.556 q5 (warm) | **0.104 q5 (warm) — fastest** |
| **X-ASR** int8 STT | ~0.7–0.85 (modern-ORT)‡ | won't load (ConvInteger)‡ | onnx-only | onnx-only |

*(**warm** = persistent process, load/JIT amortized (matcha_e2e_test best-of-3; SenseVoice 2nd VAD seg);
**cold** = single-shot `rs-tts` CLI, launch-bound ~1.5 s. matcha-on-RapidSpeech uses the
`matcha_e2e_test`+`MATCHA_IDS` token-injection harness; the converted gguf is published at
[Luigi/matcha-zh-tw-en-8k-gguf](https://huggingface.co/Luigi/matcha-zh-tw-en-8k-gguf), real-utterance
validated (2.40 s @ 8 kHz, matches ONNX reference).)*

### Peak RSS / RAM fit on the 4 GB Nano

| Run | Peak RSS | Min sys-avail | Fits? |
|---|---|---|---|
| RapidSpeech **ggml-CUDA** melo8k | **455 MB** | 1747 MB | ✅ |
| RapidSpeech **ggml-CUDA** matcha8k | (well under budget) | — | ✅ |
| RapidSpeech **ggml-CUDA** SenseVoice q5 | **456 MB** | 1654 MB | ✅ |
| sherpa-onnx **ORT-CUDA** SenseVoice int8 | ~1.07 GB + arena (~2.8 GB demand) | 350 MB | ❌ OOM-killed |
| sherpa-onnx CPU (melo8k / SenseVoice) | 305 / 577 MB | — | ✅ |

### Verdict — compute-density × warm-vs-cold (not a flat "GPU never helps")

- **GB10's *speed* verdict survives for WARM compute-heavy graphs.** SenseVoice STT (50-layer SAN-M):
  **ggml-CUDA RTF 0.104 vs 0.556/0.589 CPU**; matcha8k TTS (3-step-ODE CFM): **ggml-CUDA 0.184 vs
  0.347/0.507 CPU** (and < sherpa cuDNN-free CUDA 0.502). Both beat every CPU path *and* fit where
  ORT-CUDA OOMs. It only **fails** for the tiny melo8k vocoder and for cold single-shot CLI calls
  (launch-bound on the weak Maxwell — 128 cores, ~0.5 TFLOP, no tensor cores). GB10-as-Nano overstates
  *absolute* GPU RTF ~50–100×, but the qualitative "GPU helps the warm heavy model" result holds.
  (Two earlier drafts were wrong: "no GPU beats CPU" was a melo8k-only artifact — SenseVoice *and*
  matcha on ggml-CUDA refute it; "matcha not benchmarkable" was wrong too — see the harness note above.)
- **GB10's *RAM* verdict holds and is now measured.** ggml-CUDA melo8k 455 MB / SenseVoice 456 MB — both
  **fit**; sherpa ORT-CUDA **OOM-kills (~2.8 GB)**. **ggml-CUDA is the only engine that can use the
  Maxwell GPU on gen1 at all.**
- **cuDNN-free EP bug exposed only without cuDNN:** `CudnnFilterDescriptor` ctor called
  `cudnnCreateFilterDescriptor` → SIGABRT at session init on the real Nano (GB10 had cuDNN present).
  Fixed (guard under `ORT_CUDA_NO_CUDNN`), pushed to the onnxruntime fork.
- **† SenseVoice int8 + sherpa ORT-CUDA** is a dead end (281 `MatMulInteger` → CPU EP, fragmented graph,
  empty output) — *but* RapidSpeech runs SenseVoice **q5 on ggml-CUDA** fine (the row above). fp32
  SenseVoice (937 MB) doesn't fit 4 GB on either engine.
- **‡ Version scissor:** Nano Maxwell caps ORT at 1.11, but int8 X-ASR needs `ConvInteger` (ORT ≥ ~1.14)
  → won't load. You get the GPU **or** modern int8 models, never both (sherpa side).

**Bottom line:** for the **deployed** one-off-TTS + X-ASR pipeline, **sherpa-onnx CPU is optimal** (X-ASR
is ONNX-only; one-off calls are launch-bound). But the GPU is *not* useless on gen1: for a **persistent
server with heavy graphs**, **RapidSpeech ggml-CUDA wins warm — SenseVoice 0.104 and matcha8k 0.184**,
beating every CPU path, and is the only engine that fits the Maxwell GPU at all.
