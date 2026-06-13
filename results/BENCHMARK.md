# sherpa-onnx (ONNX) vs RapidSpeech.cpp (ggml) — GPU/CPU benchmark

**Host:** DGX Spark — NVIDIA GB10 (Grace-Blackwell, aarch64, sm_121), CUDA 13.0, 128 GB unified
memory, driver 580.159.03. Research comparison box, **not** the deployment target (Jetson Nano
gen1, sm_53, CUDA 10.2). Goal = relative sherpa-vs-RapidSpeech GPU picture.

- **sherpa-onnx:** pip `sherpa-onnx` 1.13.2 (CPU) for the CPU columns; CUDA via a from-source
  onnxruntime build (see *sherpa CUDA* note + `build/sherpa.md`).
- **RapidSpeech.cpp:** `vieenrose/RapidSpeech.cpp@jetson-nano-gen1` built on modern ggml with
  `-DRS_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121` (see `build/rapidspeech.md`).
- **Models:** SenseVoice-small (STT, both), X-ASR-480ms streaming zipformer int8 (STT,
  sherpa-only), melo8k / openvoice2-melo8k-zh (TTS, both), silero-VAD (both), TEN-VAD
  (sherpa-only).
- **RTF** measured on a 36.85 s zh/en/ja/ko/yue clip (STT/VAD) or per-utterance (TTS). **cold** =
  first invocation (graph warm-up); **warm** = median of 5 steady-state iterations.
- **GPU memory:** `N/A` — the GB10's unified memory does not report `memory.used` via
  `nvidia-smi` (returns N/A).

> **Measurement note (important).** All "warm" RTF is **steady-state inside one persistent
> context** (model + backend initialised once; median of iters ≥2). RapidSpeech's ggml CUDA path
> is heavily **launch-bound**: the *first* inference compiles the CUDA graph (~220 ms for melo8k),
> then every subsequent call is ~6–10× faster. A naïve harness that spawns one process per
> utterance (as a CLI does) re-pays that one-time warmup every call and makes CUDA look slow — an
> earlier draft of this table did exactly that for TTS. The numbers below use a persistent
> context for every framework, so they are apples-to-apples.

## RTF — warm (steady-state), lower is better

| Model (task) | sherpa CPU | sherpa CUDA | RapidSpeech CPU | RapidSpeech CUDA |
|---|---|---|---|---|
| SenseVoice (STT)   | 0.0102 | ⏳ | 0.0739 | **0.0031** |
| X-ASR-480ms (STT)  | 0.0666 | ⏳ | — *(sherpa-only)* | — |
| melo8k (TTS)       | 0.0209 | ⏳ | 0.0657 | **0.0107** |
| silero-VAD         | **0.0023** | ⏳ | 0.0055 | 0.0057 |
| TEN-VAD            | 0.0029 | ⏳ | — *(sherpa-only)* | — |

### SenseVoice STT — three engines (whole 30 s clip, in-process warm)

`sensevoice.cpp` (lovemefan) is a 3rd ggml SenseVoice inferencer. Note **RapidSpeech.cpp and
sensevoice.cpp are by the same author** — RapidSpeech is the newer multi-model successor and
already ships modern ggml; sensevoice.cpp shipped a pre-CUDA-graph ggml.

| Engine (SenseVoice-small q5_k) | CPU | CUDA |
|---|---|---|
| sherpa-onnx (ONNX, int8) | **0.0102** | ⏳ |
| RapidSpeech.cpp (ggml, modern) | 0.0739 | **0.0031** |
| sensevoice.cpp — **shipped** (old ggml) | 0.044 | 0.051 *(GPU slower than CPU!)* |
| sensevoice.cpp — **fixed** (modern ggml + cont) | 0.044 | **0.0058** |

The shipped sensevoice.cpp GPU (0.051) is *slower* than its CPU — its old ggml starves the GB10
(no CUDA-graph batching, GPU util 6–50 %). Bumping it to modern ggml + a `ggml_cont` on the
silero-VAD LSTM transpose (else modern ggml-cuda aborts: `mmvf.cu: stride_col_y%2==0`) yields
**0.0058 — 8.8× faster**, in line with RapidSpeech. Pushed as
[vieenrose/SenseVoice.cpp#1](https://github.com/vieenrose/SenseVoice.cpp/pull/1).

## RTF — cold (incl. graph warm-up)

| Model | sherpa CPU | RapidSpeech CPU | RapidSpeech CUDA |
|---|---|---|---|
| SenseVoice  | 0.0115 | 0.0680 | 0.0390 |
| X-ASR-480ms | 0.0666 | — | — |
| melo8k      | 0.0196 | 0.0802 | 0.1101 |
| silero-VAD  | 0.0024 | 0.0059 | 0.0078 |
| TEN-VAD     | 0.0030 | — | — |

RapidSpeech's CUDA path is **launch-bound** — huge first-graph cost, fast warm steady-state:
SenseVoice cold 0.039 → warm 0.0065 (6×), melo8k cold 0.110 → warm 0.0107 (10×). sherpa-onnx
shows almost no cold/warm gap (its ONNX graph is pre-compiled at session creation). This is the
single most important behavioural difference between the two engines, and it's why the
deployment pattern matters: RapidSpeech-CUDA only pays off with a **persistent server process**,
never a per-utterance CLI.

## Accuracy

| Metric | Result |
|---|---|
| SenseVoice STT — sherpa↔RapidSpeech agreement (CER) | **0.015** (near-identical; only diff is a int8-vs-q5_k quant homophone 开放/开饭) |
| X-ASR-480ms (zh-en) | clean on zh/en; ja/ko out-of-domain (expected) |
| melo8k TTS — round-trip ASR CER (synth→SenseVoice) | sherpa 0.083, **RapidSpeech 0.000** |
| silero-VAD — frame-agreement vs TEN testset (30 wavs) | 0.799 |
| TEN-VAD — frame-agreement vs TEN testset | 0.704 |

## Verdict (preliminary — sherpa CUDA pending)

On **CPU**, sherpa-onnx wins decisively, as on x86: SenseVoice **6.7×** faster (0.0102 vs
0.0685), melo8k **3.9×** (0.0209 vs 0.0810), silero-VAD **2.4×**. RapidSpeech's ggml CPU path is
simply slower for these small speech models.

**GPU flips STT for RapidSpeech.** RapidSpeech-CUDA SenseVoice warm RTF **0.0065** beats
sherpa-onnx **CPU** (0.0102) by ~1.6× — and beats its own CPU by 10×. This is the ggml-CUDA
thesis landing for ASR.

**GPU also flips TTS for RapidSpeech — once warm.** melo8k on CUDA warm **0.0107** beats
sherpa-CPU (0.0209) by ~2× and its own CPU (0.066) by 6×. The catch is the launch cost: the
first synthesis is ~0.11 (220 ms graph warmup), so GPU only wins when the context is reused
across utterances (a persistent TTS server) — exactly RapidSpeech's intended deployment.
*(An earlier draft reported 0.133 here; that measured one OS process per utterance, charging the
one-time warmup to every call. Corrected to in-process steady-state.)*

**VAD: too small to matter.** silero is far below real-time on every backend; sherpa-onnx CPU
(0.0023) is fastest, and RapidSpeech-CUDA (0.0057) is statistically tied with its own CPU
(0.0055) — a per-512-sample model has no compute to offload, so the GPU neither helps nor hurts.

The decisive sherpa-onnx-CUDA numbers depend on the from-source onnxruntime build (CUDA 13 +
sm_121); status in `build/sherpa.md` / `BLOCKERS.md`.
