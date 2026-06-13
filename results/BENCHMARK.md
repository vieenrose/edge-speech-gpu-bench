# Edge speech engines — GPU/CPU benchmark on NVIDIA GB10

**Host:** DGX Spark — NVIDIA **GB10** (Grace-Blackwell, aarch64, sm_121, integrated GPU, UMA),
CUDA 13.0, 128 GB unified memory, driver 580.159.03. Research comparison box, **not** the
deployment target (Jetson Nano gen1, sm_53, CUDA 10.2).

**Engines × backends**

| Engine | Backends measured | Models |
|---|---|---|
| sherpa-onnx (ONNX Runtime) | CPU (CUDA = blocked, see below) | SenseVoice, X-ASR, melo8k, silero-VAD, TEN-VAD |
| RapidSpeech.cpp (ggml) | **CPU / CUDA / Vulkan** | SenseVoice, melo8k, silero-VAD |
| sensevoice.cpp (ggml) | CPU / CUDA | SenseVoice |
| sherpa-ncnn (ncnn) | CPU (Vulkan = blocked) | streaming zipformer |

**Method.** Warm RTF = steady-state in a **persistent context** (model + backend init once, median
of iters ≥2). This matters enormously: the ggml engines are **launch-bound** — the first
inference compiles the GPU graph/pipeline (CUDA ~0.2 s, Vulkan up to ~1.7 s for melo8k), then warm
calls are 6–250× faster. A per-process CLI re-pays that every call. GPU memory: `N/A` (GB10 UMA;
`nvidia-smi` reports N/A).

## RTF — warm steady-state (lower is better)

| Model (task) | sherpa-onnx CPU | RapidSpeech CPU | RapidSpeech CUDA | RapidSpeech Vulkan | sensevoice.cpp CPU | sensevoice.cpp CUDA | sherpa-ncnn CPU |
|---|---|---|---|---|---|---|---|
| **SenseVoice** STT | **0.0102** | 0.0739 | **0.0031** | **0.0031** | 0.044 | 0.0058 | — |
| **X-ASR-480ms** STT | 0.0666 | — | — | — | — | — | — |
| **streaming zipformer** STT | — | — | — | — | — | — | 0.0535 |
| **melo8k** TTS | 0.0209 | 0.0657 | **0.0107** | 0.0245 | — | — | — |
| **silero-VAD** | **0.0023** | 0.0055 | 0.0057 | 0.0413 | — | — | — |
| **TEN-VAD** | 0.0029 | — | — | — | — | — | — |

*(cold RTF and per-row notes in `full_rtf.csv`. X-ASR and the sherpa-ncnn zipformer are different
streaming-ASR models, both sherpa/ncnn-only; not cross-comparable to each other.)*

## SenseVoice STT — engine + backend matrix (whole 30 s clip, warm)

| | CPU | CUDA | Vulkan |
|---|---|---|---|
| sherpa-onnx (ONNX int8) | **0.0102** | ⛔ blocked | — |
| RapidSpeech.cpp (ggml) | 0.0739 | **0.0031** | 0.0031 |
| sensevoice.cpp (ggml) | 0.044 | 0.0058 | CPU-fallback¹ |

¹ sensevoice.cpp selects the GPU via the ggml device registry; the GB10's Vulkan device isn't
picked up there (works for CUDA), so it silently runs CPU. RapidSpeech calls
`ggml_backend_vk_init` directly and does use Vulkan.

## Accuracy

| Metric | Result |
|---|---|
| SenseVoice — sherpa↔RapidSpeech agreement (CER) | **0.015** (one quant homophone) |
| melo8k TTS — round-trip ASR CER | sherpa 0.083, **RapidSpeech 0.000** |
| silero-VAD / TEN-VAD frame-agreement vs TEN labels | 0.799 / 0.704 |
| X-ASR / sherpa-ncnn (zh-en streaming) | clean on zh/en; ja/ko out-of-domain |

## Verdict

- **CPU:** sherpa-onnx wins decisively (SenseVoice 6.7×, melo8k 3.9×, VAD 2.4× vs RapidSpeech).
  Among ggml engines, sensevoice.cpp is ~1.7× faster than RapidSpeech on CPU SenseVoice.
- **CUDA flips the verdict for RapidSpeech.** SenseVoice 0.0031 and melo8k 0.0107 both beat
  sherpa-**CPU** (0.0102 / 0.0209). The ggml-CUDA thesis lands — *if* you keep a persistent
  process (launch-bound).
- **CUDA vs Vulkan (ggml, same GPU):** identical on the bigger model (SenseVoice 0.0031 both);
  CUDA ~2.3× better on melo8k TTS (0.0107 vs 0.0245); Vulkan *loses to CPU* on tiny silero-VAD
  (0.041 vs 0.0055) — Vulkan's per-dispatch overhead dominates sub-millisecond models.
- **sensevoice.cpp GPU needed a fix:** its shipped (old) ggml made GPU *slower* than CPU (0.051);
  modern ggml + a `ggml_cont` fix → 0.0058 (8.8×). Same author as RapidSpeech, so this backports
  their newer engine's ggml. (PR: vieenrose/SenseVoice.cpp#1.)

## Blockers (see `BLOCKERS.md`)
- **sherpa-onnx CUDA:** no prebuilt path on GB10/CUDA-13; from-source onnxruntime build (CUDA-13
  CUTLASS/cccl fixes applied) — the CUDA provider was still compiling when this was written.
- **sherpa-ncnn Vulkan:** needs ncnn built with glslang; the glslang dev toolchain wasn't
  installable in the no-sudo sandbox.
- **sensevoice.cpp Vulkan:** device-registry selection doesn't pick the Vulkan GB10 (CPU fallback).
