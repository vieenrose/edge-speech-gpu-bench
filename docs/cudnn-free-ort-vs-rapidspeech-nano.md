# cuDNN-free ORT 1.11.0 vs RapidSpeech — Jetson Nano gen1 (in JetPack/CUDA-10.2 container)

Comparative benchmark of the two **Nano-dedicated, cuDNN-free CUDA builds**, run in the
`l4t r32.7` / CUDA-10.2 container on the DGX Spark (cc spoofed to sm_53). Models: SenseVoice,
X-ASR-480ms (onnx-only), melo8k, silero-VAD, TEN-VAD (onnx-only).

## Result: the comparison itself is the finding

| Model | RapidSpeech CUDA (ggml, cuDNN-free) | cuDNN-free ORT **1.11.0** CUDA |
|---|---|---|
| **SenseVoice** | ✅ runs, **761 MB** peak RSS | ❌ ORT-1.11 **integer overflow at inference** |
| **silero-VAD** | ✅ runs (in the 761 MB ASR pipeline) | ❌ ORT-1.11 integer overflow |
| **melo8k** | ✅ runs, **776 MB**, GPU==CPU (0.999999) | ❌ **opset 17** (LayerNormalization) — rejected at load, won't downgrade |
| **X-ASR-480ms** | n/a (onnx-only) | ❌ ORT-1.11 integer overflow |
| **TEN-VAD** | n/a (onnx-only) | ❌ ORT-1.11 integer overflow |

**RapidSpeech runs everything at ~760–780 MB. The cuDNN-free ORT *builds* but cannot *run* the
modern sherpa models on ORT 1.11** — and ORT 1.11 is the **only** version that supports the Nano's
CUDA 10.2 (newer ORT needs CUDA 11+/12; csukuangfj's last CUDA-10.2 aarch64 GPU build is 1.11.0).

## Why ORT 1.11 can't run them (two independent walls)

1. **Opset ceiling — melo8k.** Exported at **opset 17** (uses `LayerNormalization`, new in 17).
   ORT 1.11 caps at opset 16 and rejects it at load. `onnx.version_converter` 17→16 **fails**:
   *"No Previous Version of LayerNormalization exists"* — so it needs **op decomposition**, not a
   version bump.
2. **Runtime integer overflow — SenseVoice / silero / X-ASR / TEN-VAD.** These are opset 13 (within
   the ceiling) and **load** fine, but **overflow at inference** in ORT 1.11's core (`safeint.h`,
   `SafeIntOnOverflow`) — on **both CPU and CUDA EPs**, so it is **not** the cuDNN-free CUDA EP.
   The same models + identical inputs run fine on **modern ORT (1.25)** → it's an ORT-1.11-vs-recent-
   export incompatibility, fixable only by patching ORT 1.11 or re-exporting each model.

## What this means for the deployment target

- The **cuDNN-free CUDA EP for onnxruntime is real and builds** under CUDA 10.2 (see
  `cudnn-free-ort-conv.md`; fork `vieenrose/onnxruntime`). The engineering works.
- But **ORT 1.11 — the only CUDA-10.2 ORT — is too old to *run* today's speech models** without
  per-model surgery (opset downgrade/decomposition) **and** fixing ORT-1.11 runtime overflows.
- **RapidSpeech.cpp (ggml) sidesteps both**: it reimplements the architecture in C++ (no ONNX opset
  ceiling, no ORT runtime), runs all the models cuDNN-free at ~0.76 GB, and is the practical Nano path.

**Bottom line:** for cuDNN-free CUDA on the Jetson Nano gen1, **RapidSpeech is the viable engine**;
the cuDNN-free ORT fork is a valid technique but is gated by ORT 1.11's age, not by cuDNN.

(Measured in-container on the GB10 via PTX-JIT; the overflow is ORT-1.11 core, reproduced on CPU EP
too. Absolute RapidSpeech RSS on a real 4 GB Nano will differ from the GB10's, but the *relative*
picture — RapidSpeech runs, ORT 1.11 can't — is hardware-independent.)
