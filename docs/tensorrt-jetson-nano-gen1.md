# TensorRT on Jetson Nano gen1 — feasibility, efficiency, memory

Can the deployment-target models (SenseVoice, X-ASR, melo8k, silero-VAD, TEN-VAD)
run via **TensorRT** on the Jetson Nano gen1, and is it worth it? Numbers below are
authoritative (from the L4T `r32.7` apt repo) + op analysis of the actual ONNX graphs.

## What's available

JetPack 4.6.1 ships **TensorRT 8.2.1** (L4T r32.7). Two access paths:
- **Via sherpa-onnx → ORT TensorRT EP.** The prebuilt ORT 1.11.0 already bundles
  `libonnxruntime_providers_tensorrt.so` + `tensorrt_provider_factory.h`, so sherpa-onnx
  can route models through TensorRT with no extra code. This is the realistic path.
- **Standalone TRT** (onnx→engine via the ONNX parser): more control, much more work.

## How hard, per model

cuDNN-requiring op counts come from the ONNX graphs (`Conv`/`LSTM`/`ConvTranspose`).

| Model | Ops of note | TRT difficulty | Why |
|---|---|---|---|
| **silero-VAD** | 18 Conv + 2 LSTM | **Moderate** | LSTM supported; hidden state must be plumbed as I/O bindings. Tiny — TRT overhead likely exceeds the A57 CPU. |
| **TEN-VAD** | small | **Moderate** | Same class; benefit questionable at this size. |
| **SenseVoice** | 70 Conv + attention | **Moderate** | Transformer parses in TRT 8.2; needs optimization profiles for dynamic audio length; the int8 model does **not** map to TRT-INT8 directly (TRT runs its own calibration from FP32 + a calibration set). |
| **melo8k** | 166 Conv + 3 ConvTranspose (VITS) | **Hard** | **Data-dependent output length** (durations predicted at runtime) + flow/coupling + possible stochastic ops fight TRT's static engine; likely needs graph surgery into static sub-engines. |
| **X-ASR** | streaming zipformer | **Hard** | Custom ops (TensorAsStrided/Stack/SimpleUpsample — the same ones that blocked ncnn-Vulkan) need **custom TRT plugins (C++)**, plus streaming cache-state management. Most TRT-hostile. |

## Efficiency

- **Compute-heavy supported models** (SenseVoice encoder, melo8k vocoder): a TRT engine is
  usually the **fastest** option on NVIDIA — fusion + FP16/INT8 + optimal tactic selection,
  often **1.5–3× over ORT-CUDA**.
- **Tiny models** (silero, TEN-VAD): TRT launch/engine overhead means GPU still likely
  **loses to the A57 CPU** — consistent with the VAD result in `results/BENCHMARK.md`.
- Engine build is **slow** (minutes/model) and **hardware + TRT-version locked** — engines must
  be built on/for sm_53 + TRT 8.2 and rebuilt if anything changes.

## Memory overhead — the key finding

| Component | Footprint |
|---|---|
| `libnvinfer.so.8.2.1` (TRT core) | **157 MB** (+ plugin/myelin → ~300–400 MB loaded) |
| **cuDNN + cuBLAS** | **+782 MB** — TRT 8.2 uses them as **default tactic sources** |
| CUDA context (Tegra) | ~400–600 MB |
| Engine + workspace | model-dependent (workspace cappable) |

**TensorRT does not inherently escape cuDNN.** By default it loads cuDNN/cuBLAS as tactic
sources, so the resident footprint is **~1.3–2.0 GB per model — comparable to ORT+cuDNN, not
lighter.** You can force TRT's own kernels and drop cuDNN with
`config->setTacticSources(~(kCUDNN | kCUBLAS))` (→ ~0.6–1.0 GB, RapidSpeech-class), but then
conv-heavy models may **lose tactics or fail to build** — fragile exactly where it matters
(melo8k's 166 convs).

## Verdict for this project

- TensorRT is the **fastest** for SenseVoice/melo8k *if* they build — but melo8k (VITS dynamic
  length) and X-ASR (custom plugins + streaming) are **hard**, and the **memory is no better than
  ORT** unless you fight cuDNN tactics off (risky on conv-heavy graphs).
- The **cuBLAS-only RapidSpeech path** (~0.76 GB measured, no engine build, runs on any sm_53, no
  version lock) stays the better fit for the Nano's RAM budget. TRT wins only on raw speed for the
  two big models, at high integration cost.
- The **cuDNN-free ORT CUDA EP** (see `gen1-cuda-validation.md` / the onnxruntime fork) is the
  middle ground: ORT's flexibility (all these models + dynamic shapes) without cuDNN's 782 MB.

See also: [`gen1-cuda-validation.md`](gen1-cuda-validation.md),
[`jetson-nano-gen1-feasibility.md`](jetson-nano-gen1-feasibility.md).
