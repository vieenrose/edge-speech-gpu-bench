# cuDNN-free CUDA Conv for onnxruntime (sherpa-onnx) — PoC

Goal: a **cuDNN-free, cuBLAS-only CUDA path** for sherpa-onnx on the Jetson Nano gen1, so
the CUDA EP doesn't pull in cuDNN's **782 MB** (see `tensorrt-jetson-nano-gen1.md` and the
RAM table in `gen1-cuda-validation.md`). sherpa-onnx runs on onnxruntime, so this is a fork
of **onnxruntime's CUDA EP**, not of sherpa-onnx itself.

## What was done (PoC — validated)

`patches/onnxruntime-cudnn-free-conv.patch` (against onnxruntime `v1.23.1`) replaces
`cudnnConvolutionForward` for the `Conv` op with **im2col + cuBLAS GEMM**:

- **`conv_nocudnn.cu`** — a Caffe-style CUDA **im2col** kernel (NCHW) + a bias-add kernel,
  float/half/double launchers.
- **`conv.cc`** — a cuDNN-free branch at the top of `Conv::ComputeInternal`, gated by
  `-DORT_CUDA_NO_CUDNN_CONV`: computes the output shape with ORT's own `ConvAttributes`,
  then for each batch/group does **im2col → `cublasGemmHelper` → bias**. No cuDNN calls.
  Reuses ORT infrastructure (`cublasGemmHelper`, `GetScratchBuffer`, `GetCublasHandle`,
  `Consts`, `ConvAttributes`) so it's idiomatic.

Handles stride, padding, dilation, **groups**, **Conv1d** (mapped to H=1), and bias.

## Validation

A tiny multi-Conv ONNX (2D stride+pad, 2D dilation, grouped g=2, Conv1d, all with bias) run
through the rebuilt ORT, CUDA EP (cuDNN-free conv) vs CPU EP reference, same inputs:

```
  c2d_y    (2D stride+pad)  max|CPU-CUDA| = 7.2e-07
  c2dil_y  (2D dilation)    max|CPU-CUDA| = 9.5e-07
  cgrp_y   (grouped g=2)    max|CPU-CUDA| = 9.5e-07
  c1d_y    (Conv1d)         max|CPU-CUDA| = 2.4e-07
  OVERALL max|CPU-CUDA| = 9.5e-07  ->  PASS
```

Matches the reference to **float epsilon (~1e-6)** — the cuDNN-free conv is numerically correct
across every conv variant SenseVoice / melo8k / silero use.

## What remains for the full RAM win (Phase 2)

The PoC proves convolution works without cuDNN. To actually **drop cuDNN from the binary** (the
~782 MB saving), the CUDA EP's other cuDNN ops used by these models must also go cuDNN-free, then
cuDNN must be unlinked:

| Op | Used by | cuDNN-free approach |
|---|---|---|
| **Softmax** | SenseVoice attention | custom reduction kernel (simple) |
| **Pooling** | minor | custom max/avg kernel |
| **LSTM** | silero-VAD | cuBLAS gates + elementwise, or run on CPU EP |
| **ConvTranspose** | melo8k | col2im + cuBLAS (mirror of this conv) |

Then build the CUDA EP with cuDNN stubbed/unlinked and re-measure RSS (expect to approach the
RapidSpeech cuBLAS-only ~0.76 GB rather than the ~1.5–2 GB cuDNN path).

Reproduce: apply the patch to onnxruntime v1.23.1, build the CUDA EP with
`-DCMAKE_CXX_FLAGS=-DORT_CUDA_NO_CUDNN_CONV`, run the validator in `patches/` notes.
