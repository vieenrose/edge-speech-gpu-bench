# cuDNN-free CUDA Conv for onnxruntime (sherpa-onnx) — PoC

Goal: a **cuDNN-free, cuBLAS-only CUDA path** for sherpa-onnx on the Jetson Nano gen1, so
the CUDA EP doesn't pull in cuDNN's **782 MB** (see `tensorrt-jetson-nano-gen1.md` and the
RAM table in `gen1-cuda-validation.md`). sherpa-onnx runs on onnxruntime, so this is a fork
of **onnxruntime's CUDA EP**, not of sherpa-onnx itself.

## What was done (PoC — validated)

Lives as a fork branch: **[`vieenrose/onnxruntime@cudnn-free-cuda-ep`](https://github.com/vieenrose/onnxruntime/tree/cudnn-free-cuda-ep)**
(off onnxruntime `v1.23.1`, commit `af7c2c8`) — the **reference implementation on modern ORT
(CUDA 12/13)**. The deployable **Jetson Nano gen1 / CUDA-10.2** port is on the
**[`cudnn-free-cuda-jetson-nano-gen1`](https://github.com/vieenrose/onnxruntime/tree/cudnn-free-cuda-jetson-nano-gen1)**
branch (onnxruntime `v1.11.0`, the last release supporting the Nano's GPU). The same diff is mirrored here as
`patches/onnxruntime-cudnn-free-conv.patch`. It replaces `cudnnConvolutionForward`
for the `Conv` op with **im2col + cuBLAS GEMM**:

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

## Skipping cuDNN init (Phase 2 start) — and the measured RAM mechanism

`cudnnCreate` is called at CUDA-EP init (`cuda_stream_handle.cc`, `cuda_execution_provider.cc`)
**regardless of which ops run** — that handle creation alone faults cuDNN into RAM. Added
`-DORT_CUDA_NO_CUDNN` to skip it (keeping cuBLAS, unlike ORT's `USE_CUDA_MINIMAL` which drops
cuBLAS too). conv_test still passes with the handle skipped.

A microbenchmark (`patches/cudnn_ram_microbench.cu`) measures libcudnn resident pages directly
(cuDNN 9 on the GB10 host):

```
start:               libcudnn RSS =    200 KB   (mapped, unused)
after cuBLAS GEMM:    libcudnn RSS =    200 KB   <- cuDNN-free conv path: cuDNN NOT loaded
after cudnnCreate:    libcudnn RSS = 49,356 KB   <- handle init alone faults in 49 MB
after cuDNN conv fwd: libcudnn RSS = 130,200 KB  <- stock ORT Conv faults in 130 MB
```

So the cuBLAS-only path keeps cuDNN at **~0 (200 KB)**, while a single stock cuDNN conv faults
in **130 MB here** — and on the Nano's cuDNN 8 it's the **395 MB conv lib (~782 MB total)**.
Skipping `cudnnCreate` + the cuDNN-free Conv is exactly what avoids it. (Absolute *total* RSS on
the GB10 is dominated by the ~1 GB CUDA-13 context, which masks this in a gross RSS number — hence
the per-library measurement; on the Nano's small CUDA-10.2 context the cuDNN delta dominates.)

Fork: **[`vieenrose/onnxruntime@cudnn-free-cuda-ep`](https://github.com/vieenrose/onnxruntime/tree/cudnn-free-cuda-ep)**
(commit `885c0ad`); mirror `patches/onnxruntime-cudnn-free-cuda.patch`.

## Coverage: all three target models are now cuDNN-free

| Model | cuDNN-requiring ops | cuDNN-free handling | Validated |
|---|---|---|---|
| **SenseVoice** | Conv (70), Softmax | Conv → im2col+cuBLAS; Softmax → ORT's own kernels | Conv 9.5e-7 |
| **melo8k** | Conv (166), ConvTranspose (3) | Conv + ConvTranspose (col2im+cuBLAS) | Conv 9.5e-7, ConvT 4.8e-7 |
| **silero-VAD** | Conv (18), LSTM (2) | Conv → cuBLAS; LSTM → CPU EP (small, CPU-preferred) | LSTM route 1.4e-4 |

None use Pooling or BatchNorm, so with Conv + ConvTranspose cuDNN-free, Softmax already custom,
and RNN/GRU/LSTM routed to CPU under `-DORT_CUDA_NO_CUDNN`, **the CUDA EP runs all three with no
cuDNN op** — and `cudnnCreate` is skipped, so cuDNN is never initialized.

Build the cuDNN-free EP with a single switch: `-Donnxruntime_CUDA_NO_CUDNN=ON` (validated end-to-end: Conv 9.5e-7, ConvTranspose 4.8e-7, LSTM->CPU runs with cudnnCreate skipped).

## Status: the ops are done — what remains

Conv, ConvTranspose, Softmax (already custom), and RNN/GRU/LSTM (→ CPU) are all handled, so the
CUDA EP runs SenseVoice / melo8k / silero with **no cuDNN op** and `cudnnCreate` skipped — cuDNN is
never *loaded* into RAM (the measured win). Two items remain, both small / off-box:

1. **Unlink cuDNN from the binary.** Right now cuDNN is still *linked* (the unused RNN/BN/Pool/LRN
   op sources still reference it), just never loaded. To also drop it from disk/`NEEDED`, exclude
   those op sources under the option (à la `USE_CUDA_MINIMAL`, but keeping cuBLAS). Not required for
   the RAM win — RSS only counts faulted-in pages — but tidier.
2. **On-device gross-RSS measurement.** On the real Nano (small CUDA-10.2 context) the ~782 MB
   delta shows up directly in total RSS; on the GB10 its ~1 GB CUDA-13 context masks it, which is
   why the win was measured per-library (above).

Reproduce: apply `patches/onnxruntime-cudnn-free-cuda.patch` to onnxruntime v1.23.1, build the CUDA
EP with `-Donnxruntime_CUDA_NO_CUDNN=ON`, and run `patches/ort_conv_test.cc <model.onnx>` to compare
CUDA (cuDNN-free) vs CPU.
