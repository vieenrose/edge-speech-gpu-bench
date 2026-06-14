# cuDNN-free ORT 1.11.0 vs RapidSpeech — Jetson Nano gen1 (in JetPack/CUDA-10.2 container)

Comparative benchmark of the two **Nano-dedicated, cuDNN-free CUDA builds**, in the `l4t r32.7` /
CUDA-10.2 container on the DGX Spark (cc → sm_53 via PTX-JIT). Models: SenseVoice, X-ASR-480ms
(onnx-only), melo8k, silero-VAD, TEN-VAD (onnx-only).

> **Correction:** an earlier draft claimed ORT 1.11 "can't run" these models due to an integer
> overflow. That overflow was a **bug in the benchmark harness** (a dangling `Ort::TypeInfo` →
> garbage shape passed to `CreateTensor`), not ORT. With it fixed, the cuDNN-free ORT 1.11.0 runs
> most models. Numbers below are the corrected, real measurements.

## Results (peak RSS + warm time) — full 5/5 on cuDNN-free ORT 1.11.0

| Model | RapidSpeech CUDA (ggml) | cuDNN-free ORT **1.11.0** CUDA |
|---|---|---|
| **SenseVoice** | **761 MB** | ✅ 375 ms / **1215 MB** |
| **silero-VAD** | ~in ASR pipeline | ✅ 2.6 ms / **560 MB** |
| **melo8k** | **776 MB**, GPU==CPU | ✅ 45 ms / **732 MB** (via `model.opset16.onnx`) |
| **TEN-VAD** | n/a (onnx-only) | ✅ **0.4 ms / 559 MB** |
| **X-ASR-enc** (480ms) | n/a (onnx-only) | ✅ 323 ms / **1194 MB** |

All five now run cuDNN-free on ORT 1.11.0. Two issues that *looked* fatal turned out tractable:
- **melo8k opset-17** → decomposed `LayerNormalization` to opset 16 (`model.opset16.onnx`, numerically
  identical, on the HF repo).
- **TEN-VAD Conv→cuDNN** → the depthwise conv used TF-style "SAME"/asymmetric padding, which the
  initial cuDNN-free Conv skipped (`!post_slice` bail → cuDNN fallback). Fixed by computing per-dim
  pads with `ComputePadAndOutputShape(..., force_symmetric=false)` so im2col handles asymmetric pads
  directly. Conv numerics unchanged (max|CPU-CUDA| = 9.5e-7).
- (An earlier "integer overflow" that appeared to block everything was a **benchmark-harness bug** —
  a dangling `Ort::TypeInfo` — not ORT.)

*(RSS is GB10-inflated: the box's ~1 GB CUDA-13 context dominates and masks the cuDNN saving — see
`cudnn-free-ort-conv.md`. On a real 4 GB Nano with the small CUDA-10.2 context, absolute numbers are
much lower; the **relative** picture holds.)*

## What it took to get ORT models running cuDNN-free

ORT's CUDA EP is **deeply cuDNN-coupled** — many op families call cuDNN, not just Conv. Getting each
model to run cuDNN-free was whack-a-mole, each op routed/replaced under `-Donnxruntime_CUDA_NO_CUDNN`:

| cuDNN op family | How handled | Unblocked |
|---|---|---|
| Conv / ConvTranspose | im2col / col2im + cuBLAS (replaced) | SenseVoice |
| Softmax | ORT already has custom kernels | — |
| RNN / GRU / LSTM | route to CPU EP | silero |
| **Pooling** (`cudnnPoolingForward`) | route to CPU EP | (TEN-VAD partial) |
| **Reduce*** (`cudnnReduceTensor`) | route to CPU EP | **X-ASR** |
| **FusedConv** (conv+act fusion) | disable fusion (`ORT_ENABLE_BASIC`) | silero |
| Conv asym/TF-SAME pad (ComputePadAndOutputShape) | **TEN-VAD** | — |

## Takeaways for the deployment target

- The **cuDNN-free CUDA EP works** — SenseVoice, silero, and the X-ASR streaming zipformer all run
  on ORT 1.11.0 (CUDA 10.2) with **no cuDNN**.
- Both walls were cleared: **melo8k** runs via opset-16 LayerNorm decomposition; **TEN-VAD** runs
  after handling TF-SAME/asymmetric conv padding. But the per-op whack-a-mole (Conv/ConvTranspose/Pool/
  Reduce/RNN/FusedConv all needed work) shows how deeply cuDNN-coupled ORT's CUDA EP is.
- **On RAM, ggml/RapidSpeech wins clearly**: SenseVoice **761 MB** vs ORT **1445 MB** on the same box
  — even cuDNN-free, ORT's framework/arena overhead roughly doubles ggml's footprint.
- **RapidSpeech runs all three of its models** (SenseVoice/silero/melo8k) at ~0.76 GB, with **no opset
  ceiling and no per-op cuDNN surgery**.

**Bottom line:** the cuDNN-free ORT 1.11 fork is a viable, now-working technique (the overflow was a
harness bug), but for the Nano gen1 it's lower-RAM-efficient and higher-friction than RapidSpeech —
opset ceiling (melo8k), residual cuDNN ops (TEN-VAD), and ~2× the RAM. **RapidSpeech remains the
better cuDNN-free Nano engine**; the ORT fork is the right choice only when you specifically need
ORT's ecosystem and your models are opset-16-clean.
