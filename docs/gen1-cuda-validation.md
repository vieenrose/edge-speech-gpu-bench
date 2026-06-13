# Jetson Nano gen1 — RapidSpeech.cpp CUDA: validation & fix

The deployment target is the **Jetson Nano gen1** (sm_53 Maxwell, JetPack 4.6.1,
CUDA 10.2), not the GB10 research box. This records the work that made
RapidSpeech.cpp's CUDA backend actually build and run on that toolchain, and how
it was validated without the device in hand.

Delivered on the fork: **`vieenrose/RapidSpeech.cpp@jetson-nano-gen1`** (commit `7e802ce`)
— `patches/ggml-cuda-10.2-sm53.patch`, `scripts/build_jetson_nano_gen1_native.sh`,
`docs/CUDA_10.2_JETSON_NANO.md`, ggml pinned to `b2a092a7`.

## The functional bug

ggml's **batched** cuBLAS matmul (`ggml_cuda_mul_mat_batched_cublas`) hardcoded
`CUBLAS_COMPUTE_16F` + `CUBLAS_GEMM_DEFAULT_TENSOR_OP` for every non-AMD GPU.
**Tensor-op GEMM needs tensor cores (sm_70+).** The Nano gen1 is **sm_53 Maxwell —
no tensor cores**, so the first ASR/TTS matmul aborts with
`CUBLAS_STATUS_NOT_SUPPORTED`. (The non-batched path was already guarded `cc>=VOLTA`
and fell back to `cublasSgemm`; only the batched path was missing the guard.)

**Fix:** a pre-Volta NVIDIA branch (`cc < 700`) → FP32 compute + non-tensor
`CUBLAS_GEMM_DEFAULT`, mirroring the existing AMD/CDNA fallback. On real sm_53 this
path is taken automatically; MMQ (quantized matmul) is also auto-skipped because it
needs DP4A (sm_61+), so ggml dequantizes and uses cuBLAS.

## CUDA-10.2 build patch set

| Patch | Why |
|---|---|
| pre-Volta batched-cuBLAS → FP32/non-tensor | sm_53 has no tensor cores (the bug) |
| NEON `_x2`/`_x4` load inlines for gcc<9 | gcc-8 `arm_neon.h` lacks `vld1q_*_x4` |
| no-op `__builtin_assume` | gcc lacks the builtin |
| **gcc-8** host compiler | gcc-7.5 (JetPack default) lacks `<charconv>` |
| `cuda_bf16.h` stub | CUDA 10.2 has no bf16 header (bf16 unused on sm_53) |
| `CUDA_R_16BF` / `CUBLAS_COMPUTE_*` `-D` back-defines | CUDA-11+ enums |
| `CMAKE_CUDA_STANDARD=14` | nvcc 10.2 caps device code at C++14 |

## Validation (in `dustynv/l4t-pytorch:r32.7.1`, genuine nvcc 10.2 + real CUDA-10.2 cuBLAS)

To exercise the exact sm_53 code paths on the GB10, the patch adds two **test-only,
env-gated** hooks (no effect unless set; never set on a real Nano):
`RS_FORCE_CC=530` (clamps the detected compute capability so the whole ggml dispatch
behaves as sm_53) and `RS_GEMM_NO_TENSOR=1`.

| Check | Result |
|---|---|
| Compiles under nvcc 10.2 / gcc-8 / C++14 / sm_53 | ✅ genuine sm_53 SASS |
| SenseVoice on GPU | ✅ `开放时间早上9点至下午5点。` (correct), EXIT 0 |
| melo8k on GPU vs CPU | ✅ bit-equivalent, **corr 0.999999**, EXIT 0 |
| Model actually runs on GPU | ✅ scheduler dump: **2470/2470 encoder nodes on `CUDA0`** (every MUL_MAT); only VAD on CPU (by design) |

## What is NOT answered here: speed

GB10 GPU timings are meaningless for the Nano on two counts: (1) cuBLAS 10.2 has no
native sm_53 SASS, so every kernel **JIT-compiles to sm_121 on first use** (one
SenseVoice encode measured ~59 s, essentially all JIT); (2) Blackwell ≠ Maxwell.

So "is CUDA faster than CPU-only on the Nano?" is **unresolved until the real device**.
Directional expectation, from the sm_53 architecture (~256 GFLOPS FP32 GPU vs
~20–40 GFLOPS NEON CPU, but no tensor cores / no MMQ → dequant + FP32 GEMM, unified
LPDDR4): **SenseVoice likely a modest GPU win; melo8k a toss-up; silero-VAD stays on
CPU** (it loses to CPU even on the GB10). The build is ready to settle it with a
`--gpu true/false` A/B on the device.

See also: [`jetson-nano-gen1-feasibility.md`](jetson-nano-gen1-feasibility.md).
