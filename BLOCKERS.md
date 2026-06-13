# Blockers / hard parts on the GB10 (Blackwell sm_121, aarch64, CUDA 13.0)

## 1. sherpa-onnx CUDA — no prebuilt path; from-source onnxruntime required
On this box, every prebuilt route to sherpa-onnx-CUDA fails:
- pip `sherpa-onnx` is CPU-only (compiled `SHERPA_ONNX_ENABLE_GPU=OFF`).
- The CUDA-13 aarch64 `onnxruntime-gpu` nightly loads the CUDA EP but aborts at execution with
  `cudaErrorUnsupportedPtxVersion` — built with CUDA 13.1 toolchain, no sm_121 SASS, PTX
  rejected by the 13.0 driver.
- sherpa-onnx's prebuilt aarch64-GPU ORT only goes to v1.18.1 / CUDA 12.

→ Mitigation: build onnxruntime 1.23.1 from source for CUDA 13 + sm_121 (see `build/sherpa.md`).
**Status: the ORT build now SUCCEEDS** — `libonnxruntime_providers_cuda.so` (98 MB) links with
native sm_121 SASS (so it avoids the `cudaErrorUnsupportedPtxVersion` that kills the pip nightly)
and pulls in CUDA-13 cublas/cudart + cuDNN 9. The required CUDA-13 fixes had to be applied to
**both** the nvcc *and* the host `CXX_FLAGS` — the CUDA EP has host `.cc` files that include
CUTLASS / use `longlong4`, and the fixes were initially only on the nvcc flags:
- `-I/usr/local/cuda-13.0/targets/sbsa-linux/include/cccl` (CUDA 13 relocated libcu++;
  CUTLASS's `<cuda/std/utility>` is otherwise not found),
- `-Wno-error=deprecated-declarations` (`longlong4` deprecated in CUDA 13 → `longlong4_16a`),
- disable flash-/memory-efficient-attention.

**Remaining step (not done):** a GPU-enabled sherpa-onnx (C++ + Python wheel) built against this
local ORT — sherpa-onnx's `onnxruntime-linux-aarch64-gpu.cmake` only knows prebuilt ORT ≤1.18.1,
so it needs a local-tarball + hash-bypass hook. This whole chain (ORT → repackage → sherpa-onnx-GPU
wheel) is exactly the friction that makes sherpa-onnx-CUDA impractical on the actual Jetson Nano
gen1 deployment target (sm_53, CUDA 10.2, RAM-constrained) — which is why it's banned there. The
sherpa-CUDA column in `results/BENCHMARK.md` therefore stays empty pending that wheel build.

## 2. GPU memory not reportable
`nvidia-smi --query-gpu=memory.used,memory.total` returns `[N/A]` on the GB10 (Grace-Blackwell
unified memory). GPU-mem columns are therefore `N/A`, not zero.

## 3. RapidSpeech.cpp melo8k vocoder lives only on the fork's gen1 branch
Upstream `RapidAI/RapidSpeech.cpp` main has no Vocos8k decoder, so melo8k produces noise there.
The working decoder is on `vieenrose/RapidSpeech.cpp@jetson-nano-gen1`, which is pinned to an old
ggml tree; building it on modern ggml/CUDA-13 needs the `RS_GGML_HAS_SET_ROWS` compat guards
(see `build/rapidspeech.md`).

## 4. X-ASR shipped fp32; quantized to int8 here
`GilgameshWind/X-ASR-zh-en` chunk-480ms ships fp32 encoder/decoder/joiner. Quantized
encoder+joiner to int8 (`onnxruntime.quantization.quantize_dynamic`, QInt8) to match the int8
deployment; decoder kept fp32 (embedding-based). It is a **zh-en** model — ja/ko inputs are
out-of-domain.

## 5. ggml-Vulkan on the GB10 — RESOLVED
- **ggml-Vulkan (RapidSpeech):** builds & runs on the GB10, but Ubuntu 24.04's `glslc` (shaderc
  2023.8) is too old for modern ggml's shaders (empty `.cpp` → undefined `*_data` symbols at
  link). Fixed by fetching `glslc`/`libshaderc` **2026.2** from a newer Ubuntu pool
  (`apt download` + `dpkg -x`, no sudo) + SPIRV-Headers + libvulkan-dev. Then it works:
  RapidSpeech-Vulkan = CUDA on SenseVoice, ~2.3× behind CUDA on melo8k, worse than CPU on VAD.

GPU memory is `N/A` for Vulkan too (GB10 UMA).
