# Blockers / hard parts on the GB10 (Blackwell sm_121, aarch64, CUDA 13.0)

## 1. sherpa-onnx CUDA — no prebuilt path; from-source onnxruntime required
On this box, every prebuilt route to sherpa-onnx-CUDA fails:
- pip `sherpa-onnx` is CPU-only (compiled `SHERPA_ONNX_ENABLE_GPU=OFF`).
- The CUDA-13 aarch64 `onnxruntime-gpu` nightly loads the CUDA EP but aborts at execution with
  `cudaErrorUnsupportedPtxVersion` — built with CUDA 13.1 toolchain, no sm_121 SASS, PTX
  rejected by the 13.0 driver.
- sherpa-onnx's prebuilt aarch64-GPU ORT only goes to v1.18.1 / CUDA 12.

→ Mitigation: build onnxruntime 1.23.1 from source for CUDA 13 + sm_121 (see `build/sherpa.md`).
Required CUDA-13 fix: add `-I/usr/local/cuda-13.0/targets/sbsa-linux/include/cccl` to the CUDA
flags (CUDA 13 relocated libcu++; CUTLASS's `<cuda/std/utility>` is otherwise not found), and
disable flash-/memory-efficient-attention. **Current status of this build + the sherpa-onnx GPU
link against it is recorded in `results/BENCHMARK.md` (sherpa-CUDA column).** This is a heavy,
multi-stage build (ORT → repackage → sherpa-onnx-GPU), exactly the kind of friction that makes
sherpa-onnx-CUDA impractical on the actual Jetson Nano gen1 deployment target (sm_53, CUDA 10.2,
RAM-constrained) — which is why it's banned there.

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

## 5. Vulkan backends — RESOLVED (all three now run on the GB10)
- **ggml-Vulkan (RapidSpeech):** builds & runs on the GB10, but Ubuntu 24.04's `glslc` (shaderc
  2023.8) is too old for modern ggml's shaders (empty `.cpp` → undefined `*_data` symbols at
  link). Fixed by fetching `glslc`/`libshaderc` **2026.2** from a newer Ubuntu pool
  (`apt download` + `dpkg -x`, no sudo) + SPIRV-Headers + libvulkan-dev. Then it works:
  RapidSpeech-Vulkan = CUDA on SenseVoice, ~2.3× behind CUDA on melo8k, worse than CPU on VAD.
- **sensevoice.cpp Vulkan:** FIXED — its GPU selector only matched `..._TYPE_GPU`; ggml reports
  UMA/integrated GPUs as `..._TYPE_IGPU`. Accepting IGPU → Vulkan warm 0.0058 (= CUDA). PR #1.
- **sherpa-ncnn Vulkan:** FIXED — ncnn's FetchContent tree lacked its `nihui/glslang` submodule;
  cloned it in manually + a CLI env toggle for `use_vulkan_compute`. Runs on GB10 but 16× slower
  than CPU (streaming ASR = tiny per-chunk forwards, GPU dispatch-bound).

GPU memory is `N/A` for Vulkan too (GB10 UMA).
