# Building sherpa-onnx with CUDA on GB10 (Blackwell sm_121, CUDA 13) — the hard part

## CPU (trivial)
`pip install sherpa-onnx` (1.13.2). CPU EP only. Used for all CPU columns and X-ASR/TEN-VAD.

## CUDA — no prebuilt path works on this box

1. **pip `sherpa-onnx`** is compiled `SHERPA_ONNX_ENABLE_GPU=OFF`; `provider="cuda"` silently
   falls back to CPU (`Please compile with -DSHERPA_ONNX_ENABLE_GPU=ON`).
2. **Prebuilt `onnxruntime-gpu` (pip):** the CUDA-13 aarch64 nightly
   (`onnxruntime-gpu==1.25.0.dev...` from the aiinfra `ort-cuda-13-nightly` feed) + pip
   `nvidia-cudnn-cu13` **loads** the CUDAExecutionProvider but **aborts at kernel execution**:
   `CUDA error cudaErrorUnsupportedPtxVersion: the provided PTX was compiled with an
   unsupported toolchain`. It was built with the CUDA 13.1 toolchain and ships no GB10 (sm_121)
   SASS; the 13.0 driver can't JIT its PTX.
3. **sherpa-onnx's own `onnxruntime-linux-aarch64-gpu.cmake`** only offers prebuilt ORT up to
   v1.18.1 (CUDA 12) — nothing for CUDA 13 / Blackwell.

## CUDA — from-source onnxruntime (attempted)
Build onnxruntime 1.23.1 from source for CUDA 13.0 + sm_121, then sherpa-onnx GPU against it.

```bash
# cuDNN 9 from pip wheel provides headers+libs:
pip install nvidia-cudnn-cu13
mkdir -p /tmp/cudnn_home && ln -s <wheel>/include /tmp/cudnn_home/include && ln -s <wheel>/lib /tmp/cudnn_home/lib

cd onnxruntime  # v1.23.1
./build.sh --config Release --parallel 20 --build_shared_lib --use_cuda \
  --cuda_home /usr/local/cuda --cudnn_home /tmp/cudnn_home \
  --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=121 onnxruntime_BUILD_UNIT_TESTS=OFF \
    onnxruntime_USE_FLASH_ATTENTION=OFF onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF \
  --skip_tests --allow_running_as_root --compile_no_warning_as_error
```

**CUDA-13 fixes required during the build:**
- CUTLASS `#include <cuda/std/utility>` fails — CUDA 13 moved libcu++ to
  `/usr/local/cuda-13.0/targets/sbsa-linux/include/cccl/`. Add it to the CUDA flags:
  `cmake -DCMAKE_CUDA_FLAGS="-I/usr/local/cuda-13.0/targets/sbsa-linux/include/cccl" .` then resume.
- `nvcc fatal: Unknown option '-Wstrict-aliasing'` appears as a non-fatal CMake probe.
- **The CUDA EP has host `.cc` files** (e.g. `contrib_ops/cuda/bert/multihead_attention.cc`) that
  include CUTLASS and use `longlong4` — so the two fixes above must ALSO be on the **host
  `CXX_FLAGS`**, not just the nvcc CUDA flags. Append to
  `CMakeFiles/onnxruntime_providers_cuda.dir/flags.make` `CXX_FLAGS` (or `CMAKE_CXX_FLAGS`):
  `-Wno-error=deprecated-declarations -I/usr/local/cuda-13.0/targets/sbsa-linux/include/cccl`.
  (`longlong4` is deprecated in CUDA 13 → `longlong4_16a`; the bare trailing `-Werror` on the host
  flags otherwise turns it fatal.) With these, `libonnxruntime_providers_cuda.so` (98 MB, sm_121
  SASS) links cleanly.

**Status:** ORT + CUDA EP build succeeds. The remaining sherpa-onnx-GPU wheel against this local
ORT is not built (see `BLOCKERS.md` §1) — sherpa-onnx's GPU cmake only knows prebuilt ORT ≤1.18.1.

Status of this build + the sherpa-onnx-GPU link against it: see `results/BENCHMARK.md`
(sherpa-CUDA column) and `BLOCKERS.md`.
