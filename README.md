# edge-speech-gpu-bench

GPU/CPU benchmark of **edge speech engines** on the NVIDIA **GB10** (DGX Spark — Grace-Blackwell,
aarch64, sm_121, CUDA 13, 128 GB unified memory). Compares, on the same models and audio:

| Engine | Backend | Models benchmarked |
|---|---|---|
| **sherpa-onnx** | ONNX Runtime (CPU; CUDA via from-source ORT) | SenseVoice, X-ASR, melo8k, silero-VAD, TEN-VAD |
| **RapidSpeech.cpp** | ggml (CPU / CUDA / Vulkan) | SenseVoice, melo8k, silero-VAD |

> This is the **GPU half** of an existing CPU benchmark. On CPU, sherpa-onnx wins; RapidSpeech's
> thesis is the ggml CUDA backend — this measures whether GPU flips the verdict. See
> [`results/BENCHMARK.md`](results/BENCHMARK.md) for the full table and verdict.

## Headline results (warm RTF, lower is better)

| Model | sherpa CPU | RapidSpeech CPU | RapidSpeech CUDA | RapidSpeech Vulkan |
|---|---|---|---|---|
| SenseVoice STT | 0.0102 | 0.0739 | **0.0031** | 0.0031 |
| melo8k TTS | 0.0209 | 0.0657 | **0.0107** | 0.0245 |
| silero-VAD | **0.0023** | 0.0055 | 0.0057 | 0.0413 |

- **CPU:** sherpa-onnx wins (6–7× on STT/TTS).
- **GPU:** RapidSpeech's ggml-CUDA flips both STT (0.0031) and TTS (0.0107) ahead of sherpa-CPU —
  but it's **launch-bound** (huge first-graph warmup), so it only pays off in a persistent
  process. See [`results/BENCHMARK.md`](results/BENCHMARK.md).

## Fixes produced along the way
- **RapidSpeech.cpp melo8k**: implemented the missing Vocos8k vocoder + CUDA im2col fix
  (validated bit-exact vs ONNX). The fix already lives on the author's `jetson-nano-gen1` branch.
- **ggml-Vulkan on the GB10**: built RapidSpeech.cpp against modern ggml's Vulkan backend after
  fetching a newer `glslc` (Ubuntu 24.04's is too old for modern ggml shaders).

## Reproduce
```bash
./setup.sh                 # fetch models + build the engines (see build/*.md)
. .venv/bin/activate
python3 tools/bench_full.py       # RTF matrix → results/full_rtf.csv
python3 tools/bench_accuracy.py   # CER / round-trip / VAD agreement
```

## Layout
```
tools/            benchmark + analysis scripts (bench_full, bench_accuracy, compare_melo8k, vocos_ref)
results/          BENCHMARK.md (unified table + verdict) + *.csv
build/            rapidspeech.md, sherpa.md — exact build commands + CUDA-13 fixes
BLOCKERS.md       sherpa-onnx-CUDA build journey, GB10 unified-memory notes, X-ASR int8 notes
```

## Deployment-target deliverable: RapidSpeech.cpp CUDA on Jetson Nano gen1
The real target is the **Jetson Nano gen1** (sm_53, CUDA 10.2), not this GB10 box.
The CUDA backend is now fixed to build *and run* there — the key bug was ggml's
batched cuBLAS matmul requesting tensor-op GEMM, which sm_53 Maxwell can't do (no
tensor cores → `CUBLAS_STATUS_NOT_SUPPORTED`). Fix + full CUDA-10.2 build patch set
shipped to [`vieenrose/RapidSpeech.cpp@jetson-nano-gen1`](https://github.com/vieenrose/RapidSpeech.cpp/tree/jetson-nano-gen1)
(commit `7e802ce`); validated for **correctness** in a JetPack-4.6.1 container
(SenseVoice transcribes correctly, melo8k GPU==CPU corr 0.999999, whole model runs
on `CUDA0`). **Speed pending the real device.** See
[`docs/gen1-cuda-validation.md`](docs/gen1-cuda-validation.md) +
[`docs/jetson-nano-gen1-feasibility.md`](docs/jetson-nano-gen1-feasibility.md).

## Hardware / notes
- GB10 reports `N/A` for `nvidia-smi` GPU memory (unified memory).
- sherpa-onnx CUDA on this box needs a from-source onnxruntime (CUDA-13 / sm_121); see
  `build/sherpa.md` + `BLOCKERS.md`.
- The deployment target is **not** this box — it's Jetson Nano gen1 (sm_53, CUDA 10.2), where
  sherpa-onnx-CUDA is banned for RAM. This is a research comparison host.
