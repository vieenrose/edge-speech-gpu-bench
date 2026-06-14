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

| Model | sherpa CPU | sherpa CUDA (cuDNN-free)† | RapidSpeech CPU | RapidSpeech CUDA | RapidSpeech Vulkan |
|---|---|---|---|---|---|
| SenseVoice STT | 0.0102 | 0.079 † | 0.0739 | **0.0031** | 0.0031 |
| melo8k TTS | 0.0209 | 0.028 † | 0.0657 | **0.0107** | 0.0245 |
| silero-VAD | **0.0023** | — | 0.0055 | 0.0057 | 0.0413 |

- **CPU:** sherpa-onnx wins (6–7× on STT/TTS).
- **GPU (GB10-native columns):** RapidSpeech's ggml-CUDA flips both STT (0.0031) and TTS (0.0107)
  ahead of sherpa-CPU — but it's **launch-bound** (huge first-graph warmup), so it only pays off in a
  persistent process. See [`results/BENCHMARK.md`](results/BENCHMARK.md).
- **† sherpa CUDA (cuDNN-free)** is the *new* path this project enabled: sherpa-onnx running
  end-to-end on the **cuDNN-free onnxruntime 1.11 CUDA EP** (so the Nano can use the GPU without
  cuDNN's 782 MB). These two figures are measured in the **CUDA-10.2 container with sm_53 PTX-JIT**
  (the Nano toolchain) — a *different context* from the GB10-native columns, so they're relative-only,
  not a like-for-like RTF. Warm (JIT+load excluded); validated end-to-end (melo8k round-trips back to
  the input sentence). Notably, even cuDNN-free GPU here does **not** beat sherpa-CPU on these small
  models — consistent with the open "does GPU even help on the Nano" question. (silero-VAD not run via
  sherpa's cuDNN-free path; its raw-graph cost is in the deliverable #2 section.)

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
TensorRT 8.2.1 path analyzed too — see [`docs/tensorrt-jetson-nano-gen1.md`](docs/tensorrt-jetson-nano-gen1.md) (it doesn't escape cuDNN's 782 MB by default).

## Deployment-target deliverable #2: cuDNN-free onnxruntime CUDA (sherpa-onnx's engine)
sherpa-onnx runs on **onnxruntime**, whose stock CUDA EP pulls in **cuDNN (~782 MB resident** on the
Nano's cuDNN 8) — prohibitive on a 4 GB device. So the second deliverable is a **cuDNN-free,
cuBLAS-only CUDA Execution Provider** for onnxruntime, built with one switch
`-Donnxruntime_CUDA_NO_CUDNN=ON`: Conv/ConvTranspose → im2col/col2im + cuBLAS (incl. asymmetric/TF-SAME
pad), Pooling/Reduce/RNN → CPU EP, and `cudnnCreate` skipped so cuDNN is never loaded. All **5 models
run cuDNN-free** on the Nano toolchain (ORT 1.11.0 / CUDA 10.2 / sm_53): SenseVoice, silero-VAD,
melo8k (opset-16), TEN-VAD, X-ASR. Shipped to **[vieenrose/onnxruntime](https://github.com/vieenrose/onnxruntime)**:
- [`cudnn-free-cuda-jetson-nano-gen1`](https://github.com/vieenrose/onnxruntime/tree/cudnn-free-cuda-jetson-nano-gen1) — ORT 1.11.0, the **deployable Nano build** (CUDA 10.2; last ORT to support it).
- [`cudnn-free-cuda-ep`](https://github.com/vieenrose/onnxruntime/tree/cudnn-free-cuda-ep) — ORT 1.23.1, modern-CUDA **reference** (does *not* run on the Nano — needs CUDA 12/13).

And **sherpa-onnx itself** now builds + runs end-to-end on that cuDNN-free EP (one source tweak for the
ORT-1.11 API + a build recipe): **[vieenrose/sherpa-onnx@`cudnn-free-ort-1.11-jetson-nano`](https://github.com/vieenrose/sherpa-onnx/tree/cudnn-free-ort-1.11-jetson-nano)**.
Run with `--provider=cuda:cfg.txt` where `cfg.txt` has `GraphOptimizationLevel=1` (disables the
cuDNN-only FusedConv).

### Head-to-head: cuDNN-free ORT vs RapidSpeech (both on the Nano toolchain)
Both engines cuDNN-free, in the CUDA-10.2 container with sm_53 dispatch. Warm = model-load **and**
one-time PTX-JIT excluded; inputs matched. "sherpa-onnx" = the full pipeline (frontend + ORT GPU
forward); "raw ORT graph" = the bare ONNX forward (my `ort_bench`/`melo_synth`).

| Model | RapidSpeech (ggml) | sherpa-onnx end-to-end (cuDNN-free ORT) | raw ORT graph |
|---|---|---|---|
| SenseVoice | **756 MB** / 501 ms | 1227 MB / 443 ms | 1224 MB / 376 ms |
| melo8k (*same* utterance) | **572 MB** / 268 ms | 736 MB / **55 ms** | 721 MB / 54 ms |
| **Matcha-TTS** (zh-tw/en 8k) | port in progress † | **681 MB / 139 ms** (3.96 s audio, RTF ~0.035) | — |

† **Matcha-TTS** ([Luigi/matcha-zh-tw-en-8k](https://huggingface.co/Luigi/matcha-zh-tw-en-8k), a code-mixed
zh-TW/en 8 kHz CFM model): runs end-to-end on sherpa-onnx cuDNN-free CUDA **today** — output matches the
HF reference wav (3.96 s, rms 0.144 vs 0.147). This required a **new cuDNN-free EP adaptation**: Matcha's
flow-decoder UNet uses `InstanceNormalization`, whose CUDA kernel calls cuDNN batchnorm → routed to the
CPU EP under `ORT_CUDA_NO_CUDNN` (pushed to the onnxruntime fork). The vocos vocoder's opset-17
`LayerNormalization` was decomposed to opset 16 (same trick as melo8k). The **RapidSpeech/ggml** port is
underway — the onnx→gguf weight converter is done + validated (385 tensors round-trip), the full CFM
arch (text encoder + flow-matching ODE loop + iSTFT vocos head) is in progress; see
[the port design doc](https://github.com/vieenrose/RapidSpeech.cpp/blob/jetson-nano-gen1/docs/MATCHA_TTS_GGML_PORT.md).

- **RAM:** RapidSpeech is **~2× lighter** (ggml vs ORT's arena+framework overhead).
- **Warm speed:** cuDNN-free **ORT/sherpa-onnx is faster on both** (melo8k ~5×) — the reverse of the RAM
  picture. sherpa-onnx end-to-end ≈ the raw ORT forward + a small CPU frontend, confirming the pipeline
  adds little over the graph.
- **Correctness:** sherpa-onnx transcribes zh.wav correctly and its melo8k TTS **round-trips back to the
  input sentence** ("人工智能正在改变世界"); the raw melo8k run is also verified bit-fair (identical
  17664-sample output, cuDNN-free **GPU == CPU corr 1.000000**).
- **Caveat:** these are GB10 numbers via sm_53 PTX-JIT'd-then-cached kernels — *relative* only; real
  Nano Maxwell silicon is still TBD.

Full detail + per-op cuDNN-free table: [`docs/cudnn-free-ort-vs-rapidspeech-nano.md`](docs/cudnn-free-ort-vs-rapidspeech-nano.md),
[`docs/cudnn-free-ort-conv.md`](docs/cudnn-free-ort-conv.md).

## Hardware / notes
- GB10 reports `N/A` for `nvidia-smi` GPU memory (unified memory).
- sherpa-onnx CUDA on this box needs a from-source onnxruntime (CUDA-13 / sm_121); see
  `build/sherpa.md` + `BLOCKERS.md`.
- The deployment target is **not** this box — it's Jetson Nano gen1 (sm_53, CUDA 10.2), where
  sherpa-onnx-CUDA is banned for RAM. This is a research comparison host.

## Forks & models used in this benchmark
**My forks (the changes this benchmark produced):**
- **[vieenrose/onnxruntime](https://github.com/vieenrose/onnxruntime)** — cuDNN-free CUDA Execution
  Provider. Branches: [`cudnn-free-cuda-jetson-nano-gen1`](https://github.com/vieenrose/onnxruntime/tree/cudnn-free-cuda-jetson-nano-gen1)
  (ORT 1.11, the Nano build) and [`cudnn-free-cuda-ep`](https://github.com/vieenrose/onnxruntime/tree/cudnn-free-cuda-ep) (ORT 1.23 reference).
- **[vieenrose/RapidSpeech.cpp](https://github.com/vieenrose/RapidSpeech.cpp)** — Jetson Nano gen1
  CUDA-10.2 / sm_53 build + melo8k Vocos8k vocoder, on the [`jetson-nano-gen1`](https://github.com/vieenrose/RapidSpeech.cpp/tree/jetson-nano-gen1) branch.
- **[vieenrose/sherpa-onnx](https://github.com/vieenrose/sherpa-onnx)** — builds + runs end-to-end on
  the cuDNN-free ORT 1.11 (ORT-1.11 API fix + build recipe), on the [`cudnn-free-ort-1.11-jetson-nano`](https://github.com/vieenrose/sherpa-onnx/tree/cudnn-free-ort-1.11-jetson-nano) branch.

**Models** (fetched by `setup.sh`):

| Model | sherpa-onnx / cuDNN-free ORT (ONNX) | RapidSpeech (gguf) |
|---|---|---|
| SenseVoice | [k2-fsa sense-voice-…-int8-2024-07-17](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2) | [RapidAI/RapidSpeech](https://huggingface.co/RapidAI/RapidSpeech) · `sense-voice-small-q5_k.gguf` |
| melo8k TTS | [Luigi/vits-melo-tts-zh_en-8k](https://huggingface.co/Luigi/vits-melo-tts-zh_en-8k) · `model.opset16.onnx` (the ORT-1.11 / opset-16 build produced here) | [Luigi/openvoice2-melo8k-zh-gguf](https://huggingface.co/Luigi/openvoice2-melo8k-zh-gguf) |
| silero-VAD | [k2-fsa silero_vad.onnx](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx) | [RapidAI/RapidSpeech](https://huggingface.co/RapidAI/RapidSpeech) · `silero_vad_v6.gguf` |
| TEN-VAD | [k2-fsa ten-vad.onnx](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx) | — (onnx-only) |
| X-ASR (480 ms) | [GilgameshWind/X-ASR-zh-en](https://huggingface.co/GilgameshWind/X-ASR-zh-en) | — (onnx-only) |
