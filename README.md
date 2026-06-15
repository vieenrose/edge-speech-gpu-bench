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
>
> **⬇ Update (2026-06-15): now measured on the real Jetson Nano gen1.** The GB10 was always a proxy
> (sm_53 SASS JIT'd onto Blackwell). On real Maxwell the answer is **model-size dependent**: for the
> **compute-heavy SenseVoice STT, ggml-CUDA wins** (RTF 0.104 vs CPU ~0.56, ~5×, *and* fits where
> ORT-CUDA OOMs); for **small TTS (melo8k/matcha), CPU wins** (GPU is launch-bound). ggml-CUDA's *RAM*
> win is real and decisive (it's the only engine that fits the GPU). See
> [Real Jetson Nano gen1 measurements](#-real-jetson-nano-gen1--the-device-measurements-2026-06-15).

## Headline results — **measured on GB10** (warm RTF, lower is better)

> ⚠️ **Every number in this table is from the GB10**, not the Nano. CPU/Vulkan columns are GB10-native
> (sm_121); the CUDA columns are GB10 with **sm_53 SASS PTX-JIT'd** (the "Nano toolchain" container).
> For the **real Jetson Nano gen1 hardware** numbers (where the GPU verdict flips), see
> [Real Jetson Nano gen1 measurements](#-real-jetson-nano-gen1--the-device-measurements-2026-06-15).

| Model (all **GB10**) | sherpa CPU | sherpa CUDA (cuDNN-free)† | RapidSpeech CPU | RapidSpeech CUDA | RapidSpeech Vulkan |
|---|---|---|---|---|---|
| SenseVoice STT | 0.0102 | 0.079 † | 0.0739 | **0.0031** | 0.0031 |
| melo8k TTS | 0.0209 | 0.028 † | 0.0657 | **0.0107** | 0.0245 |
| **Matcha-TTS** ‡‡ | 0.034 | 0.029 ‡‡ | 0.036 | **0.011** | — |
| silero-VAD | **0.0023** | — | 0.0055 | 0.0057 | 0.0413 |

- **CPU:** sherpa-onnx wins (6–7× on STT/TTS).
- **GPU (GB10-native columns):** RapidSpeech's ggml-CUDA flips both STT (0.0031) and TTS (0.0107)
  ahead of sherpa-CPU — but it's **launch-bound** (huge first-graph warmup), so it only pays off in a
  persistent process. See [`results/BENCHMARK.md`](results/BENCHMARK.md).
- **Matcha-TTS is the first model where ggml-CUDA beats sherpa's *cuDNN-free CUDA* head-to-head**:
  on the **exact same phoneme tokens**, warm synth is **ggml-CUDA 26 ms vs sherpa 55 ms (~2.1×)** at
  lower RSS (579 vs 667 MB). The deep 3-step-ODE CFM decoder is where ggml's hand-written graph pulls
  ahead of ORT. **‡‡** Both Matcha figures are measured *in the same Nano-toolchain container* on
  identical tokens (so directly comparable to each other, unlike the † columns); the RTFs are each vs
  that engine's own output length (the two runtimes disagree on synthesized length — see the
  [same-utterance section](#head-to-head-cudnn-free-ort-vs-rapidspeech-both-on-the-nano-toolchain)
  below). The directly-comparable figure is the warm-synth wall-clock.
- **† sherpa CUDA (cuDNN-free)** is the *new* path this project enabled: sherpa-onnx running
  end-to-end on the **cuDNN-free onnxruntime 1.11 CUDA EP** (so the Nano can use the GPU without
  cuDNN's 782 MB). These two figures are measured in the **CUDA-10.2 container with sm_53 PTX-JIT**
  (the Nano toolchain) — a *different context* from the GB10-native columns, so they're relative-only,
  not a like-for-like RTF. Warm (JIT+load excluded); validated end-to-end (melo8k round-trips back to
  the input sentence). Notably, even cuDNN-free GPU here does **not** beat sherpa-CPU on these small
  models — consistent with the open "does GPU even help on the Nano" question. (silero-VAD not run via
  sherpa's cuDNN-free path; its raw-graph cost is in the deliverable #2 section.)

## ⬇ Real Jetson Nano gen1 — the device measurements (2026-06-15)

Everything above is the **GB10 proxy** (sm_53 SASS PTX-JIT'd onto a Blackwell GPU) — the README
flags it throughout as *relative only, real Maxwell TBD* and "**speed pending the real device**".
The device is now in hand. All engines were **x86-cross-compiled** for the Nano (CUDA-10.2 / gcc-8.3
container) and run on **real sm_53 Maxwell** under the live product stack. The headline: **the GPU
verdict flips — on the real Maxwell, no GPU path beats CPU.**

### Measured on real Nano gen1 (single-shot / per-call, the deployment pattern)

> Every RTF and every RSS (MB) in the three tables of this section is measured on **real Jetson Nano
> gen1 hardware** (sm_53 Maxwell, CUDA 10.2, 4 GB) — *not* the GB10.

| Model (RTF + RSS all **real Nano**) | sherpa-onnx **CPU** | sherpa cuDNN-free **CUDA** | RapidSpeech **ggml-CPU** | RapidSpeech **ggml-CUDA** |
|---|---|---|---|---|
| **melo8k** TTS (small) | **0.457** (305 MB) | 0.701 (591 MB) — *slower* | 3.60 *(launch-bound)* | 0.9–1.2, fits @ 455 MB — *slower* |
| **matcha8k** TTS (small) | **0.347** @t4 (210 MB) | 0.502 — *slower* | (harness only) §§ | (harness only) §§ |
| **SenseVoice** STT (heavy) | 0.589 int8 (577 MB) | OOM / empty (int8) † | 0.556 q5 (322 MB) | **0.104 q5 — fits @ 456 MB, ~5× faster** |
| **X-ASR** int8 STT (deployed) | ~0.7–0.85 (modern-ORT static) | **won't load** — `ConvInteger` ‡ | onnx-only | onnx-only |

*(TTS rows are single-shot/per-call — RapidSpeech is **launch-bound** (~1.5 s wall ≈ model-load +
first-graph compile), so per-call RTF is poor even though the GPU runs. The SenseVoice row is **warm
steady-state** (2nd VAD segment of a longer clip, load amortized): RapidSpeech CPU 0.556 vs ggml-CUDA
**0.104** — a clean ~5.3× GPU win, transcript correct on both. §§ matcha-on-RapidSpeech is a
token-injection research harness (`matcha_e2e_test`+`MATCHA_IDS`), not a standard `rs-tts` path, and no
matcha gguf is published — so it isn't benchmarked here.)*

**The model-size split is the real story:** for **small TTS** graphs (melo8k, matcha) the GPU is
launch-bound and **CPU wins**; for the **compute-heavy SenseVoice** STT (50-layer SAN-M attention)
**ggml-CUDA wins decisively** (0.104 vs CPU 0.556/0.589) *and* fits where ORT-CUDA OOMs. So "GPU never
helps on the Nano" is **wrong** — it helps exactly where the model is heavy enough to amortize launch.

### GB10 vs real Nano — what the proxy got right and wrong

| Claim from the GB10 study | On the **real Nano gen1** |
|---|---|
| *"ggml-CUDA flips the verdict — beats sherpa-CPU"* (GB10: SenseVoice CUDA 0.0031 ≪ CPU 0.0102; melo8k CUDA 0.0107 < CPU 0.0209) | **Model-size dependent on real Maxwell.** For the **compute-heavy SenseVoice** it **holds**: ggml-CUDA **0.104** vs CPU 0.556 (RS) / 0.589 (sherpa) — a ~5× GPU win, *and* it fits where ORT-CUDA OOMs. For **small TTS** (melo8k/matcha) it **fails**: the weak Maxwell (128 cores, ~0.5 TFLOP, no tensor cores) is launch-bound on tiny single-utterance graphs, so CPU wins. The GB10's *absolute* RTFs (a Blackwell GPU wearing sm_53 SASS) overstate the real GPU ~50–100×, but the *qualitative* "GPU helps the heavy model" lesson **survives**. |
| *"ggml-CUDA is ~2× lighter than ORT-CUDA"* (GB10 RSS, context-masked) | **TRUE and decisive — now measured.** ggml-CUDA melo8k: **peak RSS 455 MB, min sys-avail 1747 MB → fits** the 4 GB Nano with headroom. sherpa-onnx ORT-CUDA: **~2.8 GB demand → OOM-killed** under the live stack (the ~1 GB CUDA-10.2 context + ~1 GB ORT framework + BFC arena). **ggml-CUDA is the only engine that can use the Maxwell GPU at all on gen1.** |
| *"cuDNN-free EP runs all 5 models"* (validated on GB10 **with cuDNN present**) | **Partly.** On the real Nano (**no cuDNN installed**) the cuDNN-free Conv kernel is correct (`ort_conv_test` PASS, 9.5e-7), but **`CudnnFilterDescriptor`'s ctor still called `cudnnCreateFilterDescriptor`** — every Conv default-constructs one → SIGABRT at session init without cuDNN. GB10 never caught it (its l4t container *had* cuDNN 8.2). **Fixed** (guard under `ORT_CUDA_NO_CUDNN`), pushed to the onnxruntime fork. |

### Two hard limits the device exposed
- **† SenseVoice int8 + ORT-CUDA is a dead end** regardless of RAM: ORT 1.11's CUDA EP has no
  `MatMulInteger`/`DynamicQuantizeLinear`, so the 281 int8 matmuls run on **CPU**, fragmenting the graph
  into hundreds of CPU↔GPU boundary crossings — empty/garbage output on real sm_53, and CPU-bound by
  construction (no speedup possible). The fp32 SenseVoice (937 MB) doesn't fit 4 GB at all.
- **‡ Version scissor.** The Nano's Maxwell caps onnxruntime at **1.11** (last CUDA-10.2 ORT), but the
  deployed **int8 X-ASR** needs `ConvInteger` (ORT ≥ ~1.14). You get the GPU **or** the modern int8
  models, never both. fp32 TTS (melo8k/matcha) only need a mechanical opset-17→16 downgrade and *do*
  run on ORT 1.11 — but, per above, slower than CPU.

### Verdict (real silicon)
**It depends on model size — there is no single winner.**
- **Small TTS (melo8k, matcha8k):** **sherpa-onnx CPU wins.** The Maxwell GPU is *reachable* (ggml-CUDA
  fits and runs correct 8 kHz audio — `Tegra X1 cc 5.3`, `Backend: CUDA0` — where ORT-CUDA OOMs) but
  too weak/launch-bound to beat NEON on the A57s for these tiny graphs.
- **Compute-heavy STT (SenseVoice):** **RapidSpeech ggml-CUDA wins** — RTF **0.104** vs every CPU path
  (0.556 RS, 0.589 sherpa), ~5×, *and* fits (456 MB) where sherpa ORT-CUDA OOMs. The 50-layer SAN-M
  attention is heavy enough to amortize the launch and exploit the GPU's matmul throughput.
- **Deployed pipeline is unaffected either way:** the STT is **X-ASR** (ONNX-only — can't run on ggml,
  and won't load on the GPU-pinned ORT 1.11), and the TTS is small — so the box still runs **CPU**.

The GB10's *RAM* lesson (ggml ≪ ORT; ggml-CUDA is the only engine that fits) holds and is now measured.
Its *speed* lesson (GPU > CPU) **survives for heavy models** but **not for small ones** — the absolute
GB10 RTFs (Blackwell-as-sm_53) overstate the real Maxwell ~50–100×, yet the qualitative split is right.

> **Cross-build recipes** for all of the above (cuDNN-free ORT 1.11 + sherpa fork, and the full kreier
> CUDA-10.2 patch set for RapidSpeech — NEON `_x4` shim, fake `cuda_bf16.h`, `__builtin_assume`/
> `CUDA_R_16BF`/`CUBLAS_COMPUTE_*` back-defines, `stdc++fs`) live in the firmware tree
> (`tools/x86_testrig/nano_xcompile/`). RapidSpeech ggml-CUDA cross-built clean from x86 with gcc-8.3.

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

### Head-to-head: cuDNN-free ORT vs RapidSpeech (both in the Nano-toolchain container) — **measured on GB10**
> ⚠️ **Both tables below are GB10 numbers** (CUDA-10.2 container, sm_53 SASS PTX-JIT'd onto the GB10's
> Blackwell GPU) — *not* real Nano hardware. The "Nano toolchain" = the CUDA-10.2/sm_53 build env, run
> here on the GB10. RSS in MB and times in ms are all GB10. For real Nano numbers see the
> [device section](#-real-jetson-nano-gen1--the-device-measurements-2026-06-15).

Both engines cuDNN-free, in the CUDA-10.2 container with sm_53 dispatch (on GB10). Warm = model-load
**and** one-time PTX-JIT excluded; inputs matched. "sherpa-onnx" = the full pipeline (frontend + ORT GPU
forward); "raw ORT graph" = the bare ONNX forward (my `ort_bench`/`melo_synth`).

| Model (RSS/ms all **GB10**) | RapidSpeech (ggml) | sherpa-onnx end-to-end (cuDNN-free ORT) | raw ORT graph |
|---|---|---|---|
| SenseVoice | **756 MB** / 501 ms | 1227 MB / 443 ms | 1224 MB / 376 ms |
| melo8k (*same* utterance) | **572 MB** / 268 ms | 736 MB / **55 ms** | 721 MB / 54 ms |
| **Matcha-TTS** (zh-tw/en 8k) ‡ | **CPU 87 ms / CUDA 26 ms** (154 / 579 MB) | **667 MB / 55 ms** | — |

**Matcha-TTS — same-utterance head-to-head.** The row above is now *apples-to-apples*: the **exact
same phoneme tokens** are fed to both engines (sentence `這個星期的研究進度。` → sherpa's frontend emits
`[2069, 614, 1886, 1397, 420, 1927, 829, 814, 489, 5]`, dumped via `--debug=1` and injected into the ggml
harness so no text-frontend is needed there). Warm synth = best-of-3 in a persistent context (ggml) /
sherpa's own per-call generation timer (model already loaded):

| Backend (same tokens, all **GB10**) | warm synth | output audio | RTF | peak RSS |
|---|---|---|---|---|
| RapidSpeech ggml **CPU** | 87 ms | 2.40 s | 0.036 | **154 MB** |
| RapidSpeech ggml **CUDA** | **26 ms** | 2.40 s | **0.011** | 579 MB |
| sherpa-onnx **CPU** | 64 ms | 1.88 s | 0.034 | 201 MB |
| sherpa-onnx **cuDNN-free CUDA** | 55 ms | 1.88 s | 0.029 | 667 MB |

For identical token input the **ggml CUDA path (26 ms) is ~2.1× faster than sherpa-onnx's cuDNN-free CUDA
(55 ms)** and ~1.2× lighter (579 vs 667 MB); ggml CPU is the leanest at **154 MB**. On **CPU** the usual
order holds (sherpa 64 ms vs ggml 87 ms — ORT's CPU kernels are faster), so it's CUDA where the ggml port
pulls decisively ahead. Per audio-second the
gap is wider still (ggml CUDA does ~28 % more frames). **One honest caveat:** the two runtimes disagree on
the synthesized *length* for the same tokens — ggml's duration regulator matches host ONNX Runtime (~150
mel frames → 2.40 s), whereas the deployment **ORT 1.11** runtime sherpa links yields ~118 frames → 1.88 s.
This is an ORT-version / duration-regulator divergence (not a vocoder bug — both read `n_fft=512 hop=128`);
each RTF above is therefore computed against that engine's own output, and the directly-comparable number
is the **warm-synth wall-clock for identical input**.

† **Matcha-TTS** ([Luigi/matcha-zh-tw-en-8k](https://huggingface.co/Luigi/matcha-zh-tw-en-8k), a code-mixed
zh-TW/en 8 kHz CFM model): runs end-to-end on sherpa-onnx cuDNN-free CUDA — output matches the
HF reference wav (3.96 s, rms 0.144 vs 0.147). This required a **new cuDNN-free EP adaptation**: Matcha's
flow-decoder UNet uses `InstanceNormalization`, whose CUDA kernel calls cuDNN batchnorm → routed to the
CPU EP under `ORT_CUDA_NO_CUDNN` (pushed to the onnxruntime fork). The vocos vocoder's opset-17
`LayerNormalization` was decomposed to opset 16 (same trick as melo8k).

‡ **Matcha-TTS on RapidSpeech ggml — now COMPLETE.** A full from-scratch port of the CFM model to ggml:
text encoder (RoPE attention) → duration predictor → length regulator → **CFM decoder** (a 1-D UNet —
ResnetBlock1D / BasicTransformerBlock / SnakeBeta / down-up sample — solved with a 3-step Euler ODE) →
Vocos ConvNeXt → iSTFT. **Every stage validated against ONNX** (encoder rel ≤1e-4, length-reg rel 0,
decoder mel **corr 0.999993**, vocos+iSTFT corr 1.0; end-to-end audio corr 0.952). Built with a 12-agent
spec-extraction-and-adversarial-verify workflow then staged ggml validators; see
[the port design doc](https://github.com/vieenrose/RapidSpeech.cpp/blob/jetson-nano-gen1/docs/MATCHA_TTS_GGML_PORT.md)
and [vieenrose/RapidSpeech.cpp@`jetson-nano-gen1`](https://github.com/vieenrose/RapidSpeech.cpp/tree/jetson-nano-gen1)
(`arch/matcha.cpp`).

**Then auto-optimized + made CUDA-capable** (all timings in this paragraph measured on **GB10**). Two passes:
1. **Profiling-driven (CPU):** the iSTFT was 75 % of runtime as a naive O(N²) DFT; a radix-2 FFT made it
   **400× faster (730 → 1.8 ms, GB10)**, **4.06× of the whole pipeline (971 → 239 ms, GB10)**, audio bit-identical.
2. **Backend refactor (CUDA):** `PushText` was CPU-only (it used `ggml_graph_compute_with_ctx`). Rewriting
   it to the backend-agnostic `ggml_backend_sched` path made it **run on CUDA** (opt in `MATCHA_USE_CUDA=1`)
   *and* **2× faster on CPU as a bonus** (dropped the 6 GB per-call context). On the **same-utterance**
   tokens (above), warm synth on the GB10 is **CPU 87 ms** (RTF 0.036, 154 MB) / **CUDA 26 ms** (RTF 0.011,
   579 MB, + ~57 s one-time sm_53→sm_121 JIT cached in `CUDA_CACHE_PATH`). Audio matches the validated path
   (CPU corr 0.99999, CUDA corr 0.996).

   On the gen1 the default stays CPU: ggml gates CUDA-graph replay on `cc ≥ 800` (Ampere) and the Nano is
   sm_53, so a launch-bound CFM graph there can't amortize launches — whether CUDA beats the (now 87 ms)
   CPU path on real Maxwell is an open hardware question; on Ampere+ (Orin) cuda-graphs engage and CUDA
   should win. On the GB10 with identical tokens, **ggml-CUDA (26 ms) already beats sherpa-onnx's
   cuDNN-free CUDA (55 ms) ~2.1×** at lower RSS — see the same-utterance table above.

- **RAM:** RapidSpeech is **~2× lighter** (ggml vs ORT's arena+framework overhead).
- **Warm speed:** on the small VITS/SenseVoice graphs cuDNN-free **ORT/sherpa-onnx is faster** (melo8k ~5×)
  — sherpa-onnx end-to-end ≈ the raw ORT forward + a small CPU frontend, so the pipeline adds little over
  the graph. **Matcha-TTS flips this**: on identical tokens **ggml-CUDA (26 ms) beats sherpa cuDNN-free
  CUDA (55 ms) ~2.1×** — the deep 3-step-ODE CFM decoder is where ggml's hand-written graph pulls ahead.
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
