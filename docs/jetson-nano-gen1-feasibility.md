# Running sherpa-onnx & RapidSpeech.cpp on the Jetson Nano gen1 — feasibility

Side investigation: which of {sherpa-onnx, RapidSpeech.cpp} × {CPU, CUDA, Vulkan} are actually
runnable on the **deployment target** — Jetson Nano gen1, JetPack 4.6.1 (L4T 32.7.1), Ubuntu
18.04, **CUDA 10.2**, Maxwell GPU **sm_53**, **4 GB shared LPDDR4**, gcc 7/8.

This is the opposite corner from the GB10 research box (CUDA 13, sm_121). The constraints that
dominate here are **nvcc 10.2's C++14 wall**, the **4 GB shared RAM**, and **Ubuntu 18.04's
ancient toolchain**.

## The matrix

| Engine \ Backend | CPU | CUDA (sm_53 / CUDA 10.2) | Vulkan (Maxwell) |
|---|---|---|---|
| **sherpa-onnx** (ONNX Runtime) | ✅ ships / works | ⚠️ builds, but **RAM-impractical** → effectively banned | ❌ **impossible** — ORT has no Vulkan EP |
| **RapidSpeech.cpp** (ggml) | ✅ works (deployment path) | ✅ **the intended GPU path** — lean, sm_53/CUDA-10.2 | 🟡 possible but pointless (CUDA is the native, better path) |

## Per-cell detail

### sherpa-onnx — CPU ✅
The deployment baseline. Prebuilt `sherpa-onnx` aarch64 wheels exist; or build from source. Note
JetPack 4.6 ships **Python 3.6**, so pick a matching wheel. ONNX Runtime's CPU EP is NEON-tuned
and fast for these small speech models (this is why sherpa wins CPU in the GB10 benchmark too).

### sherpa-onnx — CUDA ⚠️ (buildable, but the RAM killer)
- ONNX Runtime *can* be built with the CUDA EP for CUDA 10.2 / JetPack 4.6. sherpa-onnx even
  hard-codes the recipe: `-DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.11.0` downloads a
  prebuilt `onnxruntime-linux-aarch64-gpu-1.11.0` (CUDA 10.2) from `csukuangfj/onnxruntime-libs`.
  Build from source works too (`--use_cuda --cuda_home /usr/local/cuda --cudnn_home
  /usr/lib/aarch64-linux-gnu`), though it's a multi-hour on-device build.
- **Why it's banned in practice:** the ONNX Runtime CUDA EP + **cuDNN 8.2** pull in hundreds of
  MB–GB of device/host memory just to initialise, on top of the model — on **4 GB shared** RAM,
  with the rest of the live-call pipeline (ASR `-t 2`, LiveKit/Asterisk, the agent) already
  spoken for, it OOMs. ORT-CUDA is built for discrete-GPU memory budgets, not a 4 GB Tegra. This
  is the documented reason sherpa-onnx-CUDA is off the table on gen1.

### sherpa-onnx — Vulkan ❌ (does not exist)
**ONNX Runtime has no Vulkan execution provider** — only standing feature requests since 2021
(microsoft/onnxruntime #7433, #10603, #21917). The only Vulkan-ish path is the experimental
**WebGPU EP**, which is not available/relevant for a CUDA-10.2-era Jetson. So this cell is
architecturally impossible, independent of the hardware.

### RapidSpeech.cpp — CPU ✅
ggml's NEON CPU backend builds cleanly on Ubuntu 18.04. This is what the `jetson-nano-gen1`
branch's `build_jetson_nano_gen1.sh` targets, and what's device-accepted today.

### RapidSpeech.cpp — CUDA ✅ (the whole thesis)
This is the **point** of RapidSpeech on gen1: offload the conv-heavy 85 % (flow + Vocos decoder)
to the otherwise-idle Maxwell GPU **without** ORT/TensorRT/cuDNN memory overhead.
- **Toolchain:** ggml-CUDA does compile under **nvcc 10.2 + gcc 8.5 + sm_53**, but *only with an
  old ggml tree* — modern ggml requires C++17/newer CUDA, which nvcc 10.2 rejects (the C++14
  wall). The gen1 branch pins the **sensevoice-era ggml** for exactly this reason; community
  proof exists (kreier/llama.cpp-jetson compiles ggml-CUDA on the Nano with gcc 8.5 / nvcc 10.2).
- **Required CUDA-10.2 patches** (well-known): define the missing cuBLAS enums for
  `CUDA_VERSION < 11000` (`CUBLAS_COMPUTE_16F/32F`, `CUBLAS_TF32_TENSOR_OP_MATH`), and on gcc-8
  aarch64 drop the `__builtin_assume` in `fattn-common.cuh`. Build arch `sm_53`.
- **RAM:** ggml-CUDA is lean — ~45 MB f16 weights + tens of MB activations on the UMA, which is
  the entire reason it fits where ORT-CUDA can't.
- **Note on this repo:** the GB10 build is the *mirror* problem — there I had to take the gen1
  branch to **modern** ggml + CUDA 13 / sm_121, bridging via the `rs_ggml_compat.h` shims
  (`RS_GGML_HAS_SET_ROWS`). gen1 = old-ggml/CUDA-10.2; GB10 = modern-ggml/CUDA-13; same source,
  opposite ggml vintages.

### RapidSpeech.cpp — Vulkan 🟡 (possible, but the wrong tool here)
- Maxwell **does** expose Vulkan (pre-installed with JetPack; Vulkan 1.1/1.2 compute), and ggml
  has a Vulkan backend, so in principle RapidSpeech could run Vulkan on gen1.
- **But the practicalities fight you:** ggml's Vulkan backend lives in **modern** ggml, whereas
  the CUDA path needs **old** ggml — so a Vulkan build would be a *separate, Vulkan-only* build
  (no nvcc, so the C++14 wall doesn't apply; gcc 7+ gives C++17). It needs a **recent glslc**
  (Ubuntu 18.04's is far too old — same wall I hit on the GB10, where I back-ported glslc 2026.2)
  and Maxwell lacks the modern Vulkan features (coopmat, etc.) ggml prefers — it would fall back
  to the slow generic paths. There are also documented **Jetson-Nano Vulkan driver crashes**
  (Tencent/ncnn #978: `destroy_gpu_instance` segfaults on the Nano).
- **And the payoff is ~zero:** Vulkan's reason to exist is *cross-vendor* portability. On an
  NVIDIA Jetson, CUDA is the first-class, better-optimised, lower-overhead path. The GB10
  benchmark already shows Vulkan only *ties* CUDA on big batched work and *loses* on small ones —
  on the much weaker Maxwell with old drivers, Vulkan would be strictly worse than the CUDA path.

## Bottom line

On the Jetson Nano gen1 there are really only **three viable cells**, and only **one** sensible
GPU story:

1. **sherpa-onnx CPU** — fast, simple, the safe baseline.
2. **RapidSpeech.cpp CPU** — works; slower than sherpa on CPU.
3. **RapidSpeech.cpp CUDA** — *the* way to use the Maxwell GPU within the 4 GB budget; ORT-CUDA
   can't fit and ORT has no Vulkan EP, so ggml-CUDA is the only practical GPU accelerator on this
   device. This is precisely the niche RapidSpeech.cpp was built for.

Vulkan is a dead end on gen1 (impossible for sherpa, pointless for RapidSpeech). sherpa-onnx-CUDA
is a RAM dead end.

## Deep-dive: RapidSpeech.cpp CUDA per model on gen1

**The framing that changes everything.** The Maxwell GPU is a **single, scarce** resource (128
CUDA cores, ~0.5 TFLOP fp32), and the 4 A57 cores are *already spoken for during a live call*
(`X-ASR -t 2` + LiveKit/Asterisk/agent ≈ 1 core). So the GPU-offload value is **freeing CPU cores
under a real-time load**, *not* lowering RTF in isolation. A model is worth offloading only if it
(a) is heavy enough that the weak Maxwell still beats the A57 *and* (b) would otherwise hog CPU
cores you need for the concurrent call. That makes the per-model verdicts very lopsided.

### melo8k TTS (openvoice2-melo8k-zh) — ✅ the one true offload target
- **Why:** measured device profile (`jetson-tts/ggml_offload/PLAN.md`): **flow ≈ 70–75 %**,
  enc+SDP ≈ 15 %, **Vocos8k dec ≈ 12.6 %**. The GPU-portable 85 % (flow + dec) is **conv-only** —
  WaveNet dilated `conv1d` stacks + ConvNeXt/iSTFT — exactly what `im2col`+GEMM accelerates on
  Maxwell (~3–6× over the A57). enc + duration/SDP (attention + splines) stays on CPU.
- **Numbers:** CPU-only melo8k on the Nano is **0.79 @ t=1 / 0.34 @ t=4**. The split
  (`text → ORT-CPU enc/SDP → ggml-CUDA flow+dec`) predicts **RTF ≈ 0.25–0.35 on ONE core**, with
  **3 cores freed** for the call — that's the whole point.
- **Memory:** ~**45 MB f16** weights + tens of MB activations on the UMA (vs the 127 MB fp32
  ONNX). Trivially fits 4 GB. This is the lean budget ORT-CUDA can't match.
- **Implementation status (from this repo):** the Vocos8k decoder is implemented in
  `openvoice2.cpp` and the conv path is `conv1d`/`conv_1d_dw` → `im2col`. ⚠️ **one cross-version
  gotcha:** the GB10 `im2col` fix I made forces the conv *data* to **F32** because *modern*
  ggml-cuda's `mmvf`/`im2col` kernels assert F32 — but **old (sensevoice-era) ggml-cuda on sm_53
  wants the data as F16** (the original code cast to F16 for exactly that). On gen1 the *original*
  cast is correct, not my GB10 fix. The Vocos/flow port must be validated against the old ggml's
  im2col on-device (parity vectors already exist in `jetson-tts/ggml_offload/`).

### silero-VAD — ❌ keep on CPU (do **not** offload)
- It's a ~1–2 M-param model invoked **per 512-sample frame, continuously**, for the whole call.
  On the GB10 (a vastly stronger GPU) it was already **GPU-neutral** (CUDA 0.0057 ≈ CPU 0.0055);
  on Maxwell, with much higher per-launch latency, GPU would be **slower than CPU** — same shape
  as the GB10 Vulkan result (0.041 ≫ 0.0055).
- It's also *cheap* on one A57 (well under real-time), and offloading it would **contend with the
  TTS** for the single Maxwell. Verdict: VAD stays CPU; the GPU is reserved for TTS flow+dec.

### SenseVoice-small ASR — ⚠️ would benefit, but doesn't fit the gen1 picture
- It's the opposite of VAD: **50 SAN-M encoder layers, 209 MB q5_k** — compute-heavy, the kind of
  model that won **10× on GB10 CUDA**. In isolation the Maxwell would help its big matmuls.
- **But three things kill it on gen1:** (1) **RAM** — 209 MB weights + encoder activations on a
  4 GB shared budget that already holds the TTS model, OS, and the call stack; (2) **GPU
  contention** — it's offline/whole-utterance and would fight the real-time TTS for the one
  Maxwell; (3) **the deployment uses streaming `X-ASR` (zipformer), not SenseVoice**, for ASR. So
  SenseVoice-CUDA on gen1 is *technically* favourable but practically out — if SenseVoice were
  used at all, it'd run CPU to leave the GPU for TTS.
- (Its SAN-M encoder is also **attention-heavy**, and Maxwell ggml-cuda has no flash-attention and
  weak fp16 — so the realized speedup would be well below the GB10's 10×.)

### X-ASR streaming zipformer — CPU (GPU-hostile by construction)
- Not a RapidSpeech model, but for completeness: streaming ASR decodes in **tiny per-chunk
  forwards** — the single most GPU-hostile pattern in the whole benchmark (GB10 showed streaming
  loses badly to CPU). On Maxwell it's strictly CPU.

### Maxwell ggml-CUDA gotchas that bite per-model
- **fp16:** sm_53 has only partial-rate fp16; q5_k **MMQ** kernels dequant-then-GEMM. Keep weights
  **f16** (PLAN does) and expect the matmul advantage to be modest vs the GB10.
- **`im2col` F16/F32:** the central cross-version trap above — *opposite* requirement on old
  (sm_53/CUDA-10.2) vs modern (sm_121/CUDA-13) ggml-cuda. Any conv-heavy model (melo flow/dec)
  must be re-validated on the old kernel.
- **No CUDA graphs / coopmat:** old ggml + Maxwell means per-op launch overhead isn't amortised —
  which is *why* only the genuinely conv-heavy, sustained TTS workload is worth offloading, and
  the small/streaming ones (VAD, zipformer) are not.

## Modern ggml on gen1 — what actually compiles (the C++17 wall, precisely)

The single fact that shapes everything: **modern ggml requires C++17**. Verified in the tree this
repo uses (`ggml@57ea0bc`):
- `CMakeLists.txt`: `set(CMAKE_CXX_STANDARD 17)` (REQUIRED), and
  `target_compile_features(<target> PRIVATE c_std_11 cxx_std_17)` **with the literal comment
  `# don't bump`** — it pins *at least* C++17 on **every** target, CPU and CUDA alike.
- `ggml-cuda/CMakeLists.txt` gates features at CUDA **11.6 / 11.8 / 12.8 / 12.9**; ~30 `.cu/.cuh`
  files use `if constexpr`, `__nv_bfloat16`, `cp.async`, etc. — all post-CUDA-10 / C++17.

Cross that against the gen1 toolchain (nvcc **10.2**, gcc 7/8, L4T 32.x):

| Modern-ggml backend | gen1? | The deciding fact |
|---|---|---|
| **CPU** | ✅ buildable | Needs a **C++17 host compiler only**. Ubuntu 18.04's stock gcc 7.5 is borderline; install **gcc-8/9** (Ubuntu toolchain PPA — nvcc isn't involved for the CPU backend). NEON kernels run on the A57; i8mm/SVE paths are arch-guarded off. |
| **CUDA** | ⚠️ **possible with a vintage-dependent patch set** (not impossible — corrected below) | nvcc 10.2 caps device code at C++14, but ggml's bf16/tensor-core/`cp.async` C++17 code is mostly behind `#if __CUDA_ARCH__>=N` guards that the preprocessor strips for sm_53. Force `-DCMAKE_CUDA_STANDARD=14`, build `sm_50/61`, and patch the *un-guarded* C++17 that remains. Proven on a real recent ggml — see **"Patch modern ggml for CUDA 10.2"** below. The Nano still can't move past CUDA 10.2 (L4T 32.x ceiling), and Maxwell can't use tensor cores/bf16/dp4a, so the realized kernels are the basic ones (~20 % over CPU in practice). |
| **Vulkan** | 🟡 the *only* modern-ggml GPU route, and it's fragile | The Vulkan backend **never invokes nvcc** — GPU code is GLSL→SPIR-V via **glslc**, compiled by the host C++17 compiler. So with gcc-8/9 + a **back-ported recent glslc** (18.04's is far too old — the exact wall I hit on the GB10 and solved with `apt download` glslc 2026.2) it can build. But Maxwell's Vulkan lacks coopmat/modern features (ggml falls back to slow generic shaders), and the Jetson-Nano Vulkan driver is flaky (ncnn #978). Buildable ≠ worthwhile. |

**Why the gen1 branch is built the way it is.** RapidSpeech.cpp's *application* code is made
ggml-version-agnostic by `rs_ggml_compat.h` (it shims the ops old ggml lacks — `ggml_set_rows`,
`ggml_interpolate`, …; on the GB10 I went the *other* way and defined `RS_GGML_HAS_SET_ROWS` to
skip those shims on modern ggml). So the **only** thing that differs per platform is the **ggml
submodule pin**:
- **gen1 (CUDA):** pin the **old, sensevoice-era ggml** — it compiles as-is on nvcc 10.2 / C++14 /
  sm_53 with *zero* CUDA patches. (A *modern* ggml CUDA build is also reachable via the kreier
  C++14-force patch set — see below — but it buys nothing on Maxwell and grows a maintenance tax,
  so old ggml is the rational default, not the only option.)
- **GB10 (CUDA):** pin **modern ggml** + CUDA 13 / sm_121 (what this repo's benchmark used).
- **Either box (Vulkan):** modern ggml, no nvcc — but only practical on the GB10.

**Net:** for the gen1 **CUDA** GPU, **old ggml is the rational pin** (zero patches, exactly the
Maxwell-usable kernels). Modern-ggml-CUDA is *possible* there too (force C++14 + the kreier patch
set), just not worth its upkeep; modern-ggml-Vulkan is possible but fragile and pointless on
Maxwell; and the only *immovable* limits are the L4T-32 CUDA-10.2 ceiling and Maxwell's missing
tensor-core/bf16/dp4a hardware.

## "Patch modern ggml to build on CUDA 10.2?" — yes, it's been done (corrected)

**Correction to an earlier draft of this doc.** I first called this "impossible / a 37 k-line
rewrite." That's wrong, and [kreier's llama.cpp-jetson guide](https://kreier.github.io/llama.cpp-jetson/)
proves it: a **recent** llama.cpp (**b5050 / commit `23106f9`, April 2025** — modern ggml, *with*
bf16) builds with the CUDA backend on the 2019 Jetson Nano under **nvcc 10.2 + gcc 8.5**.

**Why it works despite the C++14 cap:** ggml's heavy C++17 / bf16 / tensor-core / `cp.async` code
is overwhelmingly behind `#if __CUDA_ARCH__ >= N` / `*_AVAILABLE` guards. Targeting **sm_50/61**,
the preprocessor strips those blocks *before* nvcc parses them — so **189 of the ~231 `if
constexpr` are never seen**, and only the bits compiled for Maxwell need touching.

**kreier's recipe (b5050, the working reference):**
- `nvcc 10.2`, **`gcc/g++ 8.5`** (nvcc 10.2 refuses gcc > 8), cmake ≥ 3.14.
- `set(CMAKE_CUDA_ARCHITECTURES 50 61)` in `CMakeLists.txt`.
- `-DGGML_CUDA=ON -DGGML_NATIVE=off -DGGML_CPU_ARM_ARCH=armv8-a`
  **`-DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_STANDARD_REQUIRED=true`** (force C++14 over ggml's 17).
- **~4 source edits:** remove a stray `constexpr` in `ggml-cuda/common.cuh`; comment the
  `__builtin_assume(...)` in the 3 fattn files (gcc-8 aarch64 ICE); `target_link_libraries(ggml
  PRIVATE stdc++fs)` for `std::filesystem`; **stub bf16** — either `#define`/typedef
  `nv_bfloat16 → half` in a fake `cuda_bf16.h`, or replace the ~8 `nv_bfloat16` uses with `half`
  across 3 files.
- Plus the classic `CUBLAS_COMPUTE_*` back-defines for `CUDA_VERSION < 11000`.
- Result: **~85 min build**, runs, **~20 % faster than CPU** (5–6 tok/s, 1B LLM, full offload).

**The real cost is "patch surface scales with ggml vintage", not 37 k lines.** kreier's
April-2025 tree needed ~4–6 edits. The **June-2026** tree this repo uses (`57ea0bc`) already has
**~42 un-guarded `if constexpr`** in the Maxwell-active kernels (`mmvf/mmvq/convert/im2col`) — each
must be rewritten to a plain `if` / template specialization for C++14. So expect *dozens* of edits
on a current ggml, and to **re-apply them on every `sync : llama.cpp`**. It's a maintained patch
set, not a one-off.

**And the payoff stays modest:** Maxwell can't use tensor cores, bf16, or `dp4a` regardless, so
after patching you run the *basic* `im2col`+GEMM / fp16 / dequant kernels — the same ones an old
ggml already gives you. kreier's measured win is ~20 % over CPU for LLM token-gen; for
RapidSpeech's conv-heavy TTS the `im2col` path could do better (PLAN.md's 3–6×), but that gain
comes from the *basic conv kernels*, not from anything modern-ggml-specific.

**So the decision for RapidSpeech on gen1 is a genuine trade, not impossible-vs-possible:**
- **Old ggml (what the gen1 branch pins):** *zero* CUDA patches, compiles as-is, exposes exactly
  the Maxwell-usable kernels. Lowest maintenance. Recommended.
- **Recent ggml + kreier patch set:** reachable, gives you modern ggml's broader CPU/op coverage,
  but you carry a vintage-growing patch set (force C++14, sm_50/61, bf16 stub, the un-guarded
  `if constexpr` rewrites) re-applied per bump — for the *same* basic Maxwell CUDA kernels.

Either way, the heavy custom ops the TTS offload needs (zero-insert upsample, overlap-add iSTFT,
per `jetson-tts/ggml_offload/PLAN.md`) sit on top, and `rs_ggml_compat.h` keeps the app code
ggml-version-agnostic. Net: **modern-ggml-CUDA on the Nano is possible (kreier-style), just not
worth its maintenance tax — old ggml remains the rational pin, but the door is open, not walled.**

### gen1 strategy in one line
Run **everything on CPU except melo8k TTS's flow+dec**, which goes to the Maxwell via ggml-CUDA to
free the 3 CPU cores the live call needs. VAD, the streaming ASR, and (if present) SenseVoice all
stay on CPU — the Maxwell is too weak and too small-RAM to time-share, and only the TTS has both
the conv-heavy compute *and* the core-pressure that justify the offload.

### Sources
- ORT GPU build on Jetson / RAM limits: NVIDIA Developer Forums (onnxruntime-gpu build/install
  threads), microsoft/onnxruntime Discussion #11226.
- ORT has no Vulkan EP: microsoft/onnxruntime issues #7433, #10603, #21917, #22973.
- ggml-CUDA on Jetson Nano (gcc 8.5 / nvcc 10.2 / sm_53 + patches): github.com/kreier/llama.cpp-jetson,
  ggml-org/llama.cpp issue #4099.
- Vulkan on Jetson Nano flakiness: Tencent/ncnn issue #978; ggml Vulkan backend: ggml-org/llama.cpp.
