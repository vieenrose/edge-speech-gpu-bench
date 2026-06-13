# Building RapidSpeech.cpp with CUDA on GB10 (Blackwell sm_121, CUDA 13)

Source: `vieenrose/RapidSpeech.cpp`, branch **`jetson-nano-gen1`** (it carries the Vocos8k
melo8k decoder + the CUDA im2col fix; upstream `main` does not). Built on **modern ggml**, not
the old CUDA-10.2 tree the branch normally targets.

```bash
git clone https://github.com/vieenrose/RapidSpeech.cpp
cd RapidSpeech.cpp && git checkout jetson-nano-gen1
git submodule update --init --recursive   # or clone ggml-org/ggml master into ggml/

# The gen1 branch ships rs_ggml_compat.h shims for an OLD ggml that lacks
# ggml_set_rows / ggml_interpolate / ggml_softplus / ... On modern ggml those
# exist, so the shims collide. Two fixes:
#   1. define RS_GGML_HAS_SET_ROWS (skips the swiglu_split/set_rows shims)
#   2. wrap the unguarded CosyVoice3 shims (softplus/fill_inplace/scale_bias/
#      pad_ext/interpolate, only used by FunASR-nano, not melo) in the same guard.

cmake -B build-cuda -DRS_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DRS_GGML_HAS_SET_ROWS"
cmake --build build-cuda --target rs-asr-offline rs-tts-offline -j $(nproc)
```

Notes:
- `CMAKE_CUDA_ARCHITECTURES=121` is auto-promoted to `121a` (GB10) by ggml's cmake.
- Binaries link `libggml-cuda.so` → `libcudart.so.13` / `libcublas.so.13`. Set
  `LD_LIBRARY_PATH=build-cuda:build-cuda/ggml/src:build-cuda/ggml/src/ggml-cuda`.
- **SenseVoice ASR** runs on `CUDA0`. **melo8k TTS** is CPU-pinned by default (`prefer_cpu` for
  arch `openvoice2`, since the small Vocos vocoder is faster on CPU); force GPU with
  `RS_FORCE_GPU=1` (the Vocos decoder runs correctly on CUDA, bit-identical to CPU).
- Standalone VAD timing: a ~40-line harness against `librapidspeech-core` using
  `rs_vad_init_from_file` / `rs_vad_push_audio` (see `tools/`).
```bash
# run
./build-cuda/rs-asr-offline -m sense-voice-small-q5_k.gguf -w clip.wav -v silero_vad_v6.gguf --gpu true
RS_FORCE_GPU=1 ./build-cuda/rs-tts-offline -m openvoice2-melo8k-zh.gguf -t "你好" -o out.wav --lang Chinese --gpu true
```
