#!/usr/bin/env bash
# Fetch models + test audio and build the engines for edge-speech-gpu-bench (GB10).
# See build/rapidspeech.md and build/sherpa.md for the exact CUDA-13 build details.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

python3 -m venv .venv && . .venv/bin/activate
pip install -q --upgrade pip
pip install -q huggingface_hub soundfile librosa jiwer matplotlib onnx gguf \
               sherpa-onnx sherpa-ncnn

echo "== models =="
mkdir -p models/sherpa models/rapidspeech
# sherpa SenseVoice + silero VAD + TEN-VAD
( cd models/sherpa
  curl -sL https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2 | tar xj
  curl -sL -o silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  curl -sL -o ten-vad.onnx   https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/ten-vad.onnx )
# melo8k (sherpa ONNX) + RapidSpeech/sensevoice.cpp gguf
hf download Luigi/vits-melo-tts-zh_en-8k --local-dir models/sherpa/melo8k
hf download RapidAI/RapidSpeech ASR/SenseVoice/sense-voice-small-q5_k.gguf ASR/silero_vad_v6.gguf --local-dir models/rapidspeech
hf download Luigi/openvoice2-melo8k-zh-gguf openvoice2-melo8k-zh.gguf --local-dir models/rapidspeech
hf download lovemefan/sense-voice-gguf sense-voice-small-q5_k.gguf --local-dir models/sensevoicecpp
# X-ASR streaming zipformer (quantize encoder+joiner to int8 in bench)
hf download GilgameshWind/X-ASR-zh-en "deployment/models/chunk-480ms-model/*" --local-dir models/xasr
# sherpa-ncnn streaming zipformer
hf download csukuangfj/sherpa-ncnn-streaming-zipformer-bilingual-zh-en-2023-02-13 --local-dir models/ncnn-zipformer

echo "== test audio =="
mkdir -p audio && ( cd audio && git clone --depth 1 --filter=blob:none --sparse https://github.com/TEN-framework/ten-vad.git && cd ten-vad && git sparse-checkout set testset )
python3 - <<'PY'
import soundfile as sf, numpy as np, glob, librosa
ws=sorted(glob.glob("models/sherpa/sherpa-onnx-sense-voice-*/test_wavs/*.wav")); sr=16000; clip=[]
while sum(len(c) for c in clip)/sr < 30:
    for w in ws:
        a,s=sf.read(w); a=a.mean(1) if a.ndim>1 else a
        if s!=sr: a=librosa.resample(a.astype('float32'),orig_sr=s,target_sr=sr)
        clip.append(a.astype('float32'))
        if sum(len(c) for c in clip)/sr>=30: break
sf.write("audio/rtf_clip_30s.wav", np.concatenate(clip), sr)
PY

echo "== build engines: see build/rapidspeech.md and build/sherpa.md =="
echo "setup complete. activate: . .venv/bin/activate"
