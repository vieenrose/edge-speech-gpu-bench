"""Shared helpers for edge-speech-gpu-bench."""
import os, re, subprocess, statistics, time, glob, json

ROOT = os.path.dirname(os.path.abspath(__file__))
RS_BIN = os.path.join(ROOT, "third_party/RapidSpeech.cpp/build-cuda")
SV_SHERPA = os.path.join(ROOT, "models/sherpa/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17")
SV_RS = os.path.join(ROOT, "models/rapidspeech/ASR/SenseVoice/sense-voice-small-q5_k.gguf")
SILERO_RS = os.path.join(ROOT, "models/rapidspeech/ASR/silero_vad_v6.gguf")
SILERO_ONNX = os.path.join(ROOT, "models/sherpa/silero_vad.onnx")
MELO_SHERPA = os.path.join(ROOT, "models/sherpa/melo8k")
MELO_RS = os.path.join(ROOT, "models/rapidspeech/openvoice2-melo8k-zh.gguf")
CLIP30 = os.path.join(ROOT, "audio/rtf_clip_30s.wav")
TEN_TESTSET = os.path.join(ROOT, "audio/ten-vad/testset")

# cuDNN libdir for the onnxruntime-gpu CUDA EP (if ever usable).
def cudnn_env():
    env = dict(os.environ)
    libs = glob.glob(os.path.join(ROOT, ".venv/lib/python*/site-packages/nvidia/cudnn/lib"))
    if libs:
        env["LD_LIBRARY_PATH"] = libs[0] + ":" + env.get("LD_LIBRARY_PATH", "")
    return env

def summarize(times):
    """times: list, [0]=cold, rest=warm. Returns (cold, warm_median)."""
    cold = times[0]
    warm = statistics.median(times[1:]) if len(times) > 1 else times[0]
    return cold, warm

def gpu_mem_peak_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1
