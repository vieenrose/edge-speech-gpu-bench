#!/usr/bin/env python3
"""3-way melo8k validation: ONNX (sherpa/jetson-tts runtime) vs RapidSpeech.cpp
CPU vs RapidSpeech.cpp CUDA, on the same utterances. Metric: round-trip ASR CER
(synthesize -> SenseVoice -> compare to input text). Also RS CPU<->CUDA parity."""
import os, re, subprocess, time
import numpy as np, soundfile as sf, librosa, jiwer, sherpa_onnx

ROOT = "/home/luigi/sherpa-vs-rapid"
RT = f"{ROOT}/third_party/RapidSpeech.cpp/build-cuda/rs-tts-offline"
MELO_RS = f"{ROOT}/models/rapidspeech/openvoice2-melo8k-zh.gguf"
MELO_ONNX = f"{ROOT}/models/sherpa/melo8k"
SV = f"{ROOT}/models/sherpa/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"

UTTS = [
    "人工智能正在改变世界",
    "今天天气很好我们一起去公园散步吧",
    "欢迎致电请问有什么可以帮您",
    "您的快递已经送达请注意查收",
    "请在听到提示音后留言",
    "谢谢您的来电再见",
    "语音合成技术越来越成熟了",
    "请稍等我帮您转接相关部门",
]

rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model=f"{SV}/model.int8.onnx", tokens=f"{SV}/tokens.txt", num_threads=4, use_itn=True)

def asr(wav_path):
    a, sr = sf.read(wav_path)
    if a.ndim > 1: a = a.mean(1)
    if sr != 16000: a = librosa.resample(a.astype("float32"), orig_sr=sr, target_sr=16000)
    s = rec.create_stream(); s.accept_waveform(16000, a.astype("float32")); rec.decode_stream(s)
    return re.sub(r"<[^>]*>|<\|[^|]*\|>", "", s.result.text).strip()

def norm(t):  # keep CJK + alnum only for CER
    return re.sub(r"[^一-鿿0-9a-zA-Z]", "", t)

# ONNX / sherpa runtime (the jetson-tts deployment path)
cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
        model=f"{MELO_ONNX}/model.onnx", lexicon=f"{MELO_ONNX}/lexicon.txt",
        tokens=f"{MELO_ONNX}/tokens.txt"), provider="cpu", num_threads=4))
onnx_tts = sherpa_onnx.OfflineTts(cfg)

def gen_onnx(text, out):
    a = onnx_tts.generate(text, sid=0, speed=1.0)
    sf.write(out, np.array(a.samples), a.sample_rate)

def gen_rs(text, out, gpu):
    env = dict(os.environ)
    if gpu: env["RS_FORCE_GPU"] = "1"
    subprocess.run([RT, "-m", MELO_RS, "-t", text, "-o", out, "--lang", "Chinese",
                    "--gpu", "true" if gpu else "false", "--threads", "4"],
                   capture_output=True, env=env)

os.makedirs("/tmp/melo_cmp", exist_ok=True)
rows = []
cer = {"onnx": [], "rs_cpu": [], "rs_cuda": []}
parity = []
for i, u in enumerate(UTTS):
    po = f"/tmp/melo_cmp/{i}_onnx.wav"; pc = f"/tmp/melo_cmp/{i}_rscpu.wav"; pg = f"/tmp/melo_cmp/{i}_rscuda.wav"
    gen_onnx(u, po); gen_rs(u, pc, False); gen_rs(u, pg, True)
    ho, hc, hg = asr(po), asr(pc), asr(pg)
    ref = norm(u)
    co = jiwer.cer(ref, norm(ho)); cc = jiwer.cer(ref, norm(hc)); cg = jiwer.cer(ref, norm(hg))
    cer["onnx"].append(co); cer["rs_cpu"].append(cc); cer["rs_cuda"].append(cg)
    # CPU vs CUDA waveform parity (same seed -> should match)
    wc, _ = sf.read(pc); wg, _ = sf.read(pg); n = min(len(wc), len(wg))
    corr = np.corrcoef(wc[:n], wg[:n])[0, 1] if n > 1 else float("nan")
    parity.append(corr)
    rows.append((u, ho, hc, hg, co, cc, cg, corr))

print("="*100)
print(f"{'input':<22}{'ONNX-ASR':<20}{'RS-CPU-ASR':<20}{'RS-CUDA-ASR':<20}{'cer_o':>6}{'cer_c':>6}{'cer_g':>6}{'cpu~gpu':>9}")
print("-"*100)
for u, ho, hc, hg, co, cc, cg, corr in rows:
    print(f"{u[:20]:<22}{ho[:18]:<20}{hc[:18]:<20}{hg[:18]:<20}{co:>6.2f}{cc:>6.2f}{cg:>6.2f}{corr:>9.4f}")
print("-"*100)
print(f"MEAN CER:  ONNX={np.mean(cer['onnx']):.3f}  RS-CPU={np.mean(cer['rs_cpu']):.3f}  RS-CUDA={np.mean(cer['rs_cuda']):.3f}")
print(f"MEAN RS CPU<->CUDA waveform corr: {np.nanmean(parity):.4f}")
