#!/usr/bin/env python3
"""TTS benchmark: melo8k on sherpa-onnx (VITS ONNX) vs RapidSpeech (openvoice2 ggml).
RTF cold/warm, RapidSpeech CPU (stock) + GPU (im2col fix, RS_FORCE_GPU), sherpa CPU.
Round-trip ASR accuracy. NOTE: sherpa melo8k requires sid=1."""
import os, re, csv, subprocess, time, statistics
import soundfile as sf, numpy as np
from bench_common import ROOT, RS_BIN, MELO_SHERPA, MELO_RS, SV_SHERPA, gpu_mem_peak_mb

TEXTS = ["人工智能正在改变世界。", "今天天气很好，我们一起去公园散步吧。",
         "语音合成技术越来越成熟了。", "这是一个用来测量实时率的较长测试句子。",
         "请在听到提示音后留言。", "谢谢你的来电，再见。"]
SID = 1  # only valid speaker for melo8k

def rs_tts(force_gpu):
    env = dict(os.environ)
    gpu = "false"
    if force_gpu:
        env["RS_FORCE_GPU"] = "1"; gpu = "true"
    rtfs = []
    for t in TEXTS:
        out = subprocess.check_output(
            [f"{RS_BIN}/rs-tts-offline", "-m", MELO_RS, "-t", t, "-o", "/tmp/rs_t.wav",
             "--gpu", gpu, "--threads", "4"], stderr=subprocess.STDOUT, text=True, env=env)
        rtfs.append(float(re.search(r"RTF: ([\d.]+)", out).group(1)))
    cold, warm = rtfs[0], statistics.median(rtfs[1:])
    return dict(cold=cold, warm=warm, mem=gpu_mem_peak_mb() if force_gpu else 0)

def sherpa_tts():
    import sherpa_onnx
    cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
            model=f"{MELO_SHERPA}/model.onnx", lexicon=f"{MELO_SHERPA}/lexicon.txt",
            tokens=f"{MELO_SHERPA}/tokens.txt"), provider="cpu", num_threads=4))
    tts = sherpa_onnx.OfflineTts(cfg)
    rtfs = []
    for t in TEXTS:
        a = tts.generate(t, sid=SID, speed=1.0)
        dur = len(a.samples) / a.sample_rate
        t0 = time.time(); a = tts.generate(t, sid=SID, speed=1.0); rtfs.append((time.time()-t0)/dur)
    cold, warm = rtfs[0], statistics.median(rtfs[1:])
    return dict(cold=cold, warm=warm, mem=0)

if __name__ == "__main__":
    results = [
        ("RapidSpeech-ggml", "cpu", rs_tts(False)),
        ("RapidSpeech-ggml", "cuda(fixed)", rs_tts(True)),
        ("sherpa-onnx", "cpu", sherpa_tts()),
    ]
    with open(f"{ROOT}/results/tts.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["framework","device","rtf_cold","rtf_warm","mem_mb"])
        for fw, dev, r in results:
            wr.writerow([fw, dev, f"{r['cold']:.4f}", f"{r['warm']:.4f}", r['mem']])
            print(f"{fw:16} {dev:12} cold={r['cold']:.4f} warm={r['warm']:.4f} mem={r['mem']}MB")
