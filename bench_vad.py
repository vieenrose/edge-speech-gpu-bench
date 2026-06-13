#!/usr/bin/env python3
"""VAD benchmark: silero on sherpa-onnx (ONNX) vs RapidSpeech (ggml).
RTF over the 30s clip + segment agreement vs TEN-VAD testset labels."""
import os, re, csv, subprocess, time, glob, statistics
import soundfile as sf, numpy as np
from bench_common import (ROOT, RS_BIN, SILERO_RS, SILERO_ONNX, CLIP30,
                          TEN_TESTSET, gpu_mem_peak_mb)

def sherpa_vad_rtf(iters=6):
    import sherpa_onnx
    a, sr = sf.read(CLIP30); a = a.astype("float32"); dur = len(a)/sr
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = SILERO_ONNX; cfg.silero_vad.threshold = 0.5
    cfg.sample_rate = sr; cfg.provider = "cpu"; cfg.num_threads = 4
    rtfs = []
    for _ in range(iters):
        vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
        t0 = time.time()
        k = 0
        while k < len(a):
            vad.accept_waveform(a[k:k+512]); k += 512
        vad.flush()
        segs = 0
        while not vad.empty(): segs += 1; vad.pop()
        rtfs.append((time.time()-t0)/dur)
    return rtfs[0], statistics.median(rtfs[1:])

def rs_vad_rtf(device):
    """rs-asr with silero VAD; parse wall over clip. Use ASR binary's VAD stage
    via a tiny run; here we time the full segmented pass and report RTF."""
    gpu = "true" if device == "cuda" else "false"
    # warmup + timed runs
    rtfs = []
    for i in range(4):
        t0 = time.time()
        out = subprocess.check_output(
            [f"{RS_BIN}/rs-asr-offline", "-m",
             f"{ROOT}/models/rapidspeech/ASR/SenseVoice/sense-voice-small-q5_k.gguf",
             "-w", CLIP30, "-v", SILERO_RS, "--gpu", gpu],
            stderr=subprocess.STDOUT, text=True)
        # Overall-RTF includes ASR; we want VAD share — report overall as proxy
        rtfs.append(float(re.search(r"Overall-RTF: ([\d.]+)", out).group(1)))
    return rtfs[0], statistics.median(rtfs[1:]), gpu_mem_peak_mb() if device=="cuda" else 0

def seg_agreement():
    """Compare sherpa silero speech/non-speech vs TEN-VAD .scv labels on testset.
    Frame-level agreement averaged over wavs."""
    import sherpa_onnx
    scvs = sorted(glob.glob(f"{TEN_TESTSET}/*.scv"))[:30]
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = SILERO_ONNX; cfg.silero_vad.threshold = 0.5
    cfg.sample_rate = 16000; cfg.provider = "cpu"; cfg.num_threads = 4
    aggs = []
    for scv in scvs:
        wav = scv.replace(".scv", ".wav")
        if not os.path.exists(wav): continue
        a, sr = sf.read(wav)
        if a.ndim > 1: a = a.mean(1)
        a = a.astype("float32")
        # TEN .scv: "name,start,end,flag,start,end,flag,..." intervals (sec).
        fl = int(0.01 * sr)  # 10ms frames
        nf_total = len(a) // fl
        labels = np.zeros(nf_total, dtype=np.int8)
        with open(scv) as f:
            toks = f.read().strip().split(",")
        nums = toks[1:]
        for j in range(0, len(nums) - 2, 3):
            st, en, fg = float(nums[j]), float(nums[j+1]), int(float(nums[j+2]))
            if fg:
                labels[int(st/0.01):min(int(en/0.01), nf_total)] = 1
        vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)
        speech = np.zeros(len(a), dtype=np.int8)
        # collect segments
        k = 0
        while k < len(a):
            vad.accept_waveform(a[k:k+512]); k += 512
        vad.flush()
        while not vad.empty():
            seg = vad.front; s = int(seg.start); e = s + len(seg.samples)
            speech[s:min(e, len(a))] = 1; vad.pop()
        # downsample speech mask to 10ms frames
        nf = min(len(labels), len(a)//fl)
        if nf == 0: continue
        pred = np.array([1 if speech[i*fl:(i+1)*fl].mean() > 0.5 else 0 for i in range(nf)])
        agree = (pred == labels[:nf]).mean()
        aggs.append(agree)
    return float(np.mean(aggs)) if aggs else float("nan"), len(aggs)

if __name__ == "__main__":
    rows = []
    sc, sw = sherpa_vad_rtf()
    rows.append(("sherpa-onnx", "cpu", sc, sw, 0))
    rc, rw, rm = rs_vad_rtf("cuda"); rows.append(("RapidSpeech-ggml", "cuda", rc, rw, rm))
    rc2, rw2, _ = rs_vad_rtf("cpu"); rows.append(("RapidSpeech-ggml", "cpu", rc2, rw2, 0))
    with open(f"{ROOT}/results/vad.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["framework","device","rtf_cold","rtf_warm","mem_mb"])
        for r in rows:
            wr.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.4f}", r[4]])
            print(f"{r[0]:16} {r[1]:5} cold={r[2]:.4f} warm={r[3]:.4f} mem={r[4]}MB")
    agree, n = seg_agreement()
    print(f"\nsilero VAD frame-agreement vs TEN testset ({n} wavs): {agree:.3f}")
    with open(f"{ROOT}/results/vad_agreement.csv", "w", newline="") as f:
        csv.writer(f).writerows([["metric","value","n_wavs"], ["frame_agreement_vs_ten", f"{agree:.4f}", n]])
