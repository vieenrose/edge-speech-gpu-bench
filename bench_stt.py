#!/usr/bin/env python3
"""STT benchmark: SenseVoice on sherpa-onnx (ONNX) vs RapidSpeech (ggml).
RTF cold (first/graph-warmup) and warm (steady-state), GPU mem, CER."""
import os, re, csv, subprocess, time, glob, statistics
import soundfile as sf
from bench_common import (ROOT, RS_BIN, SV_SHERPA, SV_RS, SILERO_RS, CLIP30,
                          cudnn_env, gpu_mem_peak_mb)

# --- reference transcripts (known content of SenseVoice test_wavs) ---
import sherpa_onnx, jiwer

def clip_dur(p):
    a, sr = sf.read(p); return len(a)/sr

def rs_asr(device):  # device: 'cuda'|'cpu'
    gpu = "true" if device == "cuda" else "false"
    cmd = [f"{RS_BIN}/rs-asr-offline", "-m", SV_RS, "-w", CLIP30,
           "-v", SILERO_RS, "--gpu", gpu, "-t", "4"]
    t0 = time.time()
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    wall = time.time() - t0
    rtfs = [float(m) for m in re.findall(r"seg #\d+ \| [\d.]+s \| ASR: [\d.]+s \| RTF: ([\d.]+)", out)]
    overall = float(re.search(r"Overall-RTF: ([\d.]+)", out).group(1))
    cold = rtfs[0]; warm = statistics.median(rtfs[1:]) if len(rtfs) > 1 else cold
    return dict(cold=cold, warm=warm, overall=overall, load_wall=wall, mem=gpu_mem_peak_mb() if device=="cuda" else 0)

def sherpa_asr(device, iters=6):
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=f"{SV_SHERPA}/model.int8.onnx", tokens=f"{SV_SHERPA}/tokens.txt",
        provider=device, num_threads=4, use_itn=True)
    a, sr = sf.read(CLIP30); dur = len(a)/sr
    rtfs = []
    for _ in range(iters):
        s = rec.create_stream(); s.accept_waveform(sr, a.astype("float32"))
        t0 = time.time(); rec.decode_stream(s); rtfs.append((time.time()-t0)/dur)
    cold = rtfs[0]; warm = statistics.median(rtfs[1:])
    return dict(cold=cold, warm=warm, overall=warm, load_wall=0,
                mem=gpu_mem_peak_mb() if device=="cuda" else 0)

def cer():
    """Round-trip CER: transcribe each test_wav with both, compare to sherpa-CPU
    reference text (no ground-truth shipped). Reports CER(RS vs sherpa-CPU)."""
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=f"{SV_SHERPA}/model.int8.onnx", tokens=f"{SV_SHERPA}/tokens.txt",
        provider="cpu", num_threads=4, use_itn=True)
    rows = []
    for w in sorted(glob.glob(f"{SV_SHERPA}/test_wavs/*.wav")):
        a, sr = sf.read(w); s = rec.create_stream()
        s.accept_waveform(sr, a.astype("float32")); rec.decode_stream(s)
        ref = s.result.text.strip()
        out = subprocess.check_output(
            [f"{RS_BIN}/rs-asr-offline", "-m", SV_RS, "-w", w, "--gpu", "true"],
            stderr=subprocess.STDOUT, text=True)
        m = re.search(r"→ [\d:.]+\]\s*(.+)", out)
        hyp = re.sub(r"<\|[^|]*\|>", "", m.group(1)).strip() if m else ""
        c = jiwer.cer(ref, hyp) if ref else 0.0
        rows.append((os.path.basename(w), ref[:30], hyp[:30], round(c, 3)))
    return rows

if __name__ == "__main__":
    results = []
    print("== RapidSpeech GPU =="); results.append(("RapidSpeech-ggml", "cuda", rs_asr("cuda")))
    print("== RapidSpeech CPU =="); results.append(("RapidSpeech-ggml", "cpu", rs_asr("cpu")))
    print("== sherpa CPU =="); results.append(("sherpa-onnx", "cpu", sherpa_asr("cpu")))
    try:
        print("== sherpa GPU =="); results.append(("sherpa-onnx", "cuda", sherpa_asr("cuda")))
    except Exception as e:
        print("sherpa GPU failed:", str(e)[:150])

    with open(f"{ROOT}/results/stt.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["framework","device","rtf_cold","rtf_warm","rtf_overall","mem_mb"])
        for fw, dev, r in results:
            wr.writerow([fw, dev, f"{r['cold']:.4f}", f"{r['warm']:.4f}", f"{r['overall']:.4f}", r['mem']])
            print(f"{fw:16} {dev:5} cold={r['cold']:.4f} warm={r['warm']:.4f} overall={r['overall']:.4f} mem={r['mem']}MB")

    print("\n== CER (RapidSpeech vs sherpa-CPU ref) ==")
    crows = cer()
    with open(f"{ROOT}/results/stt_cer.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["wav","ref","hyp_rs","cer"]); wr.writerows(crows)
    for r in crows: print(r)
