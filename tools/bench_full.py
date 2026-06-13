#!/usr/bin/env python3
"""Comprehensive sherpa-onnx (ONNX) vs RapidSpeech (ggml) benchmark on GB10.
Models: SenseVoice STT, X-ASR streaming (sherpa-only), melo8k TTS, silero-VAD,
TEN-VAD (sherpa-only). RTF cold/warm, accuracy, GPU mem. CPU and CUDA.
sherpa CUDA filled in separately if/when the from-source ORT build lands."""
import os, re, csv, subprocess, time, glob, statistics, json
import numpy as np, soundfile as sf, librosa, jiwer, sherpa_onnx

ROOT = "/home/luigi/sherpa-vs-rapid"
GEN1 = "/tmp/rs-gen1/build-cuda"
SV_O = f"{ROOT}/models/sherpa/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
SV_G = f"{ROOT}/models/rapidspeech/ASR/SenseVoice/sense-voice-small-q5_k.gguf"
SIL_G = f"{ROOT}/models/rapidspeech/ASR/silero_vad_v6.gguf"
SIL_O = f"{ROOT}/models/sherpa/silero_vad.onnx"
TEN_O = f"{ROOT}/models/sherpa/ten-vad.onnx"
MELO_O = f"{ROOT}/models/sherpa/melo8k"
MELO_G = f"{ROOT}/models/rapidspeech/openvoice2-melo8k-zh.gguf"
XASR = f"{ROOT}/models/xasr/deployment/models/chunk-480ms-model"
CLIP = f"{ROOT}/audio/rtf_clip_30s.wav"
TEN_TESTSET = f"{ROOT}/audio/ten-vad/testset"
TEXTS = ["人工智能正在改变世界。", "今天天气很好。", "欢迎致电请问有什么可以帮您。",
         "您的快递已经送达。", "请在听到提示音后留言。", "谢谢您的来电再见。"]
RS_ENV = dict(os.environ, LD_LIBRARY_PATH=f"{GEN1}:{GEN1}/ggml/src:{GEN1}/ggml/src/ggml-cuda:" + os.environ.get("LD_LIBRARY_PATH",""))

def gpu_mem():
    try: return int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).split("\n")[0])
    except: return -1

def cold_warm(rtfs): return rtfs[0], statistics.median(rtfs[1:])

# ---------- sherpa runners ----------
def sherpa_sensevoice(device, iters=6):
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(model=f"{SV_O}/model.int8.onnx", tokens=f"{SV_O}/tokens.txt", num_threads=4, provider=device, use_itn=True)
    a, sr = sf.read(CLIP); dur = len(a)/sr; rt=[]
    for _ in range(iters):
        s=rec.create_stream(); s.accept_waveform(sr,a.astype('float32')); t=time.time(); rec.decode_stream(s); rt.append((time.time()-t)/dur)
    return cold_warm(rt), gpu_mem() if device=="cuda" else 0

def sherpa_xasr(device, iters=6):
    rec = sherpa_onnx.OnlineRecognizer.from_transducer(
        encoder=f"{XASR}/encoder-480ms.int8.onnx", decoder=f"{XASR}/decoder-480ms.onnx",
        joiner=f"{XASR}/joiner-480ms.int8.onnx", tokens=f"{XASR}/tokens.txt",
        num_threads=4, provider=device, decoding_method="greedy_search")
    a, sr = sf.read(CLIP); a=a.astype('float32'); dur=len(a)/sr; rt=[]
    for _ in range(iters):
        s=rec.create_stream(); t=time.time()
        for i in range(0,len(a),1600): s.accept_waveform(sr,a[i:i+1600]);
        while rec.is_ready(s): rec.decode_stream(s)
        s.accept_waveform(sr,np.zeros(8000,dtype='float32')); s.input_finished()
        while rec.is_ready(s): rec.decode_stream(s)
        rt.append((time.time()-t)/dur)
    return cold_warm(rt), gpu_mem() if device=="cuda" else 0

def sherpa_tts(device, iters=None):
    cfg=sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=f"{MELO_O}/model.onnx",lexicon=f"{MELO_O}/lexicon.txt",tokens=f"{MELO_O}/tokens.txt"),provider=device,num_threads=4))
    tts=sherpa_onnx.OfflineTts(cfg); rt=[]
    for t in TEXTS:
        a=tts.generate(t,sid=0); dur=len(a.samples)/a.sample_rate
        t0=time.time(); a=tts.generate(t,sid=0); rt.append((time.time()-t0)/dur)
    return cold_warm(rt), gpu_mem() if device=="cuda" else 0

def sherpa_vad(model, device, iters=6):
    a,sr=sf.read(CLIP); a=a.astype('float32'); dur=len(a)/sr; rt=[]
    cfg=sherpa_onnx.VadModelConfig()
    if "ten" in model: cfg.ten_vad.model=model; cfg.ten_vad.threshold=0.5
    else: cfg.silero_vad.model=model; cfg.silero_vad.threshold=0.5
    cfg.sample_rate=sr; cfg.provider=device; cfg.num_threads=4
    for _ in range(iters):
        vad=sherpa_onnx.VoiceActivityDetector(cfg,buffer_size_in_seconds=60); t=time.time()
        for i in range(0,len(a),512): vad.accept_waveform(a[i:i+512])
        vad.flush()
        while not vad.empty(): vad.pop()
        rt.append((time.time()-t)/dur)
    return cold_warm(rt), gpu_mem() if device=="cuda" else 0

# ---------- RapidSpeech runners ----------
def rs_sensevoice(gpu):
    out=subprocess.check_output([f"{GEN1}/rs-asr-offline","-m",SV_G,"-w",CLIP,"-v",SIL_G,"--gpu","true" if gpu else "false","-t","4"],stderr=subprocess.STDOUT,text=True,env=RS_ENV)
    seg=[float(m) for m in re.findall(r"RTF: ([\d.]+)",out)]
    return (seg[0], statistics.median(seg[1:]) if len(seg)>1 else seg[0]), gpu_mem() if gpu else 0

def rs_tts(gpu):
    env=dict(RS_ENV);
    if gpu: env["RS_FORCE_GPU"]="1"
    rt=[]
    for t in TEXTS:
        out=subprocess.check_output([f"{GEN1}/rs-tts-offline","-m",MELO_G,"-t",t,"-o","/tmp/b.wav","--lang","Chinese","--gpu","true" if gpu else "false","--threads","4"],stderr=subprocess.STDOUT,text=True,env=env)
        rt.append(float(re.search(r"RTF: ([\d.]+)",out).group(1)))
    return cold_warm(rt), gpu_mem() if gpu else 0

def rs_vad(gpu):
    out=subprocess.check_output(["/tmp/bench_vad_rs",SIL_G,"1" if gpu else "0"],stderr=subprocess.STDOUT,text=True,env=RS_ENV)
    m=re.search(r"RESULT \w+ cold=([\d.]+) warm=([\d.]+)",out)
    return (float(m.group(1)),float(m.group(2))), gpu_mem() if gpu else 0

if __name__=="__main__":
    R=[]  # (model, framework, device, cold, warm, mem)
    def add(model,fw,dev,fn):
        try:
            (c,w),mem=fn(); R.append((model,fw,dev,c,w,mem)); print(f"{model:14} {fw:12} {dev:5} cold={c:.4f} warm={w:.4f} mem={mem}")
        except Exception as e: print(f"{model} {fw} {dev} FAILED: {str(e)[:120]}"); R.append((model,fw,dev,float('nan'),float('nan'),-1))
    print("===== sherpa-onnx CPU =====")
    add("SenseVoice","sherpa","cpu",lambda:sherpa_sensevoice("cpu"))
    add("X-ASR-480ms","sherpa","cpu",lambda:sherpa_xasr("cpu"))
    add("melo8k","sherpa","cpu",lambda:sherpa_tts("cpu"))
    add("silero-VAD","sherpa","cpu",lambda:sherpa_vad(SIL_O,"cpu"))
    add("TEN-VAD","sherpa","cpu",lambda:sherpa_vad(TEN_O,"cpu"))
    print("===== RapidSpeech CPU =====")
    add("SenseVoice","rapidspeech","cpu",lambda:rs_sensevoice(False))
    add("melo8k","rapidspeech","cpu",lambda:rs_tts(False))
    add("silero-VAD","rapidspeech","cpu",lambda:rs_vad(False))
    print("===== RapidSpeech CUDA =====")
    add("SenseVoice","rapidspeech","cuda",lambda:rs_sensevoice(True))
    add("melo8k","rapidspeech","cuda",lambda:rs_tts(True))
    add("silero-VAD","rapidspeech","cuda",lambda:rs_vad(True))
    with open(f"{ROOT}/results/full_rtf.csv","w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["model","framework","device","rtf_cold","rtf_warm","gpu_mem_mb"]); wr.writerows(R)
    print("\nwrote results/full_rtf.csv")
