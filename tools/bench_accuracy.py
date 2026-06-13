#!/usr/bin/env python3
"""Accuracy pass: STT CER (sherpa vs RapidSpeech, same SenseVoice), X-ASR sample,
TTS round-trip CER, VAD frame-agreement vs TEN-VAD testset labels."""
import os, re, subprocess, glob
import numpy as np, soundfile as sf, librosa, jiwer, sherpa_onnx

ROOT="/home/luigi/sherpa-vs-rapid"; GEN1="/tmp/rs-gen1/build-cuda"
SV_O=f"{ROOT}/models/sherpa/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
SV_G=f"{ROOT}/models/rapidspeech/ASR/SenseVoice/sense-voice-small-q5_k.gguf"
MELO_O=f"{ROOT}/models/sherpa/melo8k"; MELO_G=f"{ROOT}/models/rapidspeech/openvoice2-melo8k-zh.gguf"
SIL_O=f"{ROOT}/models/sherpa/silero_vad.onnx"; TEN_O=f"{ROOT}/models/sherpa/ten-vad.onnx"
XASR=f"{ROOT}/models/xasr/deployment/models/chunk-480ms-model"; TEN_TS=f"{ROOT}/audio/ten-vad/testset"
RS_ENV=dict(os.environ,LD_LIBRARY_PATH=f"{GEN1}:{GEN1}/ggml/src:{GEN1}/ggml/src/ggml-cuda:"+os.environ.get("LD_LIBRARY_PATH",""))
def nrm(t): return re.sub(r"[^一-鿿0-9a-zA-Z]","",t)
def strip(t): return re.sub(r"<[^>]*>|<\|[^|]*\|>","",t).strip()

rec=sherpa_onnx.OfflineRecognizer.from_sense_voice(model=f"{SV_O}/model.int8.onnx",tokens=f"{SV_O}/tokens.txt",num_threads=4,use_itn=True)
def sv_sherpa(w):
    a,sr=sf.read(w); s=rec.create_stream(); s.accept_waveform(sr,a.astype('float32')); rec.decode_stream(s); return strip(s.result.text)
def sv_rs(w):
    o=subprocess.check_output([f"{GEN1}/rs-asr-offline","-m",SV_G,"-w",w,"--gpu","false"],stderr=subprocess.STDOUT,text=True,env=RS_ENV)
    m=re.search(r"→ [\d:.]+\]\s*(.+)",o); return strip(m.group(1)) if m else ""

print("===== STT: SenseVoice sherpa vs RapidSpeech (agreement CER) =====")
wavs=sorted(glob.glob(f"{SV_O}/test_wavs/*.wav")); cers=[]
for w in wavs:
    hs,hr=sv_sherpa(w),sv_rs(w); c=jiwer.cer(nrm(hs),nrm(hr)); cers.append(c)
    print(f"  {os.path.basename(w):8} sherpa='{hs[:24]}' rs='{hr[:24]}' CER={c:.3f}")
print(f"  MEAN sherpa<->RS CER: {np.mean(cers):.3f}")

print("===== X-ASR-480ms (sherpa only) sample transcripts =====")
xr=sherpa_onnx.OnlineRecognizer.from_transducer(encoder=f"{XASR}/encoder-480ms.int8.onnx",decoder=f"{XASR}/decoder-480ms.onnx",joiner=f"{XASR}/joiner-480ms.int8.onnx",tokens=f"{XASR}/tokens.txt",num_threads=4,provider="cpu")
for w in wavs[:3]:
    a,sr=sf.read(w); s=xr.create_stream(); s.accept_waveform(sr,a.astype('float32')); s.accept_waveform(sr,np.zeros(8000,dtype='float32')); s.input_finished()
    while xr.is_ready(s): xr.decode_stream(s)
    print(f"  {os.path.basename(w):8} X-ASR='{xr.get_result(s)[:40]}'")

print("===== TTS: melo8k round-trip CER (synth -> SenseVoice -> vs input) =====")
TEXTS=["人工智能正在改变世界","欢迎致电请问有什么可以帮您","您的快递已经送达请注意查收","请在听到提示音后留言"]
cfg=sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=f"{MELO_O}/model.onnx",lexicon=f"{MELO_O}/lexicon.txt",tokens=f"{MELO_O}/tokens.txt"),provider="cpu",num_threads=4)); ot=sherpa_onnx.OfflineTts(cfg)
def rtasr(wav):
    a,sr=sf.read(wav); a=a.mean(1) if a.ndim>1 else a; w=librosa.resample(a.astype('float32'),orig_sr=sr,target_sr=16000)
    s=rec.create_stream(); s.accept_waveform(16000,w); rec.decode_stream(s); return strip(s.result.text)
sc,rc=[],[]
for t in TEXTS:
    sf.write("/tmp/o.wav",np.array(ot.generate(t,sid=0).samples),8000)
    subprocess.run([f"{GEN1}/rs-tts-offline","-m",MELO_G,"-t",t,"-o","/tmp/r.wav","--lang","Chinese","--gpu","false"],capture_output=True,env=RS_ENV)
    cs=jiwer.cer(nrm(t),nrm(rtasr("/tmp/o.wav"))); cr=jiwer.cer(nrm(t),nrm(rtasr("/tmp/r.wav"))); sc.append(cs); rc.append(cr)
    print(f"  '{t[:14]}' sherpa_rtCER={cs:.2f} rapidspeech_rtCER={cr:.2f}")
print(f"  MEAN round-trip CER: sherpa={np.mean(sc):.3f} rapidspeech={np.mean(rc):.3f}")

print("===== VAD: frame-agreement vs TEN testset labels (30 wavs) =====")
def ten_labels(scv,sr,nf):
    lab=np.zeros(nf,dtype=np.int8); toks=open(scv).read().strip().split(","); nums=toks[1:]
    for j in range(0,len(nums)-2,3):
        st,en,fg=float(nums[j]),float(nums[j+1]),int(float(nums[j+2]))
        if fg: lab[int(st/0.01):min(int(en/0.01),nf)]=1
    return lab
def sherpa_vad_agree(model):
    cfg=sherpa_onnx.VadModelConfig()
    if "ten" in model: cfg.ten_vad.model=model; cfg.ten_vad.threshold=0.5
    else: cfg.silero_vad.model=model; cfg.silero_vad.threshold=0.5
    cfg.sample_rate=16000; cfg.provider="cpu"; cfg.num_threads=4
    ag=[]
    for scv in sorted(glob.glob(f"{TEN_TS}/*.scv"))[:30]:
        wav=scv.replace(".scv",".wav")
        if not os.path.exists(wav): continue
        a,sr=sf.read(wav); a=a.mean(1) if a.ndim>1 else a; a=a.astype('float32')
        vad=sherpa_onnx.VoiceActivityDetector(cfg,buffer_size_in_seconds=60)
        speech=np.zeros(len(a),dtype=np.int8); i=0
        while i<len(a): vad.accept_waveform(a[i:i+512]); i+=512
        vad.flush()
        while not vad.empty():
            seg=vad.front; s0=int(seg.start); speech[s0:min(s0+len(seg.samples),len(a))]=1; vad.pop()
        fl=int(0.01*sr); nf=len(a)//fl
        lab=ten_labels(scv,sr,nf)
        pred=np.array([1 if speech[k*fl:(k+1)*fl].mean()>0.5 else 0 for k in range(nf)])
        ag.append((pred==lab[:nf]).mean())
    return float(np.mean(ag)), len(ag)
for mname,m in [("silero-VAD",SIL_O),("TEN-VAD",TEN_O)]:
    a,n=sherpa_vad_agree(m); print(f"  sherpa {mname:11} frame-agreement vs TEN labels: {a:.3f} ({n} wavs)")
