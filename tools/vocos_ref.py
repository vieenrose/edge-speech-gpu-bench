"""NumPy reference of Vocos8k decoder from the gguf weights, fed RapidSpeech's own
dumped z (/tmp/mel_cpp.bin). Produces the exact target waveform the ggml port must match.
Mirrors student/models.py:Vocos8k.forward + student/istft.py:ISTFTHead."""
import numpy as np, gguf, sys

NFFT, HOP, NBINS = 256, 64, 129
RESAMPLE_SCALE = 125.0 / (44100/512)  # 1.45124716

def load(path):
    r = gguf.GGUFReader(path)
    W = {}
    for t in r.tensors:
        W[t.name] = np.array(t.data).reshape(tuple(int(x) for x in t.shape))  # ggml ne order
    return W

def gelu(x):  # exact (erf) gelu, torch default
    from math import sqrt
    from scipy.special import erf
    return 0.5*x*(1.0+erf(x/sqrt(2.0)))

def layernorm(x, w, b, eps=1e-5):  # x [C,T], normalize over C per frame
    m = x.mean(0, keepdims=True); v = x.var(0, keepdims=True)
    return (x-m)/np.sqrt(v+eps)*w[:,None]+b[:,None]

def conv1d(x, w, b, pad, groups=1):
    # x [Cin,T], torch weight w [Cout,Cin/groups,K]
    Cout, Cing, K = w.shape; Cin, T = x.shape
    xp = np.pad(x, ((0,0),(pad,pad)))
    out = np.zeros((Cout, T))
    if groups == 1:
        for k in range(K):
            out += w[:,:,k] @ xp[:, k:k+T]
    else:  # depthwise: groups==Cin==Cout, Cing==1
        for k in range(K):
            out += w[:,0,k][:,None] * xp[:, k:k+T]
    return out + (b[:,None] if b is not None else 0)

def resample_linear(z, scale):  # [C,T]->[C, floor(T*scale)] align_corners=False
    C, T = z.shape; L = int(T*scale)
    i = np.arange(L)
    x = (i+0.5)/scale - 0.5
    x0 = np.floor(x).astype(int); frac = x-x0
    x0c = np.clip(x0,0,T-1); x1c = np.clip(x0+1,0,T-1)
    return z[:,x0c]*(1-frac) + z[:,x1c]*frac

def main():
    W = load("models/rapidspeech/openvoice2-melo8k-zh.gguf")
    z = np.fromfile("/tmp/mel_cpp.bin", dtype=np.float32).reshape(192, -1)
    # g = emb_g[sid=1]
    emb_g = W["emb_g.weight"]  # ggml ne [256,256] = (dim, n_spk) -> col? row?
    g = emb_g[:, 1] if emb_g.shape[1] > 1 else emb_g[:,0]
    # torch-layout helpers
    def wT(name, axes): return np.transpose(W[name], axes)
    x = resample_linear(z, RESAMPLE_SCALE)
    # conv_pre [7,192,256]gguf -> torch [256,192,7]
    cpw = wT("vocoder.conv_pre.weight", (2,1,0))
    x = conv1d(x, cpw, W["vocoder.conv_pre.bias"], pad=3)
    # cond k=1 on g: [1,256,256]gguf->torch[256,256,1]
    condw = wT("vocoder.cond.weight",(2,1,0))[:,:,0]
    x = x + (condw @ g + W["vocoder.cond.bias"])[:,None]
    x = layernorm(x, W["vocoder.student.norm_in.weight"], W["vocoder.student.norm_in.bias"])
    for i in range(8):
        p=f"vocoder.student.blocks.{i}."
        res=x
        dw = wT(p+"dw.weight",(2,1,0))  # [7,1,256]->[256,1,7]
        h = conv1d(x, dw, W[p+"dw.bias"], pad=3, groups=256)
        h = layernorm(h, W[p+"norm.weight"], W[p+"norm.bias"])  # over channels
        # pw1 [256,768]gguf=(in,out)->torch[768,256]; pw as Linear over channel dim
        pw1 = W[p+"pw1.weight"].T  # [768,256]
        pw2 = W[p+"pw2.weight"].T  # [256,768]
        hh = pw1 @ h + W[p+"pw1.bias"][:,None]
        hh = gelu(hh)
        hh = pw2 @ hh + W[p+"pw2.bias"][:,None]
        hh = W[p+"gamma"][:,None]*hh
        x = res + hh
    x = layernorm(x, W["vocoder.student.norm_out.weight"], W["vocoder.student.norm_out.bias"])
    headw = W["vocoder.student.head.weight"].T  # [258,256]
    h = headw @ x + W["vocoder.student.head.bias"][:,None]  # [258,L]
    mag = np.exp(np.minimum(h[:NBINS], 9.0)); phase = h[NBINS:]
    real = mag*np.cos(phase); imag = mag*np.sin(phase)
    # istft via conv_transpose1d. cos_w gguf [256,1,129]=(n_fft,1,n_bins); torch [n_bins,1,n_fft]
    cos_w = np.transpose(W["vocoder.student.istft.cos_w"],(2,1,0))[:,0,:]  # [129,256]
    sin_w = np.transpose(W["vocoder.student.istft.sin_w"],(2,1,0))[:,0,:]
    win_sq = W["vocoder.student.istft.win_sq"].reshape(-1)  # [256]
    L = real.shape[1]; out_len=(L-1)*HOP+NFFT
    y=np.zeros(out_len); norm=np.zeros(out_len)
    for t in range(L):
        frame = real[:,t]@cos_w + imag[:,t]@sin_w  # [256]
        s=t*HOP; y[s:s+NFFT]+=frame; norm[s:s+NFFT]+=win_sq
    y = y/(norm+1e-8)
    p=NFFT//2; y=y[p:-p]
    y.astype(np.float32).tofile("/tmp/vocos_ref.bin")
    import soundfile as sf; sf.write("/tmp/vocos_ref.wav", y.astype(np.float32), 8000)
    print(f"z {z.shape} -> resampled {x.shape if False else L} frames -> wav {len(y)} samples ({len(y)/8000:.2f}s)")
    print(f"wav max={np.abs(y).max():.3f} rms={np.sqrt((y**2).mean()):.4f}")

if __name__=="__main__": main()
