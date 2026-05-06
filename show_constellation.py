"""Interactive constellation viewer — runs the chain at Eb/N0=9 dB and shows
   the 4-stage constellation in a matplotlib window that stays open."""
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(7)
N_INFO=4000; SPS=8; RRC_BETA=0.35; RRC_SPAN=11
EBN0_DB=15.0; FREQ_OFFSET=0.003; PHASE_OFFSET=1.2

G = np.array([[1,0,0,0,1,1,0],[0,1,0,0,1,0,1],[0,0,1,0,0,1,1],[0,0,0,1,1,1,1]])
def henc(b): return ((b.reshape(-1,4)@G)%2).flatten()
def diff_encode(b): return np.concatenate([[0], np.cumsum(b)%2]).astype(int)
def diff_decode(b): return np.bitwise_xor(b[1:], b[:-1])
def rrc(beta,sps,span):
    N=span*sps; t=(np.arange(N+1)-N/2)/sps; h=np.zeros_like(t)
    for i,ti in enumerate(t):
        if abs(ti)<1e-10: h[i]=1+beta*(4/np.pi-1)
        elif abs(abs(ti)-1/(4*beta))<1e-10:
            h[i]=(beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))+(1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            n=np.sin(np.pi*ti*(1-beta))+4*beta*ti*np.cos(np.pi*ti*(1+beta))
            d=np.pi*ti*(1-(4*beta*ti)**2); h[i]=n/d
    return h/np.sqrt(np.sum(h**2))

preamble=np.tile([1,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0],4).astype(int)
inf=RNG.integers(0,2,N_INFO)
all_b=np.concatenate([preamble,inf])
all_b=np.concatenate([all_b,np.zeros((-len(all_b))%4,dtype=int)])
cw=henc(all_b); cwd=diff_encode(cw)
sym=(2*cwd-1).astype(complex)
g=rrc(RRC_BETA,SPS,RRC_SPAN)
up=np.zeros(len(sym)*SPS,dtype=complex); up[::SPS]=sym
tx=np.convolve(up,g)
sigma=np.sqrt(1/10**(EBN0_DB/10))
n=np.arange(len(tx))
noise=(RNG.standard_normal(len(tx))+1j*RNG.standard_normal(len(tx)))*sigma/np.sqrt(2)
rx=tx*np.exp(1j*(2*np.pi*FREQ_OFFSET*n+PHASE_OFFSET))+noise
mf=np.convolve(rx,g)
delay=RRC_SPAN*SPS
mfa=mf[delay:delay+len(sym)*SPS]
samp_pre=mfa[::SPS]
sq=mfa**2
NFFT=1<<int(np.ceil(np.log2(len(sq)))+1)
F=np.fft.fftshift(np.fft.fft(sq,NFFT))
fr=np.fft.fftshift(np.fft.fftfreq(NFFT))
fo=fr[np.argmax(np.abs(F))]/2
mfc=mfa*np.exp(-1j*2*np.pi*fo*np.arange(len(mfa)))
samp_coarse=mfc[::SPS]

def costas2(x,alpha_acq=0.10,n_acq=120,alpha_trk=0.005):
    out=np.zeros_like(x); phi=0.0; freq=0.0
    for i in range(len(x)):
        a=alpha_acq if i<n_acq else alpha_trk; b=a*a/4
        out[i]=x[i]*np.exp(-1j*phi)
        e=np.real(out[i])*np.imag(out[i])
        freq+=b*e; phi+=freq+a*e
    return out
samp_costas=costas2(samp_coarse)
hd=(np.real(samp_costas[:len(sym)])>0).astype(int)
ber_pre=np.mean(diff_decode(hd)!=cw)

SS=int(0.7*len(samp_costas))
fig, axes = plt.subplots(1,4, figsize=(15,4.6), facecolor="#0b1020")
fig.canvas.manager.set_window_title(f"BPSK constellation @ Eb/N0={EBN0_DB} dB  pre-FEC BER={ber_pre:.2e}")
panels=[("1. no sync",samp_pre),("2. + coarse freq corr",samp_coarse),
        ("3. + Costas (full run)",samp_costas),(f"4. steady-state (last 30%)",samp_costas[SS:])]
for ax,(name,s) in zip(axes,panels):
    ax.set_facecolor("#0b1020")
    ax.scatter(np.real(s),np.imag(s),s=5,alpha=0.4,color="#4ec9b0")
    ax.axhline(0,color="w",lw=0.4); ax.axvline(0,color="w",lw=0.4)
    ax.set_xlim(-2,2); ax.set_ylim(-2,2); ax.set_aspect("equal")
    ax.set_title(name,color="w",fontsize=11)
    ax.tick_params(colors="w")
    for sp in ax.spines.values(): sp.set_color("w")
    ax.grid(alpha=0.2,color="#445")
fig.suptitle(f"BPSK modem — constellation through synchronisation   "
             f"|  freq off true {FREQ_OFFSET:.4f} / est {fo:.4f}   |  pre-FEC BER {ber_pre:.2e}",
             color="w", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.94])
plt.show()
