"""Interactive 256-QAM constellation viewer at the current operating SNR."""
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

EBN0_DB = 25.0
SPS=8; RRC_BETA=0.35; RRC_SPAN=11
FREQ_OFFSET=0.003; PHASE_OFFSET=1.2
N_INFO = 80000

G = np.array([[1,0,0,0,1,1,0],[0,1,0,0,1,0,1],[0,0,1,0,0,1,1],[0,0,0,1,1,1,1]])
def henc(b): return ((b.reshape(-1,4)@G)%2).flatten()
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
g = rrc(RRC_BETA, SPS, RRC_SPAN)

NAT = np.arange(16); GRAY4 = NAT^(NAT>>1); INV_GRAY4 = np.argsort(GRAY4)
QN = np.sqrt(170.0)
def b2q(b):
    g8 = b.reshape(-1, 8)
    bI = g8[:,0]*8+g8[:,1]*4+g8[:,2]*2+g8[:,3]
    bQ = g8[:,4]*8+g8[:,5]*4+g8[:,6]*2+g8[:,7]
    return (2*INV_GRAY4[bI]-15 + 1j*(2*INV_GRAY4[bQ]-15))/QN
def hard_q(z):
    zr = z*QN
    lI = np.clip(2*np.round((np.real(zr)+15)/2).astype(int)-15, -15, 15)
    lQ = np.clip(2*np.round((np.imag(zr)+15)/2).astype(int)-15, -15, 15)
    return (lI + 1j*lQ)/QN

print(f"Running 256-QAM chain at Eb/N0 = {EBN0_DB} dB ...")
RNG = np.random.default_rng(7)
preamble = np.tile([1,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0],16).astype(int)
info = RNG.integers(0,2,N_INFO)
all_b = np.concatenate([preamble, info])
all_b = np.concatenate([all_b, np.zeros((-len(all_b))%4, dtype=int)])
cw = henc(all_b)
cw = np.concatenate([cw, np.zeros((-len(cw))%8, dtype=int)])
sym_tx = b2q(cw)
N_SYM = len(sym_tx)
preamble_sym = sym_tx[: len(henc(preamble))//8]

up = np.zeros(N_SYM*SPS, dtype=complex); up[::SPS] = sym_tx
tx = np.convolve(up, g)
sigma = np.sqrt((1/8)/10**(EBN0_DB/10))
n = np.arange(len(tx))
nz = (RNG.standard_normal(len(tx)) + 1j*RNG.standard_normal(len(tx)))*sigma/np.sqrt(2)
rx = tx*np.exp(1j*(2*np.pi*FREQ_OFFSET*n + PHASE_OFFSET)) + nz
mf = np.convolve(rx, g)
delay = RRC_SPAN*SPS
mfa = mf[delay:delay+N_SYM*SPS]
samp_pre = mfa[::SPS]

sq = mfa**4
Nf = 1<<int(np.ceil(np.log2(len(sq)))+1)
F = np.fft.fftshift(np.fft.fft(sq, Nf))
fr = np.fft.fftshift(np.fft.fftfreq(Nf))
fo = fr[np.argmax(np.abs(F))]/4
mfc = mfa*np.exp(-1j*2*np.pi*fo*np.arange(len(mfa)))
samp_coarse = mfc[::SPS]

# data-aided Costas (init from preamble correlation, then DA -> DD)
N_init = min(20, len(preamble_sym))
phi = float(np.angle(np.sum(samp_coarse[:N_init]*np.conj(preamble_sym[:N_init]))))
freq = 0.0
samp_costas = np.zeros_like(samp_coarse)
n_pa = len(preamble_sym)
alpha_pa = 0.30; alpha_trk = 0.05
for i in range(len(samp_coarse)):
    rotated = samp_coarse[i]*np.exp(-1j*phi)
    samp_costas[i] = rotated
    if i < n_pa:
        d = preamble_sym[i]; a = alpha_pa
    else:
        d = hard_q(np.array([rotated]))[0]; a = alpha_trk
    denom = np.abs(d)**2
    e = np.imag(rotated*np.conj(d))/max(denom,1e-3) if denom>0 else 0.0
    b = a*a/4
    freq += b*e; phi += freq + a*e

# ambiguity resolve
SETTLE=10; n_corr = len(preamble_sym)-SETTLE
win = samp_costas[SETTLE:SETTLE+n_corr]
ref = preamble_sym[SETTLE:SETTLE+n_corr]
br, bs = 0, -np.inf
for k in range(4):
    rot = np.exp(-1j*k*np.pi/2)
    sc_ = np.real(np.sum(win*rot*np.conj(ref)))
    if sc_>bs: bs=sc_; br=k
samp_final = samp_costas * np.exp(-1j*br*np.pi/2)

# stats
hd = hard_q(samp_final[:N_SYM])
ser = np.mean(hd != sym_tx[:N_SYM])
SS = int(0.7*N_SYM)
hd_ss = hard_q(samp_final[SS:N_SYM])
ser_ss = np.mean(hd_ss != sym_tx[SS:N_SYM])
print(f"  fo true={FREQ_OFFSET:.4f}  est={fo:.4f}")
print(f"  SER pre-FEC (full run)    = {ser:.3e}    ({int(ser*N_SYM)} / {N_SYM} sym)")
print(f"  SER pre-FEC (steady-state)= {ser_ss:.3e}  ({int(ser_ss*(N_SYM-SS))} / {N_SYM-SS} sym)")

# plot
fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), facecolor="#0b1020")
fig.canvas.manager.set_window_title(
    f"256-QAM constellation @ Eb/N0={EBN0_DB} dB   SER(steady)={ser_ss:.2e}")

panels = [("1. no sync",                samp_pre,      1.4),
          ("2. + coarse freq corr",     samp_coarse,   1.4),
          ("3. + Costas (full run)",    samp_final,    1.4),
          (f"4. steady-state (last 30%)", samp_final[SS:], 1.3)]

grid_levels = (np.arange(-15,16,2))/QN
for ax,(name,s,L) in zip(axes, panels):
    ax.set_facecolor("#0b1020")
    ax.scatter(np.real(s), np.imag(s), s=4, alpha=0.45, color="#9cdcfe")
    if "steady" in name or "Costas" in name:
        for lv in grid_levels:
            ax.axhline(lv, color="0.30", lw=0.25)
            ax.axvline(lv, color="0.30", lw=0.25)
    ax.axhline(0, color="w", lw=0.4); ax.axvline(0, color="w", lw=0.4)
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
    ax.set_title(name, color="w", fontsize=11)
    ax.tick_params(colors="w")
    for sp in ax.spines.values(): sp.set_color("w")
    ax.grid(False)

fig.suptitle(f"256-QAM modem — Eb/N0 = {EBN0_DB} dB    "
             f"SER(steady-state) = {ser_ss:.2e}    "
             f"freq off true {FREQ_OFFSET:.4f} / est {fo:.4f}",
             color="w", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.94])
plt.show()
