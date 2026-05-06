"""Side-by-side interactive viewer: BPSK | QPSK | 256-QAM
   All at the same Eb/N0 for a fair comparison."""
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

EBN0_DB = 25.0
SPS=8; RRC_BETA=0.35; RRC_SPAN=11
FREQ_OFFSET=0.003; PHASE_OFFSET=1.2
N_INFO = 4000

# ============== shared ==============
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
preamble = np.tile([1,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0],4).astype(int)
preamble_qam = np.tile([1,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0],16).astype(int)

def coarse_freq(mfa, p):
    sq = mfa**p
    Nf = 1<<int(np.ceil(np.log2(len(sq)))+1)
    F = np.fft.fftshift(np.fft.fft(sq,Nf))
    fr = np.fft.fftshift(np.fft.fftfreq(Nf))
    return fr[np.argmax(np.abs(F))]/p

# ============== BPSK ==============
def bpsk_chain(seed):
    rng = np.random.default_rng(seed)
    info = rng.integers(0,2,N_INFO)
    bits = np.concatenate([preamble, info])
    bits = np.concatenate([bits, np.zeros((-len(bits))%4, dtype=int)])
    cw = henc(bits)
    cwd = np.concatenate([[0], np.cumsum(cw)%2]).astype(int)
    s = (2*cwd-1).astype(complex)
    up = np.zeros(len(s)*SPS, dtype=complex); up[::SPS]=s
    tx = np.convolve(up, g)
    sigma = np.sqrt(1/10**(EBN0_DB/10))
    n = np.arange(len(tx))
    nz = (rng.standard_normal(len(tx))+1j*rng.standard_normal(len(tx)))*sigma/np.sqrt(2)
    rx = tx*np.exp(1j*(2*np.pi*FREQ_OFFSET*n + PHASE_OFFSET)) + nz
    mf = np.convolve(rx, g)
    delay = RRC_SPAN*SPS
    mfa = mf[delay:delay+len(s)*SPS]
    pre = mfa[::SPS]
    fo = coarse_freq(mfa, 2)
    coa = (mfa*np.exp(-1j*2*np.pi*fo*np.arange(len(mfa))))[::SPS]
    out = np.zeros_like(coa); phi=0; freq=0
    for i in range(len(coa)):
        a = 0.10 if i<120 else 0.005; b=a*a/4
        out[i] = coa[i]*np.exp(-1j*phi)
        e = np.real(out[i])*np.imag(out[i])
        freq+=b*e; phi+=freq+a*e
    return pre, coa, out

# ============== QPSK ==============
GRAY2 = np.array([0,1,3,2])
def b2d(b):
    p = b.reshape(-1,2); return GRAY2[p[:,0]*2+p[:,1]]
def phase_to_qpsk(p): return np.exp(1j*(p*np.pi/2 + np.pi/4))

def qpsk_chain(seed):
    rng = np.random.default_rng(seed)
    info = rng.integers(0,2,N_INFO)
    bits = np.concatenate([preamble, info])
    bits = np.concatenate([bits, np.zeros((-len(bits))%4, dtype=int)])
    cw = henc(bits)
    cw = np.concatenate([cw, np.zeros((-len(cw))%2, dtype=int)])
    db = b2d(cw)
    pidx = np.concatenate([[0], np.cumsum(db)%4]).astype(int)
    s = phase_to_qpsk(pidx)
    up = np.zeros(len(s)*SPS, dtype=complex); up[::SPS]=s
    tx = np.convolve(up, g)
    sigma = np.sqrt(0.5/10**(EBN0_DB/10))
    n = np.arange(len(tx))
    nz = (rng.standard_normal(len(tx))+1j*rng.standard_normal(len(tx)))*sigma/np.sqrt(2)
    rx = tx*np.exp(1j*(2*np.pi*FREQ_OFFSET*n + PHASE_OFFSET)) + nz
    mf = np.convolve(rx, g)
    delay = RRC_SPAN*SPS
    mfa = mf[delay:delay+len(s)*SPS]
    pre = mfa[::SPS]
    fo = coarse_freq(mfa, 4)
    coa = (mfa*np.exp(-1j*2*np.pi*fo*np.arange(len(mfa))))[::SPS]
    out = np.zeros_like(coa); phi=0; freq=0
    for i in range(len(coa)):
        a = 0.10 if i<120 else 0.005; b=a*a/4
        out[i] = coa[i]*np.exp(-1j*phi)
        re=np.real(out[i]); im=np.imag(out[i])
        e = (1.0 if re>=0 else -1.0)*im - (1.0 if im>=0 else -1.0)*re
        freq+=b*e; phi+=freq+a*e
    return pre, coa, out

# ============== 256-QAM ==============
NAT16 = np.arange(16)
GRAY4 = NAT16 ^ (NAT16>>1)
INV_GRAY4 = np.argsort(GRAY4)
QAM_NORM = np.sqrt(170.0)
def bits_to_qam256(b):
    g8 = b.reshape(-1, 8)
    binI = g8[:,0]*8+g8[:,1]*4+g8[:,2]*2+g8[:,3]
    binQ = g8[:,4]*8+g8[:,5]*4+g8[:,6]*2+g8[:,7]
    lvI = 2*INV_GRAY4[binI]-15; lvQ = 2*INV_GRAY4[binQ]-15
    return (lvI + 1j*lvQ)/QAM_NORM
def hard_qam(z):
    zr = z*QAM_NORM
    lvI = np.clip(2*np.round((np.real(zr)+15)/2).astype(int)-15,-15,15)
    lvQ = np.clip(2*np.round((np.imag(zr)+15)/2).astype(int)-15,-15,15)
    return (lvI + 1j*lvQ)/QAM_NORM

def qam256_chain(seed):
    rng = np.random.default_rng(seed)
    info = rng.integers(0,2,N_INFO)
    bits = np.concatenate([preamble_qam, info])
    bits = np.concatenate([bits, np.zeros((-len(bits))%4, dtype=int)])
    cw = henc(bits)
    cw = np.concatenate([cw, np.zeros((-len(cw))%8, dtype=int)])
    s = bits_to_qam256(cw)
    pre_sym = s[: len(henc(preamble_qam))//8]
    up = np.zeros(len(s)*SPS, dtype=complex); up[::SPS]=s
    tx = np.convolve(up, g)
    sigma = np.sqrt((1.0/8.0)/10**(EBN0_DB/10))
    n = np.arange(len(tx))
    nz = (rng.standard_normal(len(tx))+1j*rng.standard_normal(len(tx)))*sigma/np.sqrt(2)
    rx = tx*np.exp(1j*(2*np.pi*FREQ_OFFSET*n + PHASE_OFFSET)) + nz
    mf = np.convolve(rx, g)
    delay = RRC_SPAN*SPS
    mfa = mf[delay:delay+len(s)*SPS]
    pre = mfa[::SPS]
    fo = coarse_freq(mfa, 4)
    coa = (mfa*np.exp(-1j*2*np.pi*fo*np.arange(len(mfa))))[::SPS]
    # data-aided acquisition over preamble -> DD tracking
    # Init phi from preamble cross-correlation; alpha_trk=0.05 is wide enough
    # to track residual freq drift after coarse correction.
    n_pa = len(pre_sym); alpha_pa = 0.30; alpha_trk = 0.05
    N_init = min(20, n_pa)
    phi = float(np.angle(np.sum(coa[:N_init]*np.conj(pre_sym[:N_init]))))
    out = np.zeros_like(coa); freq = 0.0
    for i in range(len(coa)):
        rotated = coa[i]*np.exp(-1j*phi)
        out[i] = rotated
        if i < n_pa:
            d = pre_sym[i]; a = alpha_pa
        else:
            d = hard_qam(np.array([rotated]))[0]; a = alpha_trk
        denom = np.abs(d)**2
        e = np.imag(rotated*np.conj(d))/max(denom,1e-3) if denom>0 else 0.0
        b = a*a/4
        freq+=b*e; phi+=freq+a*e
    # ambiguity resolve
    SETTLE=10
    n_corr = len(pre_sym) - SETTLE
    win = out[SETTLE:SETTLE+n_corr]
    refp = pre_sym[SETTLE:SETTLE+n_corr]
    br, bs = 0, -np.inf
    for kk in range(4):
        rot = np.exp(-1j*kk*np.pi/2)
        ssc = np.real(np.sum(win*rot*np.conj(refp)))
        if ssc > bs: bs=ssc; br=kk
    fin = out * np.exp(-1j*br*np.pi/2)
    return pre, coa, fin

print(f"running 3 chains at Eb/N0 = {EBN0_DB} dB ...")
b_pre, b_coa, b_cos = bpsk_chain(7)
q_pre, q_coa, q_cos = qpsk_chain(7)
m_pre, m_coa, m_cos = qam256_chain(7)

# ============== plot ==============
fig, axes = plt.subplots(3, 4, figsize=(15.5, 11.4), facecolor="#0b1020")
fig.canvas.manager.set_window_title(f"BPSK | QPSK | 256-QAM constellations  @ Eb/N0={EBN0_DB} dB")

def panel(ax, samples, title, color, lim, qam_grid=False):
    ax.set_facecolor("#0b1020")
    ax.scatter(np.real(samples), np.imag(samples), s=3, alpha=0.3, color=color)
    if qam_grid:
        for lv in (np.arange(-15,16,2))/QAM_NORM:
            ax.axhline(lv, color="0.30", lw=0.25)
            ax.axvline(lv, color="0.30", lw=0.25)
    ax.axhline(0, color="w", lw=0.4); ax.axvline(0, color="w", lw=0.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(title, color="w", fontsize=9)
    ax.tick_params(colors="w")
    for sp in ax.spines.values(): sp.set_color("w")
    ax.grid(False)

SS_B=int(0.7*len(b_cos)); SS_Q=int(0.7*len(q_cos)); SS_M=int(0.7*len(m_cos))
rows = [
    ("BPSK", b_pre, b_coa, b_cos, b_cos[SS_B:], "#4ec9b0", 2.0, False),
    ("QPSK", q_pre, q_coa, q_cos, q_cos[SS_Q:], "#e377c2", 1.6, False),
    ("256-QAM", m_pre, m_coa, m_cos, m_cos[SS_M:], "#9cdcfe", 1.4, True),
]
col_labels = ["no sync", "+ coarse freq corr", "+ Costas (full)", "steady-state (last 30%)"]
for r,(name,p1,p2,p3,p4,col,lim,grid) in enumerate(rows):
    sigs = [p1,p2,p3,p4]
    for c,(s,cl) in enumerate(zip(sigs, col_labels)):
        panel(axes[r,c], s, f"{name} — {cl}", col, lim, qam_grid=(grid and c==3))

fig.suptitle(f"BPSK (top)  /  QPSK (middle)  /  256-QAM (bottom)  —  same Eb/N0 = {EBN0_DB} dB,"
             f"  same channel impairments (freq off {FREQ_OFFSET:.4f}, phase {PHASE_OFFSET:.2f} rad)",
             color="w", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96])
plt.show()
