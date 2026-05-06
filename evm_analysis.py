"""EVM analysis: runs each modem at 3 Eb/N0 points, computes reference-based
EVM over the post-sync steady-state portion, and saves a 3-panel constellation
plot per modem with EVM annotations.

  EVM_rms = sqrt( mean(|rx - tx_known|^2) / mean(|tx_known|^2) )

Outputs:
  bpsk_out/evm_vs_snr.png
  qpsk_out/evm_vs_snr.png
  qam256_out/evm_vs_snr.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = os.path.dirname(os.path.abspath(__file__))
SPS      = 8
RRC_BETA = 0.35
RRC_SPAN = 11
FREQ_OFFSET  = 0.003
PHASE_OFFSET = 1.20
N_INFO   = 4000

# ---------- Hamming(7,4) ----------
G = np.array([[1,0,0,0,1,1,0],[0,1,0,0,1,0,1],[0,0,1,0,0,1,1],[0,0,0,1,1,1,1]])
def henc(b): return ((b.reshape(-1,4)@G)%2).flatten()

def rrc(beta, sps, span):
    N = span*sps; t = (np.arange(N+1)-N/2)/sps; h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti)<1e-10: h[i] = 1+beta*(4/np.pi-1)
        elif abs(abs(ti)-1/(4*beta))<1e-10:
            h[i] = (beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))
                                    + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            num = np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))
            den = np.pi*ti*(1-(4*beta*ti)**2)
            h[i] = num/den
    return h/np.sqrt(np.sum(h**2))
g = rrc(RRC_BETA, SPS, RRC_SPAN)

preamble_short = np.tile([1,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0], 4).astype(int)
preamble_long  = np.tile([1,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0],16).astype(int)

# ---------- BPSK ----------
def diff_encode_bpsk(b): return np.concatenate([[0], np.cumsum(b)%2]).astype(int)

def costas_bpsk(x, alpha_acq=0.10, n_acq=120, alpha_trk=0.005):
    out = np.zeros_like(x); phi = 0.0; freq = 0.0
    for i in range(len(x)):
        a = alpha_acq if i<n_acq else alpha_trk; b = a*a/4
        out[i] = x[i]*np.exp(-1j*phi)
        e = np.real(out[i])*np.imag(out[i])
        freq += b*e; phi += freq + a*e
    return out

def chain_bpsk(ebn0_db, seed=11):
    rng = np.random.default_rng(seed)
    info = rng.integers(0,2,N_INFO)
    bits = np.concatenate([preamble_short, info])
    bits = np.concatenate([bits, np.zeros((-len(bits))%4, dtype=int)])
    coded = henc(bits)
    cwd = diff_encode_bpsk(coded)
    sym_tx = (2*cwd-1).astype(complex)
    up = np.zeros(len(sym_tx)*SPS, dtype=complex); up[::SPS] = sym_tx
    tx = np.convolve(up, g)
    sigma = np.sqrt(1.0/10**(ebn0_db/10))
    n = np.arange(len(tx))
    nz = (rng.standard_normal(len(tx))+1j*rng.standard_normal(len(tx)))*sigma/np.sqrt(2)
    rx = tx*np.exp(1j*(2*np.pi*FREQ_OFFSET*n + PHASE_OFFSET)) + nz
    mf = np.convolve(rx, g)
    delay = RRC_SPAN*SPS
    samples = mf[delay : delay+len(sym_tx)*SPS][::SPS]
    sq = (mf[delay : delay+len(sym_tx)*SPS])**2
    Nf = 1<<int(np.ceil(np.log2(len(sq)))+1)
    F = np.fft.fftshift(np.fft.fft(sq, Nf))
    fr = np.fft.fftshift(np.fft.fftfreq(Nf))
    fo = fr[np.argmax(np.abs(F))]/2
    samples = samples * np.exp(-1j*2*np.pi*fo*SPS*np.arange(len(samples)))
    sc = costas_bpsk(samples)
    # DBPSK polarity invariant -> evaluate on |.|
    return sym_tx, sc

# ---------- QPSK ----------
QPSK_GRAY = np.array([0,1,3,2])
def b2d(b): p = b.reshape(-1,2); return QPSK_GRAY[p[:,0]*2+p[:,1]]
def dqpsk_encode(d): return np.concatenate([[0], np.cumsum(d)%4]).astype(int)
def qpsk_phase_to_sym(p): return np.exp(1j*(p*np.pi/2 + np.pi/4))

def costas_qpsk(x, alpha_acq=0.10, n_acq=120, alpha_trk=0.005):
    out = np.zeros_like(x); phi = 0.0; freq = 0.0
    for i in range(len(x)):
        a = alpha_acq if i<n_acq else alpha_trk; b = a*a/4
        out[i] = x[i]*np.exp(-1j*phi)
        re = np.real(out[i]); im = np.imag(out[i])
        e = (1.0 if re>=0 else -1.0)*im - (1.0 if im>=0 else -1.0)*re
        freq += b*e; phi += freq + a*e
    return out

def chain_qpsk(ebn0_db, seed=11):
    rng = np.random.default_rng(seed)
    info = rng.integers(0,2,N_INFO)
    bits = np.concatenate([preamble_short, info])
    bits = np.concatenate([bits, np.zeros((-len(bits))%4, dtype=int)])
    coded = henc(bits)
    coded = np.concatenate([coded, np.zeros((-len(coded))%2, dtype=int)])
    pidx = dqpsk_encode(b2d(coded))
    sym_tx = qpsk_phase_to_sym(pidx)
    up = np.zeros(len(sym_tx)*SPS, dtype=complex); up[::SPS] = sym_tx
    tx = np.convolve(up, g)
    sigma = np.sqrt(0.5/10**(ebn0_db/10))
    n = np.arange(len(tx))
    nz = (rng.standard_normal(len(tx))+1j*rng.standard_normal(len(tx)))*sigma/np.sqrt(2)
    rx = tx*np.exp(1j*(2*np.pi*FREQ_OFFSET*n + PHASE_OFFSET)) + nz
    mf = np.convolve(rx, g)
    delay = RRC_SPAN*SPS
    samples = mf[delay : delay+len(sym_tx)*SPS][::SPS]
    sq = (mf[delay : delay+len(sym_tx)*SPS])**4
    Nf = 1<<int(np.ceil(np.log2(len(sq)))+1)
    F = np.fft.fftshift(np.fft.fft(sq, Nf))
    fr = np.fft.fftshift(np.fft.fftfreq(Nf))
    fo = fr[np.argmax(np.abs(F))]/4
    samples = samples * np.exp(-1j*2*np.pi*fo*SPS*np.arange(len(samples)))
    sc = costas_qpsk(samples)
    return sym_tx, sc

# ---------- 256-QAM ----------
NAT16 = np.arange(16); GRAY4 = NAT16^(NAT16>>1); INV_GRAY4 = np.argsort(GRAY4)
QAM_NORM = np.sqrt(170.0)
def b2q(b):
    g8 = b.reshape(-1, 8)
    bI = g8[:,0]*8+g8[:,1]*4+g8[:,2]*2+g8[:,3]
    bQ = g8[:,4]*8+g8[:,5]*4+g8[:,6]*2+g8[:,7]
    return (2*INV_GRAY4[bI]-15 + 1j*(2*INV_GRAY4[bQ]-15))/QAM_NORM
def hard_qam(z):
    zr = z*QAM_NORM
    lI = np.clip(2*np.round((np.real(zr)+15)/2).astype(int)-15,-15,15)
    lQ = np.clip(2*np.round((np.imag(zr)+15)/2).astype(int)-15,-15,15)
    return (lI + 1j*lQ)/QAM_NORM

def costas_qam(z, preamble_sym, alpha_pa=0.30, alpha_trk=0.05, init_n=20):
    n_pa = len(preamble_sym)
    N_init = min(init_n, n_pa)
    phi = float(np.angle(np.sum(z[:N_init]*np.conj(preamble_sym[:N_init]))))
    freq = 0.0; out = np.zeros_like(z)
    for i in range(len(z)):
        rotated = z[i]*np.exp(-1j*phi)
        out[i] = rotated
        if i < n_pa:
            d = preamble_sym[i]; a = alpha_pa
        else:
            d = hard_qam(np.array([rotated]))[0]; a = alpha_trk
        denom = np.abs(d)**2
        e = np.imag(rotated*np.conj(d))/max(denom, 1e-3) if denom>0 else 0.0
        b = a*a/4
        freq += b*e; phi += freq + a*e
    return out

def chain_qam(ebn0_db, seed=11):
    rng = np.random.default_rng(seed)
    info = rng.integers(0,2,N_INFO)
    bits = np.concatenate([preamble_long, info])
    bits = np.concatenate([bits, np.zeros((-len(bits))%4, dtype=int)])
    coded = henc(bits)
    coded = np.concatenate([coded, np.zeros((-len(coded))%8, dtype=int)])
    sym_tx = b2q(coded)
    pre_sym = sym_tx[:len(henc(preamble_long))//8]
    up = np.zeros(len(sym_tx)*SPS, dtype=complex); up[::SPS] = sym_tx
    tx = np.convolve(up, g)
    sigma = np.sqrt((1.0/8.0)/10**(ebn0_db/10))
    n = np.arange(len(tx))
    nz = (rng.standard_normal(len(tx))+1j*rng.standard_normal(len(tx)))*sigma/np.sqrt(2)
    rx = tx*np.exp(1j*(2*np.pi*FREQ_OFFSET*n + PHASE_OFFSET)) + nz
    mf = np.convolve(rx, g)
    delay = RRC_SPAN*SPS
    samples = mf[delay : delay+len(sym_tx)*SPS][::SPS]
    sq = (mf[delay : delay+len(sym_tx)*SPS])**4
    Nf = 1<<int(np.ceil(np.log2(len(sq)))+1)
    F = np.fft.fftshift(np.fft.fft(sq, Nf))
    fr = np.fft.fftshift(np.fft.fftfreq(Nf))
    fo = fr[np.argmax(np.abs(F))]/4
    samples = samples * np.exp(-1j*2*np.pi*fo*SPS*np.arange(len(samples)))
    sc = costas_qam(samples, pre_sym)
    SETTLE = 10
    win = sc[SETTLE:len(pre_sym)]
    ref = pre_sym[SETTLE:len(pre_sym)]
    br, bs = 0, -np.inf
    for k in range(4):
        score = np.real(np.sum(win*np.exp(-1j*k*np.pi/2)*np.conj(ref)))
        if score > bs: bs=score; br=k
    sc = sc * np.exp(-1j*br*np.pi/2)
    # amplitude calibrate
    pre_win = sc[:len(pre_sym)]
    scale = (np.real(np.sum(pre_win*np.conj(pre_sym)))
             / max(np.real(np.sum(np.abs(pre_sym)**2)), 1e-12))
    if abs(scale) > 1e-6: sc = sc/scale
    return sym_tx, sc

# ---------- EVM computation ----------
def compute_evm(sym_rx, sym_tx, mod):
    """Reference-based EVM over the steady-state portion (last 70%).
    DBPSK / DQPSK are invariant to global polarity / 90° rotation, so we
    pick the rotation that minimises EVM against the known TX symbols."""
    n = min(len(sym_rx), len(sym_tx))
    start = int(n * 0.3)
    rx = sym_rx[start:n]
    tx = sym_tx[start:n]
    p_ref = float(np.mean(np.abs(tx)**2))
    # candidate rotations
    if   mod == "bpsk": rots = [1.0, -1.0]
    elif mod == "qpsk": rots = [1.0, 1j, -1.0, -1j]
    else:               rots = [1.0]                       # 256-QAM uses absolute decoding
    best = None
    for r in rots:
        e = rx*r - tx
        p_err = float(np.mean(np.abs(e)**2))
        if best is None or p_err < best[0]:
            best = (p_err, r)
    rx_aligned = rx * best[1]
    evm = float(np.sqrt(best[0] / p_ref))
    return evm, rx_aligned, tx

def evm_theory(ebn0_db, mod):
    """Theoretical EVM = 1/sqrt(Es/N0). Es = log2(M) * Eb."""
    k = {"bpsk":1, "qpsk":2, "qam256":8}[mod]
    es_n0 = k * 10**(ebn0_db/10)
    return 1.0 / np.sqrt(es_n0)

# ---------- per-modem 3-panel plot ----------
def make_evm_plot(mod, ebn0_points, lim, color, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), facecolor="#0b1020")
    fig.suptitle(f"{mod.upper()} — EVM at 3 Eb/N0 operating points",
                 color="w", fontsize=13)
    chain = {"bpsk": chain_bpsk, "qpsk": chain_qpsk, "qam256": chain_qam}[mod]
    for ax, ebn0 in zip(axes, ebn0_points):
        sym_tx, sc = chain(ebn0)
        evm, rx_ss, tx_ss = compute_evm(sc, sym_tx, mod)
        evm_th = evm_theory(ebn0, mod)
        ax.set_facecolor("#0b1020")
        ax.scatter(np.real(rx_ss), np.imag(rx_ss), s=4, alpha=0.4, color=color)
        # overlay TX reference points (small, lighter)
        ax.scatter(np.real(tx_ss), np.imag(tx_ss), s=12, alpha=0.7,
                   color="#ffff80", marker="x", linewidths=0.6)
        ax.axhline(0, color="w", lw=0.4); ax.axvline(0, color="w", lw=0.4)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
        ax.set_title(f"Eb/N0 = {ebn0} dB\nEVM = {evm*100:.2f}%   ({20*np.log10(evm):.1f} dB)\n"
                     f"theory = {evm_th*100:.2f}%",
                     color="w", fontsize=10)
        ax.tick_params(colors="w")
        for sp in ax.spines.values(): sp.set_color("w")
        ax.grid(False)
        print(f"  {mod} @ {ebn0:>4} dB:  EVM = {evm*100:6.3f}%   "
              f"({20*np.log10(evm):+5.1f} dB)   theory = {evm_th*100:6.3f}%")
    fig.tight_layout(rect=[0,0,1,0.94])
    fig.savefig(out_path, dpi=130, facecolor="#0b1020")
    plt.close(fig)
    print(f"  -> {out_path}")

print("BPSK:")
make_evm_plot("bpsk",   [5, 10, 15],  1.6,  "#4ec9b0",
              os.path.join(ROOT, "bpsk_out",   "evm_vs_snr.png"))
print("QPSK:")
make_evm_plot("qpsk",   [5, 10, 15],  1.6,  "#e377c2",
              os.path.join(ROOT, "qpsk_out",   "evm_vs_snr.png"))
print("256-QAM:")
make_evm_plot("qam256", [20, 25, 30], 1.4,  "#9cdcfe",
              os.path.join(ROOT, "qam256_out", "evm_vs_snr.png"))
print("done.")
