"""
End-to-end 256-QAM modem in Python.

Chain (mirrors BPSK / QPSK structure, but with QAM-specific sync):
  Bits -> Hamming(7,4) FEC -> Gray-coded I/Q levels -> 256-QAM map
       -> RRC pulse shaping
       -> AWGN + carrier freq offset + carrier phase offset
       -> Matched filter -> Coarse freq estimation by 4th power
                                       (works because corner symbols dominate)
       -> Decision-directed Costas loop (no Mth-power available for arbitrary QAM)
       -> 4-fold (quadrant) ambiguity resolve via known preamble correlation
       -> Hard decisions on QAM grid -> Hamming decode -> BER

256-QAM at 15 dB is unusable (BER ~ 0.02 raw); we operate at 25 dB.

Outputs in C:\\Users\\Tadeh\\bpsk_modem\\qam256_out\\
"""
import os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import imageio_ffmpeg

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qam256_out")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(7)

# ============================ PARAMETERS ===================================
N_INFO       = 4000
SPS          = 8
RRC_BETA     = 0.35
RRC_SPAN     = 11
EBN0_DB      = 25.0          # 256-QAM needs ~25 dB to look clean
FREQ_OFFSET  = 0.003
PHASE_OFFSET = 1.20

# ============================ HAMMING(7,4) =================================
G = np.array([[1,0,0,0,1,1,0],[0,1,0,0,1,0,1],[0,0,1,0,0,1,1],[0,0,0,1,1,1,1]])
H = np.array([[1,1,0,1,1,0,0],[1,0,1,1,0,1,0],[0,1,1,1,0,0,1]])
SYN = {(0,0,0): -1}
for i in range(7):
    e = np.zeros(7, dtype=int); e[i] = 1
    SYN[tuple((H @ e) % 2)] = i

def hamming_encode(bits):
    return ((bits.reshape(-1,4)@G)%2).flatten()

def hamming_decode(bits):
    cw = bits.reshape(-1,7).copy()
    out = np.zeros((cw.shape[0],4), dtype=int)
    for k in range(cw.shape[0]):
        s = tuple((H @ cw[k]) % 2)
        idx = SYN.get(s, -1)
        if idx >= 0: cw[k, idx] ^= 1
        out[k] = cw[k, :4]
    return out.flatten()

# ============================ RRC FILTER ===================================
def rrc(beta, sps, span):
    N = span * sps
    t = (np.arange(N+1) - N/2) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-10: h[i] = 1 + beta*(4/np.pi - 1)
        elif abs(abs(ti)-1/(4*beta)) < 1e-10:
            h[i] = (beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))
                                    + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            num = np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))
            den = np.pi*ti*(1 - (4*beta*ti)**2)
            h[i] = num/den
    return h / np.sqrt(np.sum(h**2))

# ============================ 256-QAM MAP ==================================
# I and Q each take Gray-coded levels in {-15,-13,...,+13,+15}.
# Mean symbol energy E[|sym|^2] = 2 * (1/16) * sum(k^2 for k in odd[-15..15]) = 170.
# Normalise by sqrt(170) so |sym|_rms = 1.
NAT = np.arange(16)
GRAY4 = NAT ^ (NAT >> 1)            # GRAY4[natural] = gray-code value
INV_GRAY4 = np.argsort(GRAY4)       # INV_GRAY4[gray] = natural   (= level index)
QAM_NORM = np.sqrt(170.0)

def bits_to_qam256(bits):
    g = bits.reshape(-1, 8)
    binI = g[:,0]*8 + g[:,1]*4 + g[:,2]*2 + g[:,3]
    binQ = g[:,4]*8 + g[:,5]*4 + g[:,6]*2 + g[:,7]
    lvlI = 2*INV_GRAY4[binI] - 15
    lvlQ = 2*INV_GRAY4[binQ] - 15
    return (lvlI + 1j*lvlQ) / QAM_NORM

def qam256_to_bits(sym):
    z = sym * QAM_NORM
    lvlI = np.clip(2*np.round((np.real(z)+15)/2).astype(int) - 15, -15, 15)
    lvlQ = np.clip(2*np.round((np.imag(z)+15)/2).astype(int) - 15, -15, 15)
    idxI = ((lvlI + 15)//2).astype(int)
    idxQ = ((lvlQ + 15)//2).astype(int)
    binI = GRAY4[idxI]; binQ = GRAY4[idxQ]
    out = np.zeros((len(sym), 8), dtype=int)
    out[:,0] = (binI>>3)&1; out[:,1] = (binI>>2)&1
    out[:,2] = (binI>>1)&1; out[:,3] =  binI    &1
    out[:,4] = (binQ>>3)&1; out[:,5] = (binQ>>2)&1
    out[:,6] = (binQ>>1)&1; out[:,7] =  binQ    &1
    return out.flatten()

def hard_decide(z):
    """Snap z to nearest 256-QAM grid point (returns complex)."""
    zr = z * QAM_NORM
    lvlI = np.clip(2*np.round((np.real(zr)+15)/2).astype(int) - 15, -15, 15)
    lvlQ = np.clip(2*np.round((np.imag(zr)+15)/2).astype(int) - 15, -15, 15)
    return (lvlI + 1j*lvlQ) / QAM_NORM

# ============================ TX ==========================================
print("[1/9] Generating bits + Hamming(7,4) + 256-QAM Gray map")
preamble_bits = np.tile([1,1,0,0, 1,0,1,1, 0,1,0,0, 1,1,1,0], 16).astype(int)  # 256 bits
info_bits = RNG.integers(0, 2, N_INFO)
all_bits = np.concatenate([preamble_bits, info_bits])
all_bits = np.concatenate([all_bits, np.zeros((-len(all_bits)) % 4, dtype=int)])
coded = hamming_encode(all_bits)
coded = np.concatenate([coded, np.zeros((-len(coded)) % 8, dtype=int)])  # 8-bit groups
sym_tx = bits_to_qam256(coded)
N_SYM  = len(sym_tx)
preamble_sym = sym_tx[: (len(hamming_encode(preamble_bits)) // 8)]   # known sym at start
print(f"      info={len(info_bits)}  preamble={len(preamble_bits)}  coded={len(coded)}  qam_sym={N_SYM}  preamble_sym={len(preamble_sym)}")

print("[2/9] 256-QAM mapping (Gray, |sym|_rms=1)")
print("[3/9] RRC pulse shaping")
g = rrc(RRC_BETA, SPS, RRC_SPAN)
up = np.zeros(N_SYM*SPS, dtype=complex)
up[::SPS] = sym_tx
tx = np.convolve(up, g)

# ============================ CHANNEL =====================================
# Es = E[|sym|^2] = 1.  Eb = Es / log2(256) = 1/8.
# Per-complex-sample noise variance at MF output equals (1/8)/EbN0_lin.
print("[4/9] Channel: AWGN + freq + phase")
EbN0_lin = 10**(EBN0_DB/10)
sigma_ch = np.sqrt((1.0/8.0) / EbN0_lin)
n = np.arange(len(tx))
phase_ramp = np.exp(1j*(2*np.pi*FREQ_OFFSET*n + PHASE_OFFSET))
noise = (RNG.standard_normal(len(tx)) + 1j*RNG.standard_normal(len(tx))) * sigma_ch/np.sqrt(2)
rx = tx*phase_ramp + noise

# ============================ MATCHED FILTER ==============================
print("[5/9] Matched filter")
mf = np.convolve(rx, g)
delay = RRC_SPAN * SPS
mf_aligned = mf[delay : delay + N_SYM*SPS]
samp_pre = mf_aligned[::SPS]

# ============================ COARSE FREQ EST (4TH POWER) =================
# 256-QAM has no clean periodic phase, but corner symbols (4 outermost points
# in each quadrant) raised to ^4 still give a strong tone at 4*f0.  Adequate.
print("[6/9] Coarse frequency estimation by 4th power")
sq = mf_aligned**4
NFFT = 1 << int(np.ceil(np.log2(len(sq)))+1)
F = np.fft.fftshift(np.fft.fft(sq, NFFT))
freqs = np.fft.fftshift(np.fft.fftfreq(NFFT))
peak = freqs[np.argmax(np.abs(F))]
fo_est = peak / 4.0
mf_corr = mf_aligned * np.exp(-1j*2*np.pi*fo_est*np.arange(len(mf_aligned)))
samp_coarse = mf_corr[::SPS]
print(f"      true freq offset = {FREQ_OFFSET:.5f} cyc/sample  est = {fo_est:.5f}")

# ============================ DECISION-DIRECTED COSTAS ====================
# After acquisition, decisions land near-correct so DD error is well-formed:
#     e = Im(z * conj(decision)) / max(|decision|^2, eps)
# For *acquisition* on 256-QAM, DD error is noisy, so we boot-strap with a
# few bursts of "polar" (sign-of-real-and-imag) error which is QPSK-style
# and exploits the constellation's 4-fold outer symmetry.
print("[7/9] Costas loop — data-aided acquisition (preamble) -> DD tracking")
def costas_qam(z, preamble_sym, alpha_pa=0.30, alpha_trk=0.05, init_n=20):
    """For QAM, the QPSK-style error has large bias on inner symbols, so we
    boot-strap with **data-aided** acquisition using the known preamble
    (clean error: e = Im(z*conj(known_sym))/|known_sym|^2), then switch to
    decision-directed once preamble is exhausted.  Loop is initialised from
    the preamble cross-correlation phase so DA pull-in starts at residual
    near zero."""
    out = np.zeros_like(z)
    n_pa = len(preamble_sym)
    N_init = min(init_n, n_pa)
    phi = float(np.angle(np.sum(z[:N_init]*np.conj(preamble_sym[:N_init]))))
    freq = 0.0
    phi_log = np.zeros(len(z))
    for i in range(len(z)):
        rotated = z[i] * np.exp(-1j*phi)
        out[i] = rotated
        if i < n_pa:
            d = preamble_sym[i]
            alpha = alpha_pa
        else:
            d = hard_decide(np.array([rotated]))[0]
            alpha = alpha_trk
        denom = np.abs(d)**2
        e = np.imag(rotated*np.conj(d)) / max(denom, 1e-3) if denom>0 else 0.0
        beta = (alpha**2)/4
        freq += beta * e
        phi  += freq + alpha * e
        phi_log[i] = phi
    return out, phi_log

samp_costas, phi_log = costas_qam(samp_coarse, preamble_sym)

# ============================ AMBIGUITY RESOLVE ===========================
# 256-QAM has 4-fold symmetry; Costas can lock at any of 4 rotations of pi/2.
# Resolve by correlating known preamble symbols against the 4 candidate
# rotations and picking the one with maximum |correlation|.
print("[8/9] 4-fold ambiguity resolve via preamble correlation")
n_pre = len(preamble_sym)
SETTLE = 10                                          # Costas with QPSK-boot acquires fast
n_corr = n_pre - SETTLE
window = samp_costas[SETTLE : SETTLE + n_corr]
ref    = preamble_sym[SETTLE : SETTLE + n_corr]
best_rot, best_score = 0, -np.inf
for k in range(4):
    rot = np.exp(-1j*k*np.pi/2)
    sc = np.real(np.sum(window*rot * np.conj(ref)))
    if sc > best_score:
        best_score = sc; best_rot = k
samp_final = samp_costas * np.exp(-1j*best_rot*np.pi/2)
print(f"      best rotation = {best_rot}*pi/2   score = {best_score:+.2f}")

# ============================ DECISIONS ===================================
hard_sym = hard_decide(samp_final[:N_SYM])
ser_pre  = np.mean(hard_sym != sym_tx[:N_SYM])
bits_rx  = qam256_to_bits(hard_sym)
ber_pre  = np.mean(bits_rx[:len(coded)] != coded)
n7       = (len(bits_rx)//7)*7
decoded  = hamming_decode(bits_rx[:n7])
ber_post = np.mean(decoded[:len(all_bits)] != all_bits)
ber_info = np.mean(decoded[len(preamble_bits):len(preamble_bits)+N_INFO]
                  != info_bits)
print(f"      SER pre-FEC : {ser_pre:.4e}")
print(f"      BER pre-FEC : {ber_pre:.4e}")
print(f"      BER post-FEC: {ber_post:.4e}   (info-only: {ber_info:.4e})")

# ============================ REPORT ======================================
with open(os.path.join(OUT, "report.txt"), "w") as f:
    f.write(f"""256-QAM modem report
====================
samples/symbol      : {SPS}
RRC roll-off / span : {RRC_BETA} / {RRC_SPAN}
Eb/N0 (dB)          : {EBN0_DB}
True freq offset    : {FREQ_OFFSET:.6f} cycles/sample
Estimated freq off  : {fo_est:.6f} cycles/sample
True phase offset   : {PHASE_OFFSET:.4f} rad
Info bits           : {N_INFO}
QAM symbols         : {N_SYM}

SER pre-FEC         : {ser_pre:.4e}
BER pre-FEC         : {ber_pre:.4e}
BER post-FEC (all)  : {ber_post:.4e}
BER post-FEC (info) : {ber_info:.4e}
Quadrant rotation   : {best_rot} * pi/2

Carrier recovery uses 4th-power coarse acquisition followed by decision-
directed Costas tracking.  4-fold (90 degrees) phase ambiguity is resolved
by a known preamble of {len(preamble_sym)} 256-QAM symbols.
""")

# ============================ FIGURES =====================================
print("[9/9] Plotting and rendering video")

def db_fft(x, N=None):
    if N is None: N = 1 << int(np.ceil(np.log2(len(x)))+1)
    X = np.fft.fftshift(np.fft.fft(x, N))
    f = np.fft.fftshift(np.fft.fftfreq(N))
    mag = 20*np.log10(np.abs(X)/max(np.abs(X).max(), 1e-12) + 1e-12)
    return f, mag

stages = [
    ("1. 256-QAM symbols (impulse train)",          up,                "samples"),
    ("2. after RRC pulse shaping (TX out)",         tx,                "samples"),
    ("3. after channel (AWGN + freq + phase)",      rx,                "samples"),
    ("4. after matched filter",                     mf_aligned,        "samples"),
    ("5. after coarse freq correction",             mf_corr,           "samples"),
    ("6. symbol-rate samples before Costas",        samp_coarse,       "symbols"),
    ("7. after Costas + ambiguity resolve",         samp_final,        "symbols"),
]

# spectra
fig, axes = plt.subplots(len(stages), 1, figsize=(11, 2.2*len(stages)))
for ax, (name, sig, _u) in zip(axes, stages):
    f, m = db_fft(sig)
    ax.plot(f, m, lw=0.8)
    ax.set_title(name, fontsize=10, loc="left")
    ax.set_ylim(-80, 5); ax.set_xlim(-0.5, 0.5)
    ax.set_ylabel("dB"); ax.grid(alpha=0.3)
axes[-1].set_xlabel("normalised freq")
fig.suptitle(f"256-QAM modem — FFT magnitude after each block (Eb/N0 = {EBN0_DB} dB)", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.985])
fig.savefig(os.path.join(OUT, "spectra.png"), dpi=130)
plt.close(fig)

# waveforms
fig, axes = plt.subplots(len(stages), 1, figsize=(11, 1.8*len(stages)))
for ax, (name, sig, unit) in zip(axes, stages):
    n_show = min(len(sig), 800)
    ax.plot(np.real(sig[:n_show]), lw=0.8, label="Re")
    ax.plot(np.imag(sig[:n_show]), lw=0.8, label="Im", alpha=0.7)
    ax.set_title(name, fontsize=10, loc="left")
    ax.set_xlabel(unit); ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
fig.suptitle("256-QAM modem — time-domain after each block (first 800 samples)", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.985])
fig.savefig(os.path.join(OUT, "waveforms.png"), dpi=130)
plt.close(fig)

# constellations (with grid markers for the 16x16 lattice)
SS_FROM = max(0, N_SYM*7//10)
const_stages = [
    ("symbol-rate samples — no sync",          samp_pre),
    ("after coarse freq correction",           samp_coarse),
    ("after Costas (entire run)",               samp_costas),
    ("after Costas — steady state (last 30%)", samp_final[SS_FROM:]),
]
fig, axes = plt.subplots(1, len(const_stages), figsize=(4.4*len(const_stages), 4.6))
grid_levels = (np.arange(-15,16,2))/QAM_NORM
for ax, (name, s) in zip(axes, const_stages):
    ax.scatter(np.real(s), np.imag(s), s=2, alpha=0.4)
    for lv in grid_levels:
        ax.axhline(lv, color="0.8", lw=0.3); ax.axvline(lv, color="0.8", lw=0.3)
    ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
    ax.set_aspect("equal"); ax.grid(False)
    ax.set_xlim(-1.4,1.4); ax.set_ylim(-1.4,1.4)
    ax.set_title(name, fontsize=10)
fig.suptitle("256-QAM modem — constellation through synchronisation", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(OUT, "constellations.png"), dpi=130)
plt.close(fig)

# eye diagram (post-sync)
n_eye_sym = 200
start_sym = 600
phi_per_sample = np.repeat(phi_log, SPS)
if len(phi_per_sample) < len(mf_corr):
    phi_per_sample = np.concatenate(
        [phi_per_sample, np.full(len(mf_corr)-len(phi_per_sample), phi_per_sample[-1])])
mf_synced = mf_corr * np.exp(-1j*phi_per_sample[:len(mf_corr)]) * np.exp(-1j*best_rot*np.pi/2)
seg_re = mf_synced[start_sym*SPS:(start_sym+n_eye_sym)*SPS].real
seg_im = mf_synced[start_sym*SPS:(start_sym+n_eye_sym)*SPS].imag
W = 2*SPS
seg_re = seg_re[:(len(seg_re)//W)*W].reshape(-1, W)
seg_im = seg_im[:(len(seg_im)//W)*W].reshape(-1, W)
t_eye = (np.arange(W) - SPS/2) / SPS
fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
for ax, seg, lab in zip(axes, [seg_re, seg_im], ["I (in-phase)", "Q (quadrature)"]):
    for row in seg:
        ax.plot(t_eye, row, color="#1f77b4", lw=0.4, alpha=0.18)
    for lv in grid_levels:
        ax.axhline(lv, color="0.85", lw=0.3)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0, color="r", lw=0.8, ls="--")
    ax.axvline(1, color="r", lw=0.8, ls="--", label="decision instant")
    ax.set_ylabel(f"{lab}  amplitude"); ax.grid(False)
axes[0].set_title(f"256-QAM eye diagram — post-sync MF output  ({n_eye_sym} sym, Eb/N0={EBN0_DB} dB)")
axes[1].set_xlabel("symbol periods")
axes[0].legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "eye_diagram.png"), dpi=130)
plt.close(fig)

# BER vs Eb/N0 sweep
print("      BER-vs-Eb/N0 sweep")
from math import erfc
def Q(x): return 0.5*erfc(x/np.sqrt(2))
def qam_theory_ber(ebn0_db, M=256):
    k = int(np.log2(M)); ebn0 = 10**(ebn0_db/10)
    return (4.0/k)*(1 - 1/np.sqrt(M))*Q(np.sqrt(3*k/(M-1)*ebn0))

def run_chain(ebn0_db, n_info=20000, seed=11):
    rng = np.random.default_rng(seed)
    pre = preamble_bits
    inf = rng.integers(0,2,n_info)
    bits = np.concatenate([pre, inf])
    bits = np.concatenate([bits, np.zeros((-len(bits))%4, dtype=int)])
    cw = hamming_encode(bits)
    cw = np.concatenate([cw, np.zeros((-len(cw))%8, dtype=int)])
    s  = bits_to_qam256(cw)
    pre_sym_local = s[: (len(hamming_encode(pre))//8)]
    upx = np.zeros(len(s)*SPS, dtype=complex); upx[::SPS] = s
    txx = np.convolve(upx, g)
    sig = np.sqrt((1.0/8.0)/10**(ebn0_db/10))
    nn = np.arange(len(txx))
    nz = (rng.standard_normal(len(txx)) + 1j*rng.standard_normal(len(txx))) * sig/np.sqrt(2)
    rxx = txx*np.exp(1j*(2*np.pi*FREQ_OFFSET*nn + PHASE_OFFSET)) + nz
    mfx = np.convolve(rxx, g)
    mfx = mfx[delay : delay + len(s)*SPS]
    sqx = mfx**4
    Nf = 1 << int(np.ceil(np.log2(len(sqx)))+1)
    Fx = np.fft.fftshift(np.fft.fft(sqx, Nf))
    fxs = np.fft.fftshift(np.fft.fftfreq(Nf))
    fox = fxs[np.argmax(np.abs(Fx))]/4
    mfx = mfx*np.exp(-1j*2*np.pi*fox*np.arange(len(mfx)))
    sx  = mfx[::SPS]
    cox, _ = costas_qam(sx, pre_sym_local)
    n_pre_local = len(pre_sym_local)
    n_corr_local = n_pre_local - SETTLE
    win = cox[SETTLE : SETTLE + n_corr_local]
    refp = pre_sym_local[SETTLE : SETTLE + n_corr_local]
    br, bs = 0, -np.inf
    for kk in range(4):
        rot = np.exp(-1j*kk*np.pi/2)
        ssc = np.real(np.sum(win*rot*np.conj(refp)))
        if ssc > bs: bs=ssc; br=kk
    fin = cox * np.exp(-1j*br*np.pi/2)
    hd  = hard_decide(fin[:len(s)])
    bits_out = qam256_to_bits(hd)
    ber_pre = np.mean(bits_out[:len(cw)] != cw)
    n7 = (len(bits_out)//7)*7
    dec_bits = hamming_decode(bits_out[:n7])
    ber_post = np.mean(dec_bits[:len(bits)] != bits)
    return ber_pre, ber_post

snr_db = np.arange(15, 31, 1.0)
ber_pre_arr = []; ber_post_arr = []
ber_th = [qam_theory_ber(s) for s in snr_db]
for snr in snr_db:
    bp, bo = run_chain(snr)
    ber_pre_arr.append(bp); ber_post_arr.append(bo)
    print(f"      Eb/N0={snr:4.1f} dB   pre-FEC={bp:.3e}  post-FEC={bo:.3e}  th={qam_theory_ber(snr):.3e}")

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.semilogy(snr_db, ber_th, "k-", lw=2,
            label="256-QAM Gray theory (AWGN)")
ax.semilogy(snr_db, np.maximum(ber_pre_arr, 1e-7), "o-", color="#2ca02c",
            label="simulated pre-FEC", markersize=7)
ax.semilogy(snr_db, np.maximum(ber_post_arr, 1e-7), "s-", color="#bcbd22",
            label="simulated post-FEC (Hamming 7,4)", markersize=7)
ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("BER")
ax.set_title("256-QAM modem — BER vs Eb/N0  (full chain, freq+phase offset, full sync)")
ax.set_ylim(1e-7, 1.0); ax.set_xlim(snr_db[0], snr_db[-1])
ax.grid(alpha=0.4, which="both")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ber_vs_ebn0.png"), dpi=130)
plt.close(fig)

# ============================ VIDEO =======================================
DUR_S = 10.0
FPS   = 30
N_FR  = int(round(DUR_S*FPS))
N_TAIL = 2000

def render_frame(f):
    k = max(2, int(round((f+1)/N_FR * N_SYM)))
    pre  = samp_pre[:k]
    coa  = samp_coarse[:k]
    cos_ = samp_costas[:k]
    tail = samp_final[max(0, k-N_TAIL):k]
    age      = np.linspace(0.05, 1.0, len(pre))
    age_tail = np.linspace(0.20, 1.0, len(tail))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#0b1020")
    gs = fig.add_gridspec(2, 4, height_ratios=[3,1])

    titles = ["1. matched-filter samples\n(no sync)",
              "2. + coarse freq correction",
              "3. + Costas (entire run)",
              f"4. + ambiguity-resolved\nsteady-state ({N_TAIL} sym)"]
    panels = [(pre, age, 1.6), (coa, age, 1.6), (cos_, age, 1.6), (tail, age_tail, 1.4)]
    for i,(t,(p,a,L)) in enumerate(zip(titles, panels)):
        ax = fig.add_subplot(gs[0,i])
        ax.scatter(np.real(p), np.imag(p), c=a, cmap="cool",
                   s=4, alpha=0.6, vmin=0, vmax=1)
        # 16x16 grid lines on panel 4
        if i == 3:
            for lv in (np.arange(-15,16,2))/QAM_NORM:
                ax.axhline(lv, color="0.3", lw=0.25)
                ax.axvline(lv, color="0.3", lw=0.25)
        ax.axhline(0, color="w", lw=0.4); ax.axvline(0, color="w", lw=0.4)
        ax.set_facecolor("#0b1020")
        ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_aspect("equal")
        ax.set_title(t, fontsize=10, color="w")
        ax.tick_params(colors="w")
        for sp in ax.spines.values(): sp.set_color("w")

    axb = fig.add_subplot(gs[1,:])
    axb.set_facecolor("#0b1020")
    if k >= 2:
        # cumulative SER
        hd_now = hard_decide(samp_final[:k])
        target = sym_tx[:k]
        err = np.cumsum(hd_now != target) / np.maximum(1, np.arange(1,k+1))
        err_plot = np.maximum(err, 5e-5)
        axb.plot(np.arange(1,k+1), err_plot, color="#bcbd22", lw=1.4,
                 label="cumulative pre-FEC SER")
    axb.set_yscale("log")
    axb.set_ylim(5e-5, 1.0); axb.set_xlim(0, N_SYM)
    axb.set_xlabel("symbols received", color="w")
    axb.set_ylabel("SER (log)", color="w")
    axb.grid(alpha=0.3, which="both", color="#445")
    axb.tick_params(colors="w")
    for sp in axb.spines.values(): sp.set_color("w")
    axb.set_title(f"Receiver convergence — symbol {k}/{N_SYM}    "
                  f"Eb/N0={EBN0_DB} dB    fo_true={FREQ_OFFSET:.4f}, fo_est={fo_est:.4f}",
                  fontsize=11, color="w")
    leg = axb.legend(loc="upper right", facecolor="#0b1020", edgecolor="w")
    for txt in leg.get_texts(): txt.set_color("w")

    fig.suptitle("256-QAM modem — constellation convergence (live)",
                 fontsize=13, y=0.995, color="w")
    fig.tight_layout(rect=[0,0,1,0.97])

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    frame = buf[..., :3].copy()
    plt.close(fig)
    return frame

print(f"      rendering {N_FR} frames at {FPS} fps...")
mp4 = os.path.join(OUT, "constellation.mp4")
writer = imageio.get_writer(mp4, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
t0 = time.time()
for f in range(N_FR):
    fr = render_frame(f)
    writer.append_data(fr)
    if f % 30 == 0:
        print(f"        frame {f+1}/{N_FR}  ({time.time()-t0:.1f}s)")
writer.close()
print(f"      wrote {mp4}  ({os.path.getsize(mp4)/1e6:.2f} MB)")
print("Done.")
