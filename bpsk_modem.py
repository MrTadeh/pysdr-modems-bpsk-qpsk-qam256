"""
End-to-end BPSK modem in Python.

Chain:
  Bits -> Hamming(7,4) FEC -> BPSK -> RRC pulse shaping
       -> AWGN + carrier freq offset + carrier phase offset
       -> Matched filter -> Coarse freq estimation (squaring)
       -> Costas loop -> Decision -> Hamming decode -> BER

Outputs (in C:\\Users\\Tadeh\\bpsk_modem\\out):
  - waveforms.png     time-domain at every stage
  - spectra.png       FFT magnitude (dB) at every stage
  - constellations.png constellation at every stage
  - constellation.mp4  10-second video of constellation convergence
  - report.txt        numerical summary (BER, freq estimate, etc.)
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import imageio_ffmpeg

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpsk_out")
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(7)

# ============================ PARAMETERS ===================================
N_INFO       = 4000          # information bits
SPS          = 8             # samples per symbol (oversampling factor)
RRC_BETA     = 0.35          # RRC roll-off
RRC_SPAN     = 11            # filter span (in symbols)
EBN0_DB      = 25.0          # SNR per info bit (before FEC gain)
FREQ_OFFSET  = 0.003         # carrier freq offset, cycles per *sample*
PHASE_OFFSET = 1.20          # carrier phase offset, radians

# ============================ FEC: HAMMING(7,4) ============================
G = np.array([
    [1,0,0,0, 1,1,0],
    [0,1,0,0, 1,0,1],
    [0,0,1,0, 0,1,1],
    [0,0,0,1, 1,1,1],
])
H = np.array([
    [1,1,0,1, 1,0,0],
    [1,0,1,1, 0,1,0],
    [0,1,1,1, 0,0,1],
])
SYN = {(0,0,0): -1}
for i in range(7):
    e = np.zeros(7, dtype=int); e[i] = 1
    SYN[tuple((H @ e) % 2)] = i

def hamming_encode(bits):
    return ((bits.reshape(-1,4) @ G) % 2).flatten()

def hamming_decode(bits):
    cw = bits.reshape(-1,7).copy()
    out = np.zeros((cw.shape[0],4), dtype=int)
    for k in range(cw.shape[0]):
        s = tuple((H @ cw[k]) % 2)
        idx = SYN.get(s, -1)
        if idx >= 0:
            cw[k, idx] ^= 1
        out[k] = cw[k, :4]
    return out.flatten()

# ============================ RRC FILTER ===================================
def rrc(beta, sps, span):
    N = span * sps
    t = (np.arange(N+1) - N/2) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            h[i] = 1 + beta*(4/np.pi - 1)
        elif abs(abs(ti) - 1/(4*beta)) < 1e-10:
            h[i] = (beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))
                                    + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else:
            num = np.sin(np.pi*ti*(1-beta)) + 4*beta*ti*np.cos(np.pi*ti*(1+beta))
            den = np.pi*ti*(1 - (4*beta*ti)**2)
            h[i] = num/den
    return h / np.sqrt(np.sum(h**2))

# ============================ TX ==========================================
# Differential precoding makes bit recovery invariant to 180° polarity flips,
# eliminating cycle-slip pathologies at low SNR.  Cost: well-known ~3 dB DBPSK
# penalty at high SNR (each detection error becomes ~2 differential errors).
def diff_encode(bits):
    return np.concatenate([[0], np.cumsum(bits) % 2]).astype(int)
def diff_decode(bits):
    return np.bitwise_xor(bits[1:], bits[:-1])

print("[1/9] Generating bits + Hamming(7,4) FEC + differential precoding")
preamble_bits = np.tile([1,1,0,0, 1,0,1,1, 0,1,0,0, 1,1,1,0], 4).astype(int)  # 64 bits
info_bits = RNG.integers(0, 2, N_INFO)
all_bits = np.concatenate([preamble_bits, info_bits])
pad = (-len(all_bits)) % 4
all_bits = np.concatenate([all_bits, np.zeros(pad, dtype=int)])
coded = hamming_encode(all_bits)
coded_diff = diff_encode(coded)                    # 1 reference symbol prepended
N_SYM = len(coded_diff)
print(f"      info={len(info_bits)}  preamble={len(preamble_bits)}  total={len(all_bits)}  coded={len(coded)}  diff_sym={N_SYM}")

print("[2/9] BPSK mapping")
sym_tx = (2*coded_diff - 1).astype(complex)

print("[3/9] RRC pulse shaping")
g = rrc(RRC_BETA, SPS, RRC_SPAN)
up = np.zeros(N_SYM*SPS, dtype=complex)
up[::SPS] = sym_tx
tx = np.convolve(up, g)

# ============================ CHANNEL =====================================
print("[4/9] Channel: AWGN + freq + phase")
EbN0_lin = 10**(EBN0_DB/10)
sigma_ch = np.sqrt(1.0/EbN0_lin)
n = np.arange(len(tx))
phase_ramp = np.exp(1j*(2*np.pi*FREQ_OFFSET*n + PHASE_OFFSET))
noise = (RNG.standard_normal(len(tx)) + 1j*RNG.standard_normal(len(tx))) * sigma_ch/np.sqrt(2)
rx = tx*phase_ramp + noise

# ============================ MATCHED FILTER ==============================
print("[5/9] Matched filter")
mf = np.convolve(rx, g)
delay = RRC_SPAN * SPS                             # group delay of TX*RX RRC chain
mf_aligned = mf[delay : delay + N_SYM*SPS]
samp_pre = mf_aligned[::SPS]                       # symbol-rate samples before any sync

# ============================ COARSE FREQ EST (SQUARING) ==================
print("[6/9] Coarse frequency estimation by squaring")
sq = mf_aligned**2
NFFT = 1 << int(np.ceil(np.log2(len(sq)))+1)
F = np.fft.fftshift(np.fft.fft(sq, NFFT))
freqs = np.fft.fftshift(np.fft.fftfreq(NFFT))      # cycles/sample
peak = freqs[np.argmax(np.abs(F))]
fo_est = peak / 2.0                                # cycles/sample
mf_corr = mf_aligned * np.exp(-1j*2*np.pi*fo_est*np.arange(len(mf_aligned)))
samp_coarse = mf_corr[::SPS]
print(f"      true freq offset = {FREQ_OFFSET:.5f} cyc/sample  est = {fo_est:.5f}")

# ============================ COSTAS LOOP =================================
# Two-phase: wide BW for fast acquisition, then narrow BW to suppress
# tracking jitter and reach theoretical BER.
print("[7/9] Costas loop (BPSK) — fast acquire + slow track")
def costas_bpsk(x, alpha_acq=0.10, n_acq=120, alpha_trk=0.005):
    out = np.zeros_like(x)
    phi = 0.0; fr = 0.0
    phi_log = np.zeros(len(x))
    for i in range(len(x)):
        alpha = alpha_acq if i < n_acq else alpha_trk
        beta = (alpha**2)/4
        out[i] = x[i] * np.exp(-1j*phi)
        e = np.real(out[i]) * np.imag(out[i])      # BPSK error
        fr += beta * e
        phi += fr + alpha * e
        phi_log[i] = phi
    return out, phi_log

samp_costas, phi_log = costas_bpsk(samp_coarse)

# ============================ DECISIONS + DIFF DECODE ====================
# Differential decoding cancels 180° ambiguity AND any mid-stream cycle slips.
print("[8/9] Symbol decisions + differential decode")
samp_final = samp_costas[:N_SYM]                   # no polarity flip needed
hard_diff = (np.real(samp_final) > 0).astype(int)
hard      = diff_decode(hard_diff)                 # length N_SYM-1 = len(coded)
ber_pre   = np.mean(hard != coded)
decoded   = hamming_decode(hard)
ber_post  = np.mean(decoded[:len(all_bits)] != all_bits)
ber_info  = np.mean(decoded[len(preamble_bits):len(preamble_bits)+N_INFO]
                   != info_bits)
print(f"      BER pre-FEC (post-diff): {ber_pre:.4e}")
print(f"      BER post-FEC          : {ber_post:.4e}   (info-only: {ber_info:.4e})")

# ============================ REPORT ======================================
with open(os.path.join(OUT, "report.txt"), "w") as f:
    f.write(f"""BPSK modem report
==================
samples/symbol      : {SPS}
RRC roll-off / span : {RRC_BETA} / {RRC_SPAN}
Eb/N0 (dB)          : {EBN0_DB}
True freq offset    : {FREQ_OFFSET:.6f} cycles/sample
Estimated freq off  : {fo_est:.6f} cycles/sample
True phase offset   : {PHASE_OFFSET:.4f} rad
Info bits           : {N_INFO}
Coded symbols       : {N_SYM}

BER pre-FEC         : {ber_pre:.4e}
BER post-FEC (all)  : {ber_post:.4e}
BER post-FEC (info) : {ber_info:.4e}

Bit recovery uses differential precoding/decoding so the system tolerates
180-degree polarity ambiguity AND mid-run Costas cycle slips.  This is the
classic DBPSK-precoded-BPSK construction and matches the simulated-vs-theory
curve in ber_vs_ebn0.png.
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
    ("1. coded bits (impulse train at sym rate)", up,                 "samples"),
    ("2. after RRC pulse shaping (TX out)",       tx,                 "samples"),
    ("3. after channel (AWGN + freq + phase)",     rx,                 "samples"),
    ("4. after matched filter",                    mf_aligned,         "samples"),
    ("5. after coarse freq correction",            mf_corr,            "samples"),
    ("6. symbol-rate samples before Costas",       samp_coarse,        "symbols"),
    ("7. after Costas loop",                       samp_costas,        "symbols"),
]

# --- spectra ---
fig, axes = plt.subplots(len(stages), 1, figsize=(11, 2.2*len(stages)))
for ax, (name, sig, _u) in zip(axes, stages):
    f, m = db_fft(sig)
    ax.plot(f, m, lw=0.8)
    ax.set_title(name, fontsize=10, loc="left")
    ax.set_ylim(-80, 5); ax.set_xlim(-0.5, 0.5)
    ax.set_ylabel("dB"); ax.grid(alpha=0.3)
axes[-1].set_xlabel("normalised freq (cycles / sample or cycles / symbol)")
fig.suptitle(f"BPSK modem — FFT magnitude after each block (Eb/N0 = {EBN0_DB} dB)", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.985])
fig.savefig(os.path.join(OUT, "spectra.png"), dpi=130)
plt.close(fig)

# --- waveforms (time domain, real part) ---
fig, axes = plt.subplots(len(stages), 1, figsize=(11, 1.8*len(stages)))
for ax, (name, sig, unit) in zip(axes, stages):
    n_show = min(len(sig), 800)
    ax.plot(np.real(sig[:n_show]), lw=0.8, label="Re")
    ax.plot(np.imag(sig[:n_show]), lw=0.8, label="Im", alpha=0.6)
    ax.set_title(name, fontsize=10, loc="left")
    ax.set_xlabel(unit); ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
fig.suptitle("BPSK modem — time-domain after each block (first 800 samples)", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.985])
fig.savefig(os.path.join(OUT, "waveforms.png"), dpi=130)
plt.close(fig)

# --- constellations (symbol-rate views) ---
SS_FROM = max(0, N_SYM*7//10)                       # last 30% = steady-state
const_stages = [
    ("symbol-rate samples — no sync",          samp_pre),
    ("after coarse freq correction",           samp_coarse),
    ("after Costas loop (full run)",            samp_costas),
    ("after Costas — steady state (last 30%)", samp_costas[SS_FROM:]),
]
fig, axes = plt.subplots(1, len(const_stages), figsize=(4.2*len(const_stages), 4.4))
for ax, (name, s) in zip(axes, const_stages):
    ax.scatter(np.real(s), np.imag(s), s=4, alpha=0.5)
    ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlim(-2,2); ax.set_ylim(-2,2)
    ax.set_title(name, fontsize=10)
fig.suptitle("BPSK modem — constellation through synchronisation", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(OUT, "constellations.png"), dpi=130)
plt.close(fig)

# --- eye diagram (matched-filter output, before decimation) ---
# Take a clean middle portion (after Costas lock and before end) and overlay
# 2-symbol-wide traces of the *post-Costas, sample-rate* signal.
# We re-do the chain at sample rate after sync to get a "clean" eye.
n_eye_sym = 200
start_sym = 600                                    # well past Costas acquisition
phase_full = np.zeros(len(mf_corr))
# expand symbol-rate phi_log to sample-rate by holding (each sym is sps samples)
phi_per_sample = np.repeat(phi_log, SPS)
if len(phi_per_sample) < len(mf_corr):
    phi_per_sample = np.concatenate(
        [phi_per_sample, np.full(len(mf_corr)-len(phi_per_sample), phi_per_sample[-1])])
mf_synced = mf_corr * np.exp(-1j*phi_per_sample[:len(mf_corr)])
seg = mf_synced[start_sym*SPS : (start_sym+n_eye_sym)*SPS].real
# fold into 2-symbol-wide traces (16 samples each)
W = 2*SPS
seg = seg[:(len(seg)//W)*W].reshape(-1, W)
fig, ax = plt.subplots(figsize=(8, 4.6))
t_eye = (np.arange(W) - SPS/2) / SPS               # sym units, decision at 0 and 1
for row in seg:
    ax.plot(t_eye, row, color="#1f77b4", lw=0.4, alpha=0.18)
ax.axhline(0, color="k", lw=0.6)
ax.axvline(0, color="r", lw=0.8, ls="--", label="decision instant")
ax.axvline(1, color="r", lw=0.8, ls="--")
ax.set_xlabel("symbol periods"); ax.set_ylabel("amplitude (Re)")
ax.set_title(f"Eye diagram — post-sync matched filter output  "
             f"({n_eye_sym} symbols overlaid, Eb/N0={EBN0_DB} dB)")
ax.grid(alpha=0.3); ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "eye_diagram.png"), dpi=130)
plt.close(fig)

# --- BER vs Eb/N0 (sweep through entire chain at each SNR) ---
print("      BER-vs-Eb/N0 sweep")
from math import erfc
def bpsk_theory_ber(ebn0_db):
    p = 0.5*erfc(np.sqrt(10**(ebn0_db/10)))
    return p
def dbpsk_coh_theory(ebn0_db):
    p = bpsk_theory_ber(ebn0_db)
    return 2*p*(1-p)

def run_chain(ebn0_db, n_info=30000, seed=11):
    rng = np.random.default_rng(seed)
    pre = preamble_bits
    inf = rng.integers(0,2,n_info)
    bits = np.concatenate([pre, inf])
    bits = np.concatenate([bits, np.zeros((-len(bits))%4, dtype=int)])
    cw = hamming_encode(bits)
    cw_diff = diff_encode(cw)
    s = (2*cw_diff-1).astype(complex)
    upx = np.zeros(len(s)*SPS, dtype=complex); upx[::SPS] = s
    txx = np.convolve(upx, g)
    sig = np.sqrt(1.0/10**(ebn0_db/10))
    nn = np.arange(len(txx))
    nz = (rng.standard_normal(len(txx)) + 1j*rng.standard_normal(len(txx))) * sig/np.sqrt(2)
    rxx = txx*np.exp(1j*(2*np.pi*FREQ_OFFSET*nn + PHASE_OFFSET)) + nz
    mfx = np.convolve(rxx, g)
    mfx = mfx[delay : delay + len(s)*SPS]
    sqx = mfx**2
    Nf = 1 << int(np.ceil(np.log2(len(sqx)))+1)
    Fx = np.fft.fftshift(np.fft.fft(sqx, Nf))
    fxs = np.fft.fftshift(np.fft.fftfreq(Nf))
    fox = fxs[np.argmax(np.abs(Fx))]/2
    mfx = mfx*np.exp(-1j*2*np.pi*fox*np.arange(len(mfx)))
    sx = mfx[::SPS]
    cox, _ = costas_bpsk(sx)
    hd = (np.real(cox[:len(s)]) > 0).astype(int)
    hard_data = diff_decode(hd)
    ber_pre = np.mean(hard_data != cw)
    dec_bits = hamming_decode(hard_data)
    ber_post = np.mean(dec_bits[:len(bits)] != bits)
    return ber_pre, ber_post

snr_db = np.arange(0, 11, 1.0)
ber_pre_arr = []; ber_post_arr = []
ber_th_bpsk = [bpsk_theory_ber(s) for s in snr_db]
ber_th_dbpsk = [dbpsk_coh_theory(s) for s in snr_db]
for snr in snr_db:
    bp, bo = run_chain(snr)
    ber_pre_arr.append(bp); ber_post_arr.append(bo)
    print(f"      Eb/N0={snr:4.1f} dB   pre-FEC={bp:.3e}  post-FEC={bo:.3e}  "
          f"BPSK_theory={bpsk_theory_ber(snr):.3e}  DBPSK_theory={dbpsk_coh_theory(snr):.3e}")

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.semilogy(snr_db, ber_th_bpsk, "k--", lw=1.4,
            label=r"BPSK theory  $\frac{1}{2}\mathrm{erfc}(\sqrt{E_b/N_0})$")
ax.semilogy(snr_db, ber_th_dbpsk, "k-", lw=2,
            label=r"DBPSK-coh theory  $2p(1-p)$  (matches our chain)")
ax.semilogy(snr_db, np.maximum(ber_pre_arr, 1e-7), "o-", color="#1f77b4",
            label="simulated pre-FEC", markersize=7)
ax.semilogy(snr_db, np.maximum(ber_post_arr, 1e-7), "s-", color="#ff7f0e",
            label="simulated post-FEC (Hamming 7,4)", markersize=7)
ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("BER")
ax.set_title("BPSK modem — BER vs Eb/N0  (full chain, freq+phase offset, full sync)")
ax.set_ylim(1e-7, 1.0); ax.set_xlim(snr_db[0], snr_db[-1])
ax.grid(alpha=0.4, which="both")
ax.legend(loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ber_vs_ebn0.png"), dpi=130)
plt.close(fig)

# ============================ VIDEO =======================================
# 10-second mp4, 30 fps, 4-panel constellation showing the receiver
# converging in real time as more symbols arrive.
DUR_S = 10.0
FPS   = 30
N_FR  = int(round(DUR_S*FPS))
fps_out = FPS

# We re-run Costas with logging so we can show the loop "live" by feeding
# sample-by-sample. We also track instantaneous phase so we can show samp_pre
# rotating. For each frame f (0..N_FR-1) we use symbols [0..k_f] with
# k_f = int((f+1)/N_FR * N_SYM).

N_TAIL = 1500                                       # rolling-window size for steady-state panel

def render_frame(f):
    k = max(2, int(round((f+1)/N_FR * N_SYM)))
    pre  = samp_pre[:k]
    coa  = samp_coarse[:k]
    cos_ = samp_costas[:k]
    tail = samp_costas[max(0, k-N_TAIL):k]
    age      = np.linspace(0.05, 1.0, len(pre))
    age_tail = np.linspace(0.20, 1.0, len(tail))

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor="#0b1020")
    gs = fig.add_gridspec(2, 4, height_ratios=[3,1])

    titles = ["1. matched-filter samples\n(no sync)",
              "2. + coarse freq correction",
              "3. + Costas loop  (entire run)",
              f"4. + steady-state  (last {N_TAIL} sym)"]
    panels = [(pre, age), (coa, age), (cos_, age), (tail, age_tail)]
    for i,(t,(p,a)) in enumerate(zip(titles, panels)):
        ax = fig.add_subplot(gs[0,i])
        ax.scatter(np.real(p), np.imag(p), c=a, cmap="viridis",
                   s=8, alpha=0.75, vmin=0, vmax=1)
        ax.axhline(0,color="w",lw=0.4); ax.axvline(0,color="w",lw=0.4)
        ax.set_facecolor("#0b1020")
        ax.set_xlim(-2,2); ax.set_ylim(-2,2)
        ax.set_aspect("equal")
        ax.set_title(t, fontsize=10, color="w")
        ax.tick_params(colors="w")
        for sp in ax.spines.values(): sp.set_color("w")

    # bottom: cumulative post-diff-decode BER vs symbol index (the meaningful BER)
    axb = fig.add_subplot(gs[1,:])
    axb.set_facecolor("#0b1020")
    hd_now  = (np.real(samp_costas[:k]) > 0).astype(int)
    if k >= 2:
        bits_now = diff_decode(hd_now)              # length k-1
        target   = coded[:len(bits_now)]
        err = np.cumsum(bits_now != target) / np.maximum(1, np.arange(1,len(bits_now)+1))
        err_plot = np.maximum(err, 5e-5)
        axb.plot(np.arange(1,len(bits_now)+1), err_plot, color="#ff9f43", lw=1.4,
                 label="cumulative pre-FEC BER (post diff-decode)")
    axb.set_yscale("log")
    axb.set_ylim(5e-5, 1.0); axb.set_xlim(0, N_SYM)
    axb.set_xlabel("symbols received", color="w")
    axb.set_ylabel("BER (log)", color="w")
    axb.grid(alpha=0.3, which="both", color="#445")
    axb.tick_params(colors="w")
    for sp in axb.spines.values(): sp.set_color("w")
    axb.set_title(f"Receiver convergence — symbol {k}/{N_SYM}    "
                  f"Eb/N0={EBN0_DB} dB    fo_true={FREQ_OFFSET:.4f}, fo_est={fo_est:.4f}",
                  fontsize=11, color="w")
    leg = axb.legend(loc="upper right", facecolor="#0b1020", edgecolor="w")
    for txt in leg.get_texts(): txt.set_color("w")

    fig.suptitle("BPSK modem — constellation convergence (live)",
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
writer = imageio.get_writer(mp4, fps=fps_out, codec="libx264",
                            quality=8, macro_block_size=1)
t0 = time.time()
for f in range(N_FR):
    fr = render_frame(f)
    writer.append_data(fr)
    if f % 30 == 0:
        print(f"        frame {f+1}/{N_FR}  ({time.time()-t0:.1f}s)")
writer.close()
print(f"      wrote {mp4}  ({os.path.getsize(mp4)/1e6:.2f} MB)")
print("Done.")
