# BPSK / QPSK / 256-QAM modems in pure Python — built from the PySDR tutorial

Three end-to-end software modems built from scratch with NumPy + Matplotlib,
following the structure of Marc Lichtman's
[**PySDR**](https://pysdr.org/) tutorial — particularly the chapters on
[Synchronization](https://pysdr.org/content/sync.html),
[Pulse Shaping](https://pysdr.org/content/pulse_shaping.html), and
[Digital Modulation](https://pysdr.org/content/digital_modulation.html).

Same chain skeleton, same RRC pulse shape, same channel impairments, three
different modulations — useful for seeing exactly what changes between
constellations.

```
bits  ->  Hamming(7,4) FEC  ->  modulation  ->  RRC pulse shape (sps=8, β=0.35)
      ->  AWGN + carrier freq offset (0.003 cyc/sample) + phase offset (1.2 rad)
      ->  matched filter  ->  coarse freq estimation  ->  Costas loop
      ->  ambiguity resolve  ->  symbol decisions  ->  Hamming decode  ->  BER
```

## Results

### BPSK

Constellation through the synchronisation chain — full ring (no sync) →
two clusters at the residual phase offset (after coarse freq) → clusters
on the real axis (after Costas) → tight steady-state cluster:

![BPSK constellation](bpsk_out/constellations.png)

BER vs Eb/N0 sweep, simulated curve sits exactly on DBPSK-coherent theory,
with the Hamming(7,4) FEC crossover visible around 5–6 dB:

![BPSK BER curve](bpsk_out/ber_vs_ebn0.png)

EVM vs Eb/N0 (5 / 10 / 15 dB) — measured EVM matches the theoretical
`1/√(Es/N0)` lower bound to ~0.1 % across the range; yellow ✕ marks the
true TX symbol positions:

![BPSK EVM](bpsk_out/evm_vs_snr.png)

Eye diagram (200 symbols overlaid post-sync) — wide-open eye:

![BPSK eye](bpsk_out/eye_diagram.png)

### QPSK

Constellation: ring → 4 clusters at the residual angle → clusters on the
diagonals → clean steady-state:

![QPSK constellation](qpsk_out/constellations.png)

BER vs Eb/N0:

![QPSK BER curve](qpsk_out/ber_vs_ebn0.png)

EVM vs Eb/N0 (5 / 10 / 15 dB) — DQPSK is rotationally invariant so the
post-Costas constellation is aligned to its closest 90° multiple before
the EVM measurement:

![QPSK EVM](qpsk_out/evm_vs_snr.png)

Eye diagram (I and Q):

![QPSK eye](qpsk_out/eye_diagram.png)

### 256-QAM

Constellation: blob (no sync) → angled clusters (coarse freq) → 16×16 grid
emerging (Costas) → fully resolved 256-point lattice in steady state:

![256-QAM constellation](qam256_out/constellations.png)

BER vs Eb/N0, hugging Gray-coded 256-QAM theory:

![256-QAM BER curve](qam256_out/ber_vs_ebn0.png)

EVM vs Eb/N0 (20 / 25 / 30 dB) — clean clusters at each of the 256 grid
points, EVM tracks `1/√(Es/N0)` theory to within ~0.2 %:

![256-QAM EVM](qam256_out/evm_vs_snr.png)

Eye diagram — multi-level (every grid level gets a trace):

![256-QAM eye](qam256_out/eye_diagram.png)

Spectra (FFT magnitude after each block of the chain) and waveforms
are also produced — see [`bpsk_out/spectra.png`](bpsk_out/spectra.png),
[`qpsk_out/spectra.png`](qpsk_out/spectra.png),
[`qam256_out/spectra.png`](qam256_out/spectra.png) and the corresponding
`waveforms.png`.

10-second 30-fps live convergence videos in each `*_out/constellation.mp4`.

## What changes between modems

| step | BPSK | QPSK | 256-QAM |
|---|---|---|---|
| bits/symbol | 1 | 2 (Gray) | 8 (Gray, 16×16) |
| coarse freq | `argmax FFT(x²)/2` | `argmax FFT(x⁴)/4` | `argmax FFT(x⁴)/4` (corner-driven) |
| Costas error | `Re(z)·Im(z)` | `sign(Re)·Im − sign(Im)·Re` | decision-directed: `Im(z·conj(d̂))` |
| ambiguity | DBPSK precoding (180°) | DQPSK precoding (90°) | preamble-correlation (4 rotations) |
| typical Eb/N0 | 0–10 dB | 0–10 dB | 18–28 dB |

## Files

| file | what |
|---|---|
| [`bpsk_modem.py`](bpsk_modem.py) | full BPSK chain → `bpsk_out/` |
| [`qpsk_modem.py`](qpsk_modem.py) | full QPSK chain → `qpsk_out/` |
| [`qam256_modem.py`](qam256_modem.py) | full 256-QAM chain → `qam256_out/` |
| [`show_modems.py`](show_modems.py) | interactive 3-row constellation comparison |
| [`show_constellation.py`](show_constellation.py) | BPSK-only interactive viewer |

Each modem script produces:
- `spectra.png` — FFT magnitude (dB) at every stage of the chain
- `waveforms.png` — time-domain Re/Im at every stage
- `constellations.png` — constellation evolution through synchronisation
- `eye_diagram.png` — post-sync matched-filter eye (200 symbols overlaid)
- `ber_vs_ebn0.png` — simulated BER curves vs theory
- `constellation.mp4` — 10 s, 30 fps live convergence video
- `report.txt` — numerical summary

## Running

```bash
pip install -r requirements.txt
python bpsk_modem.py        # ~4 min including video render
python qpsk_modem.py        # ~4 min
python qam256_modem.py      # ~5 min
python show_modems.py       # interactive 3-modem viewer
```

Same Eb/N0 (set by `EBN0_DB` constant near the top of each script) is used
across all three modems. Default is 25 dB so 256-QAM is workable.

## Caveats — what this does NOT have

These are deliberate sim-mode shortcuts. To run on real hardware (PlutoSDR
etc.) you need to add:

1. **Symbol timing recovery** (Gardner / Mueller-Müller). The sim assumes TX
   and RX share a clock and the matched filter group delay is known exactly,
   so we just slice `mf[delay :: SPS]`. Real radios don't.
2. **Frame sync** — sim knows where bit 0 lives. Real radios receive a buffer
   of "wherever you pressed start" and need preamble cross-correlation to
   find symbol 0.
3. **AGC** — sim amplitudes are normalised constants. Real radios deliver
   int16 samples at unknown gain; you need software or hardware AGC ahead
   of the matched filter.
4. **DC notch + IQ imbalance** — Pluto's AD9363 leaks LO and has small
   I/Q imbalance; the sim has neither (AWGN is zero-mean).

The TX path (mod + RRC) is directly portable. The receiver needs the four
items above bolted in front of Costas before it'll lock on real RF.

## Sample results

At Eb/N0 = 25 dB with the default channel (0.003 cyc/sample freq offset,
1.2 rad phase offset):

- **BPSK**: pre-FEC BER ≈ 0 / 7113 sym (theoretical 4×10⁻¹³⁵ at 25 dB; we see zero)
- **QPSK**: pre-FEC BER ≈ 0 / 3556 sym
- **256-QAM**: pre-FEC BER ≈ 10⁻⁴ — the constellation is a clean 16×16 grid

Each modem's BER vs Eb/N0 sweep matches the textbook theory across the full
plotted range.

## Reference

This work follows the structure recommended by Marc Lichtman's
**[PySDR: A Guide to SDR and DSP using Python](https://pysdr.org/)**.
Particularly:

- [Synchronization in Python](https://pysdr.org/content/sync.html)
- [Pulse Shaping](https://pysdr.org/content/pulse_shaping.html)
- [Digital Modulation](https://pysdr.org/content/digital_modulation.html)
- [Filters](https://pysdr.org/content/filters.html)
- [Frequency Domain](https://pysdr.org/content/frequency_domain.html)

If you found this repo searching for **PySDR** examples, the real PySDR
site at https://pysdr.org/ is the source. This repo is just a worked-out
version of three modulations in a single coherent codebase.
