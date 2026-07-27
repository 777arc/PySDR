"""
Power Spectral Density comparison: MSK vs QPSK with Raised Cosine (α=0.35) pulse shaping.

MSK uses a half-sinusoidal pulse shape, giving a naturally compact spectrum.
QPSK with raised-cosine filtering is shown for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Parameters ──────────────────────────────────────────────────────
Rb = 1.0            # Bit rate (normalised to 1)
Tb = 1.0 / Rb       # Bit period
Ts = 2 * Tb          # Symbol period (2 bits per QPSK symbol)
alpha = 0.35         # Roll-off factor for raised-cosine filter
N = 8192             # FFT length for smooth curves
fmax = 3.0           # Max normalised frequency (f·Tb) to display

f = np.linspace(-fmax, fmax, N)
# Avoid exact zeros that cause division issues
f_safe = np.where(f == 0, 1e-30, f)

# ── 1.  MSK Power Spectral Density (closed-form) ───────────────────
# MSK PSD:  S(f) = (16·Tb / π²) · [ cos(2π·f·Tb) / (1 - 16·f²·Tb²) ]²
# Normalised so that Eb = Tb (energy per bit = bit period for unit power)
numerator = np.cos(2 * np.pi * f_safe * Tb)
denominator = 1.0 - 16.0 * (f_safe * Tb) ** 2

# Handle the removable singularities at f·Tb = ±0.25
psd_msk = np.where(
    np.abs(np.abs(f * Tb) - 0.25) < 1e-8,
    (16.0 * Tb / np.pi**2) * (np.pi / 8.0) ** 2,   # L'Hôpital limit
    (16.0 * Tb / np.pi**2) * (numerator / denominator) ** 2,
)

# ── 2.  QPSK with Raised-Cosine filter PSD ─────────────────────────
# The raised-cosine spectrum (frequency domain) for roll-off α:
#   H(f) = { Ts,                                           |f| <= (1-α)/(2Ts)
#          { Ts/2 · [1 + cos(π·Ts/α·(|f| - (1-α)/(2Ts)))], passband edge
#          { 0,                                             |f| > (1+α)/(2Ts)  }
# PSD of QPSK = Es · |H(f)|²  (Es = symbol energy, set to 2·Tb for fair comparison)

f1 = (1 - alpha) / (2 * Ts)   # passband edge
f2 = (1 + alpha) / (2 * Ts)   # stopband edge

H_rc = np.zeros_like(f)
abs_f = np.abs(f)

# Flat region
mask_flat = abs_f <= f1
H_rc[mask_flat] = Ts

# Roll-off region
mask_roll = (abs_f > f1) & (abs_f <= f2)
H_rc[mask_roll] = (Ts / 2.0) * (
    1.0 + np.cos((np.pi * Ts / alpha) * (abs_f[mask_roll] - f1))
)

# PSD of shaped QPSK (energy per symbol Es = 2·Eb = 2·Tb)
Es = 2 * Tb
psd_qpsk_rc = Es * (H_rc / Ts) ** 2   # normalise H so peak PSD matches

# ── 3.  Unfiltered QPSK (rectangular pulse) for reference ──────────
# S(f) = 2·Tb · sinc²(f·Ts)
psd_qpsk_rect = 2 * Tb * np.sinc(f * Ts) ** 2

# ── Normalise all to 0 dB peak ─────────────────────────────────────
psd_msk_dB       = 10 * np.log10(psd_msk / psd_msk.max() + 1e-30)
psd_qpsk_rc_dB   = 10 * np.log10(psd_qpsk_rc / psd_qpsk_rc.max() + 1e-30)
psd_qpsk_rect_dB = 10 * np.log10(psd_qpsk_rect / psd_qpsk_rect.max() + 1e-30)

# ── Plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(f, psd_msk_dB,       linewidth=2.2, label="MSK", color="#E63946")
ax.plot(f, psd_qpsk_rc_dB,   linewidth=2.2, label=f"QPSK  (RC α = {alpha})", color="#457B9D")
ax.plot(f, psd_qpsk_rect_dB, linewidth=1.4, label="QPSK  (rectangular)", color="#457B9D",
        linestyle="--", alpha=0.55)

ax.set_xlim(-fmax, fmax)
ax.set_ylim(-50, 3)
ax.set_xlabel("Normalized Frequency, fTₛ", fontsize=13)
ax.set_ylabel("Power Spectral Density [dB]", fontsize=13)
ax.set_title("MSK vs QPSK - Spectral Comparison", fontsize=14, fontweight="bold")
ax.legend(fontsize=12, loc="upper right")
ax.grid(True, alpha=0.3, linewidth=0.6)

plt.tight_layout()
plt.savefig('../_images/msk_vs_qpsk_spectrum.svg', dpi=150)
plt.show()
print("Plot saved to msk_vs_qpsk_spectrum.svg")