import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter, freqz

fs = 1e6
N = 512
frac_delay = 0.4  # sub-sample shift we want to apply

# Test signal: modulated Gaussian pulse, something with enough bandwidth
# to make a sub-sample shift visibly matter.
t = np.arange(N) / fs
t0 = (N / 2) / fs
width = 15 / fs
x = np.exp(-((t - t0) ** 2) / (2 * width ** 2)) * np.cos(2 * np.pi * 80e3 * t)

# Windowed-sinc fractional delay filter.
# Ideal response is sinc(n - d); we truncate it and window it to get a usable FIR.
num_taps = 81
n = np.arange(num_taps) - (num_taps - 1) / 2
print(n)
h = np.sinc(n - frac_delay) * np.blackman(num_taps)
h /= np.sum(h)  # unity DC gain

y = lfilter(h, 1.0, x)

# The filter has a bulk delay of (num_taps-1)/2 integer samples plus the
# frac_delay we asked for. Strip the integer part so the plot shows only
# the fractional shift.
bulk = (num_taps - 1) // 2
y_aligned = y[bulk:]
x_trim = x[: len(y_aligned)]
t_trim = t[: len(y_aligned)]

# Pick a small window around the pulse peak for the zoomed view.
peak = np.argmax(np.abs(x_trim))
zs, ze = peak - 12, peak + 12

fig, axes = plt.subplots(3, 1, figsize=(9, 8))

axes[0].stem(n, h, basefmt=" ")
axes[0].set_title(f"fractional delay filter taps (d = {frac_delay}, {num_taps} taps, Blackman window)")
axes[0].set_xlabel("tap index (centered)")
axes[0].set_ylabel("amplitude")
axes[0].grid(True)

axes[1].plot(t_trim * 1e6, x_trim, label="original")
axes[1].plot(t_trim * 1e6, y_aligned, "--", label=f"delayed by {frac_delay} samples")
axes[1].set_title("full signal")
axes[1].set_xlabel("time (us)")
axes[1].set_ylabel("amplitude")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(t_trim[zs:ze] * 1e6, x_trim[zs:ze], "o-", label="original")
axes[2].plot(t_trim[zs:ze] * 1e6, y_aligned[zs:ze], "s--", label=f"delayed by {frac_delay} samples")
axes[2].set_title("zoomed view - sub-sample shift is visible between the markers")
axes[2].set_xlabel("time (us)")
axes[2].set_ylabel("amplitude")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()

w, H = freqz(h, worN=2048, fs=fs)

fig2, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(9, 6))
ax_mag.plot(w / 1e3, 20 * np.log10(np.abs(H) + 1e-12))
ax_mag.set_title("frequency response - magnitude")
ax_mag.set_xlabel("frequency (kHz)")
ax_mag.set_ylabel("magnitude (dB)")
ax_mag.set_ylim(-80, 5)
ax_mag.grid(True)

ax_phase.plot(w / 1e3, np.unwrap(np.angle(H)))
ax_phase.set_title("frequency response - phase")
ax_phase.set_xlabel("frequency (kHz)")
ax_phase.set_ylabel("phase (rad)")
ax_phase.grid(True)

plt.tight_layout()
plt.show()
