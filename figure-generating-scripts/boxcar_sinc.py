import numpy as np
import matplotlib.pyplot as plt

# A boxcar (rectangular) pulse in time and its sinc-shaped spectrum

sample_rate = 100.0  # Hz
N = 4096             # number of time samples / FFT size
t = (np.arange(N) - N // 2) / sample_rate  # centered time axis, in seconds

# Boxcar pulse: amplitude A, total width T (seconds), centered at t=0
A = 1.0
T = 1.0
x = np.where(np.abs(t) <= T / 2, A, 0.0)

# Spectrum via FFT
X = np.fft.fftshift(np.fft.fft(x))
X_mag = np.abs(X) / sample_rate  # scale so height approximates A*T
f = np.fft.fftshift(np.fft.fftfreq(N, d=1 / sample_rate))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
plt.subplots_adjust(wspace=0.3)

ax1.plot(t, x, '-')
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Amplitude")
ax1.set_title("Boxcar Pulse in Time")
ax1.set_xlim(-1.5, 1.5)
ax1.set_ylim(-0.2, 1.3)
ax1.grid()

# Annotate the pulse width T with a double-headed arrow spanning -T/2 to T/2
y_ann = A + 0.05
ax1.annotate("", xy=(-T / 2, y_ann), xytext=(T / 2, y_ann),
             arrowprops=dict(arrowstyle="<->", color="red"))
ax1.text(0.05, y_ann + 0.03, "T", color="red", ha="center", va="bottom", fontsize=16)

ax2.plot(f, X_mag, '-')
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("Magnitude")
ax2.set_title("Sinc-Shaped Spectrum")
ax2.set_xlim(-4, 4)
ax2.grid()

# Point out the sinc nulls, which occur at integer multiples of 1/T
for n in [1, 2, 3]:
    ax2.annotate(rf"$\frac{{{n}}}{{T}}$", xy=(n / T, 0.05), xytext=(n / T + 0.2, 0.25),
                 color="red", ha="center", fontsize=20,
                 arrowprops=dict(arrowstyle="->", color="red"))

fig.savefig('../_images/boxcar_sinc.svg', bbox_inches='tight')
plt.show()
