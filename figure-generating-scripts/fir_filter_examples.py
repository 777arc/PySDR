import numpy as np
from scipy.signal import firwin
import matplotlib.pyplot as plt

sample_rate = 1e6  # Hz

# (num_taps, cutoff in Hz)
filters = [
    (21,  100e3),
    (21,  250e3),
    (101, 100e3),
    (101, 250e3),
]

N_fft = 4096
f = np.linspace(-sample_rate / 2, sample_rate / 2, N_fft) / 1e3  # kHz

plt.rcParams.update({'font.size': 14})
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))

for ax_freq, ax_taps, (num_taps, cutoff) in zip(axes1.ravel(), axes2.ravel(), filters):
    h = firwin(num_taps, cutoff, fs=sample_rate)

    H = np.fft.fftshift(np.fft.fft(h, N_fft))
    H_dB = 20 * np.log10(np.abs(H) + 1e-12)

    ax_freq.plot(f, H_dB)
    ax_freq.set_title(f'num_taps={num_taps}, cutoff={cutoff/1e3:.0f} kHz')
    ax_freq.set_xlabel('Frequency [kHz]')
    ax_freq.set_ylabel('Magnitude [dB]')
    ax_freq.set_ylim(-100, 5)
    ax_freq.grid(True, ls=':')

    ax_taps.plot(h, '.-')
    ax_taps.set_title(f'num_taps={num_taps}, cutoff={cutoff/1e3:.0f} kHz')
    ax_taps.set_xlabel('Tap index')
    ax_taps.set_ylabel('Amplitude')
    ax_taps.grid(True, ls=':')

fig1.tight_layout()
fig2.tight_layout()
plt.show()
