import numpy as np
from scipy.signal import firwin2
import matplotlib.pyplot as plt

sample_rate = 1e6  # Hz

freqs = [0, 100e3, 110e3, 190e3, 200e3, 300e3, 310e3, 500e3]
gains = [1, 1,     0,     0,     0.5,   0.5,   0,     0]
h = firwin2(101, freqs, gains, fs=sample_rate)

N_fft = 4096
f = np.linspace(-sample_rate / 2, sample_rate / 2, N_fft) / 1e3  # kHz
H = np.fft.fftshift(np.fft.fft(h, N_fft))
H_dB = 20 * np.log10(np.abs(H) + 1e-12)

plt.rcParams.update({'font.size': 14})
fig, (ax_req, ax_taps, ax_freq) = plt.subplots(3, 1, figsize=(9, 10))

ax_req.plot(np.array(freqs) / 1e3, gains, '.-')
ax_req.set_title('Requested frequency response')
ax_req.set_xlabel('Frequency [kHz]')
ax_req.set_ylabel('Gain (linear)')
ax_req.grid(True, ls=':')

ax_taps.plot(h, '.-')
ax_taps.set_title('Taps')
ax_taps.set_xlabel('Tap index')
ax_taps.set_ylabel('Amplitude')
ax_taps.grid(True, ls=':')

ax_freq.plot(f, H_dB)
ax_freq.set_title('Actual frequency response')
ax_freq.set_xlabel('Frequency [kHz]')
ax_freq.set_ylabel('Magnitude [dB]')
ax_freq.set_ylim(-100, 5)
ax_freq.grid(True, ls=':')

plt.tight_layout()
plt.show()

""" firwin examples (2x2 plots)
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
"""
