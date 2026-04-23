import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

# Original Avata, Ocusync 3.0

filename = '/mnt/c/Users/marclichtman/Downloads/avata_video_and_RC_56MHz_5780MHz.iq'
sample_rate = 56e6
center_freq = 5.78e9
x = np.fromfile(filename, dtype=np.complex64)

x = x[0:2000000]
print(len(x))

# Resample x to 61.44 MHz
x = signal.resample_poly(x, 61440, 56000)
sample_rate = 61.44e6


# 506200, 534800, 562300
# x = x[506200:506200+65000]

# freq shift - will be refined below
nominal_freq_shift = 3.875e6

# Spectrogram
if False:
    fft_size = 1024
    num_rows = len(x) // fft_size # // is an integer division which rounds down
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)
    # Time starts at the top and goes down, eg sample x[0] will be part of the top row displayed
    plt.imshow(spectrogram, aspect='auto', extent = [sample_rate/-2/1e6 + center_freq/1e6, sample_rate/2/1e6 + center_freq/1e6, len(x)/sample_rate, 0]) # type: ignore
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time [s]")
    plt.show()



"""
Find start of ZC sequence using symmetry property.
Reads FFT-size blocks, splits in half, reverses second half, and computes normalized cross-correlation.
The reversed second half should match the first half, producing a strong peak due to CAZAC property.
Returns complex correlation scores for each offset.
The offset with the highest value is the ZC sequence start (not the cyclic prefix start).
"""
def normalized_xcorr(x, y):
    return np.vdot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

fft_size = int(sample_rate / 15e3)
print(f"FFT size: {fft_size}")
short_cp_len = int(round(0.00000469 * sample_rate))

# Buffer for cross-correlation scores; only compute where a full window fits
scores = np.zeros(len(x) - fft_size - short_cp_len, dtype=np.complex128)
# Iterate over possible start offsets
for start_offset in range(len(scores)):
    # Skip in by one short cyclic prefix, extract fft_size samples
    window = x[start_offset + short_cp_len : start_offset + short_cp_len + fft_size]
    window_one = window[:fft_size // 2] # First half of OFDM symbol
    window_two = window[fft_size // 2:][::-1] # Second half, reversed
    scores[start_offset] = normalized_xcorr(window_one, window_two)

max_offset = np.argmax(np.abs(scores))
print(f"Maximum correlation at offset {max_offset}: {scores[max_offset]}")

plt.figure()
plt.plot(np.abs(scores))
plt.xlabel("Offset")
plt.ylabel("Normalized Cross-Correlation Magnitude")
plt.title("ZC Sequence Start Detection via Symmetry Property")
plt.grid(True)
plt.show()



# fft_size = int(sample_rate / 15e3)
# Nzc = 601
# data_carrier_count = 600
# dc = int(fft_size // 2)
# mapping = np.zeros(fft_size, dtype=int)
# mapping[dc - data_carrier_count // 2 : dc] = 1
# mapping[dc + 1 : dc + 1 + data_carrier_count // 2] = 1
# data_carrier_indices = np.where(mapping == 1)[0]

# # Pre-generate templates for both candidate roots
# templates = {}
# for root in [147, 600]:
#     zc_test = np.exp(-1j * np.pi * root * np.arange(Nzc) * np.arange(1, Nzc + 1) / Nzc)
#     zc_test = np.delete(zc_test, Nzc // 2)
#     sf = np.zeros(fft_size, dtype=complex)
#     sf[data_carrier_indices] = zc_test
#     templates[root] = np.fft.ifft(np.fft.fftshift(sf))

# # Sweep freq offsets ±1 MHz in 15 kHz steps (subcarrier spacing) using a short chunk
# freq_offsets = np.arange(-1e6, 1e6 + 1, 15e3)
# chunk = x[0:300000]
# t_chunk = np.arange(len(chunk)) / sample_rate

# best_root = -1
# best_freq_shift = nominal_freq_shift
# best_peak = 0
# print(f"Sweeping {len(freq_offsets)} freq offsets x 2 roots...")
# for df in freq_offsets:
#     fs_trial = nominal_freq_shift + df
#     chunk_shifted = chunk * np.exp(2j * np.pi * fs_trial * t_chunk)
#     for root, tmpl in templates.items():
#         corr_test = np.abs(signal.fftconvolve(chunk_shifted, tmpl[::-1].conj(), mode='valid'))
#         peak = np.max(corr_test)
#         if peak > best_peak:
#             best_peak = peak
#             best_root = root
#             best_freq_shift = fs_trial

# print(f"Best root: {best_root}, best freq shift: {best_freq_shift/1e6:.6f} MHz (offset from nominal: {(best_freq_shift - nominal_freq_shift)/1e3:.1f} kHz)")

# # Apply best freq shift to full signal
# t = np.arange(len(x)) / sample_rate
# x = x * np.exp(2j * np.pi * best_freq_shift * t)
# template = templates[best_root]

# corr = np.abs(signal.fftconvolve(x, template[::-1].conj(), mode='valid'))

# peak_idx = np.argmax(corr)
# peak_val = corr[peak_idx]
# # Use 95th percentile as "rest" level to compare against secondary peaks, not just noise floor
# p95 = np.percentile(corr, 95)
# bg_val = np.median(corr)
# print(f"Peak: {peak_val:.1f} at index {peak_idx}, median: {bg_val:.4f}, 95th pct: {p95:.4f}, peak/median: {peak_val/bg_val:.1f}x, peak/95th: {peak_val/p95:.1f}x")

# plt.figure()
# plt.plot(corr)
# plt.xlabel("Sample Offset")
# plt.ylabel("Correlation Magnitude")
# plt.title(f"Correlation with ZC root={best_root}")
# plt.grid(True)
# plt.show()