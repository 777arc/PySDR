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

# freq shift
freq_shift = 3.875e6
t = np.arange(len(x)) / sample_rate
x = x * np.exp(2j * np.pi * freq_shift * t)

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



# """
# Find start of ZC sequence using symmetry property.
# Reads FFT-size blocks, splits in half, reverses second half, and computes normalized cross-correlation.
# The reversed second half should match the first half, producing a strong peak due to CAZAC property.
# Returns complex correlation scores for each offset.
# The offset with the highest value is the ZC sequence start (not the cyclic prefix start).
# """
# def normalized_xcorr(x, y):
#     return np.vdot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

# fft_size = int(sample_rate / 15e3)
# print(f"FFT size: {fft_size}")
# short_cp_len = int(round(0.00000469 * sample_rate))

# # Buffer for cross-correlation scores; only compute where a full window fits
# scores = np.zeros(len(x) - fft_size - short_cp_len, dtype=np.complex128)
# # Iterate over possible start offsets
# for start_offset in range(len(scores)):
#     # Skip in by one short cyclic prefix, extract fft_size samples
#     window = x[start_offset + short_cp_len : start_offset + short_cp_len + fft_size]
#     window_one = window[:fft_size // 2] # First half of OFDM symbol
#     window_two = window[fft_size // 2:][::-1] # Second half, reversed
#     scores[start_offset] = normalized_xcorr(window_one, window_two)

# max_offset = np.argmax(np.abs(scores))
# print(f"Maximum correlation at offset {max_offset}: {scores[max_offset]}")

# plt.figure()
# plt.plot(np.abs(scores))
# plt.xlabel("Offset")
# plt.ylabel("Normalized Cross-Correlation Magnitude")
# plt.title("ZC Sequence Start Detection via Symmetry Property")
# plt.grid(True)
# plt.show()

fft_size = int(sample_rate / 15e3)
root = 147 #600
Nzc = 601
zc = np.exp(-1j * np.pi * root * np.arange(Nzc) * np.arange(1, Nzc + 1) / Nzc)
zc = np.delete(zc, Nzc // 2)  # # Remove the middle value (DC). Nzc//2 is 301 for Nzc=601
samples_freq = np.zeros((fft_size), dtype=complex) # Create buffer for freq domain carriers
data_carrier_count = 600
dc = int(fft_size // 2)  # Python uses 0-based indexing
mapping = np.zeros(fft_size, dtype=int)
mapping[dc - data_carrier_count // 2 : dc] = 1
mapping[dc + 1 : dc + 1 + data_carrier_count // 2] = 1
data_carrier_indices = np.where(mapping == 1)[0]
samples_freq[data_carrier_indices] = zc
template = np.fft.ifft(np.fft.fftshift(samples_freq)) # Convert to time domain, flipping spectrum left to right first

# Correlate x against the template
corr = signal.correlate(x, template, mode='valid')
template_energy = np.linalg.norm(template)
local_energy = np.sqrt(np.convolve(np.abs(x)**2, np.ones(len(template)), mode='valid'))
corr_normalized = np.abs(corr) / (local_energy * template_energy)

peak_idx = np.argmax(corr_normalized)
peak_val = corr_normalized[peak_idx]
# Measure background: median of the correlation
bg_val = np.median(corr_normalized)
print(f"Peak: {peak_val:.4f} at index {peak_idx}, Background (median): {bg_val:.4f}, Ratio: {peak_val/bg_val:.1f}x")

plt.figure()
plt.plot(corr_normalized)
plt.xlabel("Sample Offset")
plt.ylabel("Normalized Correlation")
plt.title("Normalized Correlation of Received Signal with ZC Template")
plt.grid(True)
plt.savefig('/tmp/drone_corr.png')
plt.close()

