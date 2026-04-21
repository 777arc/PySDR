import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly

filename = "GPS_L1_recording_10ms_4MHz_cf32.iq"
sample_rate = 4e6
chip_rate = 1.023e6 # chips / sec (part of the GPS spec)
num_chips = 1023 # chips per C/A code period
samples_per_code = int(round(sample_rate / chip_rate * num_chips))  # Exact number of samples in one 1 ms code period at 4 MHz
print(f"samples_per_code: {samples_per_code}")
num_integrations = 10 # non-coherent power integrations (so 10 ms total), determines how much of the IQ recording we read in and process!
detection_thresh_dB =  14.0 # Peak-to-mean ratio (PMR) threshold in dB to declare a detection, GPS C/A signals are typically 14–20 dB PMR above threshold with 10ms of integration
gps_svs = list(range(1, 33)) # 1–32 are the main ones, it's possible there are 33-37

##### C/A Code Generation #####
# The GPS C/A code is a Gold code formed by XOR-ing two 10-stage maximal-length
# shift registers (G1 and G2).  G2 is effectively delayed by a satellite-
# specific number of chips before the XOR
# Reference: IS-GPS-200, Table 3-Ia
G2_DELAY = [ # G2 phase delay (chips) for gps_svs 1–32
      5,   6,   7,   8,  17,  18, 139, 140,   #  1–8
    141, 251, 252, 254, 255, 256, 257, 258,   #  9–16
    469, 470, 471, 472, 473, 474, 509, 512,   # 17–24
    513, 514, 515, 516, 859, 860, 861, 862,   # 25–32
    863, 950, 947, 948, 950 # 33-37.  AI thought the last 2 were 744 and 441 for some reason
]

"""G1 LFSR: polynomial x^10 + x^3 + 1, all-ones init, output at stage 10."""
reg = np.ones(10, dtype=np.int8)
G1 = np.empty(num_chips, dtype=np.int8)
for i in range(num_chips):
    G1[i] = reg[9]
    fb = reg[2] ^ reg[9] # stages 3 and 10 (0-indexed: 2 and 9)
    reg = np.roll(reg, 1)
    reg[0] = fb

"""G2 LFSR: polynomial x^10+x^9+x^8+x^6+x^3+x^2+1, all-ones init."""
reg = np.ones(10, dtype=np.int8)
G2 = np.empty(num_chips, dtype=np.int8)
for i in range(num_chips):
    G2[i] = reg[9]
    fb = reg[1]^reg[2]^reg[5]^reg[7]^reg[8]^reg[9]  # taps 2,3,6,8,9,10
    reg = np.roll(reg, 1)
    reg[0] = fb

# 1023-chip C/A PRN code for SV sv (1-32) as float32, 1's and -1's, so BPSK
def make_prn(sv: int) -> np.ndarray:
    g2_delayed = np.roll(G2, G2_DELAY[sv - 1]) # G2 gets delayed by an amount specified in IS-GPS-200, Table 3-Ia
    bits = G1 ^ g2_delayed # bitwise XOR, still 0s and 1s
    return (1 - 2.0 * bits) # convert to BPSK, +1s and −1s

# Pre-compute template signals - conjugate FFTs of all upsampled PRN codes
template_signals = {}
for sv in gps_svs:
    code = make_prn(sv)
    samples = resample_poly(code, samples_per_code, num_chips) # upsample to match our sample rate
    template_signals[sv] = np.conj(np.fft.fft(samples))

# Read in IQ file
n_needed = samples_per_code * num_integrations
x = np.fromfile(filename, dtype=np.complex64, count=n_needed)

# Loop through satellites performing acquisition
print(" SV        PMR [dB]")
for sv in gps_svs:
    # Non-coherent integration: accumulate squared correlation magnitude
    corr_integrated = np.zeros(samples_per_code)
    for k in range(num_integrations):
        blk = x[k * samples_per_code:(k + 1) * samples_per_code]
        sig_fft = np.fft.fft(blk)
        corr = np.fft.ifft(sig_fft * template_signals[sv]) # cross-correlation in freq domain
        corr_integrated += np.abs(corr)**2

    # Normalize by mean and convert to dB
    pmr_db = 10.0 * np.log10(np.max(corr_integrated) / np.mean(corr_integrated))
    print(f"{sv:>3}  {pmr_db:>9.1f} dB")

    if sv == 12:
        corr_plot = corr_integrated


# Plot correlation curve for SV 12
plt.plot(corr_plot / np.max(corr_plot))
plt.title("Correlation Curve for SV 12")
plt.xlabel("Sample Index")
plt.ylabel("Correlation Magnitude")
plt.show()

delay_us = np.argmax(corr_plot) / sample_rate * 1e6
print(f"SV 12 code phase delay: {delay_us:.2f} us")