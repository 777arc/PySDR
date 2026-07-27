import numpy as np
import matplotlib.pyplot as plt

filename = "GPS_L1_recording_10ms_4MHz_cf32.iq"
sample_rate = 4e6
chip_rate = 1023000 # chips / sec (part of the GPS spec)
num_chips = 1023 # chips per C/A code period
samples_per_code = int(round(sample_rate / chip_rate * num_chips))  # Exact number of samples in one 1 ms code period at 4 MHz
doppler_min_hz = -5e3 # GPS Doppler ≈ ±4 kHz for stationary receiver
doppler_max_hz = 5e3
doppler_step_hz = 500 # good enough for a coarse search
num_integrations = 10 # non-coherent power integrations (so 10 ms total), determines how much of the IQ recording we read in and process!
detection_thresh_dB =  14.0 # Peak-to-mean ratio (PMR) threshold in dB to declare a detection, GPS C/A signals are typically 14–20 dB PMR above threshold with 10ms of integration
gps_svs = list(range(1, 33)) # 1–32

##### C/A Code Generation #####
# The GPS C/A code is a Gold code formed by XOR-ing two 10-stage maximal-length
# shift registers (G1 and G2).  G2 is effectively delayed by a satellite-
# specific number of chips before the XOR
# Reference: IS-GPS-200, Table 3-Ia
G2_DELAY = [ # G2 phase delay (chips) for gps_svs 1–32
      5,   6,   7,   8,  17,  18, 139, 140,   #  1– 8
    141, 251, 252, 254, 255, 256, 257, 258,   #  9–16
    469, 470, 471, 472, 473, 474, 509, 512,   # 17–24
    513, 514, 515, 516, 859, 860, 861, 862,   # 25–32
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
    g2_delayed = np.roll(G2, G2_DELAY[sv - 1])
    bits = G1 ^ g2_delayed           # {0, 1}
    return (1 - 2 * bits).astype(np.float32)   # BPSK: {+1, −1}

def upsample_prn(sv: int) -> np.ndarray:
    """Nearest-neighbour upsample 1023-chip C/A code → samples_per_code samples."""
    code = make_prn(sv)
    idx = (np.arange(samples_per_code) * num_chips / samples_per_code).astype(int)
    return code[idx]

# Pre-compute template signals - conjugate FFTs of all upsampled PRN codes
template_signals = {sv: np.conj(np.fft.fft(upsample_prn(sv))) for sv in gps_svs}

# Read in IQ file
n_needed = samples_per_code * num_integrations
iq = np.fromfile(filename, dtype=np.complex64, count=n_needed)
# For the full version from IQEngine use the following instead
#iq = np.fromfile(filename, dtype=np.int16, count=n_needed * 2)
#iq = (iq[0::2] + 1j * iq[1::2]).astype(np.complex64)

# Create a spectrogram to show how the signals are under the noise floor
if False:
    fft_size = 512
    num_rows = len(iq) // fft_size # // is an integer division which rounds down
    spectrogram = np.zeros((num_rows, fft_size))
    for i in range(num_rows):
        spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(iq[i*fft_size:(i+1)*fft_size])))**2)
    plt.figure(2)
    plt.imshow(spectrogram, aspect='auto', extent = [sample_rate/-2/1e6, sample_rate/2/1e6, len(iq)/sample_rate, 0])
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Time [s]")
    plt.savefig('../_images/detection_gps_spectrogram.svg', bbox_inches='tight')
    plt.show()
    exit()

# Loop through satellites performing acquisition
results = []
detected = []
print(f"  {'SV':>3}  {'Doppler (Hz)':>13}  {'Phase (chips)':>14}"
          f"  {'Phase (samp)':>13}  {'Delay (µs)':>11}  {'PMR (dB)':>9}")
doppler_bins = np.arange(doppler_min_hz, doppler_max_hz + doppler_step_hz, doppler_step_hz)
for sv in gps_svs:
    corr_map = np.zeros((len(doppler_bins), samples_per_code))
    n_total = samples_per_code * num_integrations
    for di, f_d in enumerate(doppler_bins):
        t = np.arange(n_total) / sample_rate # time vector
        mixed = iq[:n_total] * np.exp(-2j*np.pi*float(f_d)*t) # freq shift

        # Non-coherent integration: accumulate squared correlation magnitude
        for k in range(num_integrations):
            blk = mixed[k * samples_per_code:(k + 1) * samples_per_code]
            sig_fft = np.fft.fft(blk)
            corr = np.fft.ifft(sig_fft * template_signals[sv]) # cross-correlation in freq domain
            corr_map[di] += np.abs(corr)**2

    # Normalize by mean and convert to dB
    peak_val = float(np.max(corr_map))
    mean_val = float(np.mean(corr_map))
    pmr_db = 10.0 * np.log10(peak_val / mean_val)

    peak_idx = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    best_doppler_hz   = float(doppler_bins[peak_idx[0]])
    best_phase_samp   = int(peak_idx[1])
    best_phase_chips  = best_phase_samp * num_chips / samples_per_code

    r = {
        "sv": sv,
        "detected": pmr_db >= detection_thresh_dB,
        "doppler_hz": best_doppler_hz,
        "code_phase_samp": best_phase_samp, # sample offset = "start of packet"
        "code_phase_chip": best_phase_chips,
        "pmr_db": pmr_db,
        "corr_map": corr_map,
        "doppler_bins": doppler_bins,
    }
    results.append(r)

    # Print row
    delay_us = r['code_phase_samp'] / sample_rate * 1e6
    flag = "  ← DETECTED" if r['detected'] else ""
    print(f"  {sv:>3}  {r['doppler_hz']:>+13.0f}  {r['code_phase_chip']:>14.2f}"
          f"  {r['code_phase_samp']:>13d}  {delay_us:>11.3f}  {r['pmr_db']:>9.1f}{flag}")


# Plotting
sv = 11 # we detected 11, 12, 22, 25, 31, 32 although try looking at one we didnt find as well!
r = results[sv - 1] # print the dict of results for this SV to see what we got
cmap = r['corr_map'] # 2-D array of correlation power vs Doppler and code phase
d_bins = r['doppler_bins'] # Doppler bins corresponding
chips_axis = np.arange(samples_per_code) * num_chips / samples_per_code

# 2-D Doppler × code-phase map
plt.figure(0, figsize=(10, 6))
im = plt.pcolormesh(chips_axis, d_bins, cmap, shading='auto', cmap='viridis')
plt.xlabel("Code Phase (chips)")
plt.ylabel("Doppler (Hz)")
plt.title(f"SV {sv}  —  2-D Acquisition Map  (PMR = {r['pmr_db']:.1f} dB)")
plt.legend(fontsize=8, loc='upper right')
plt.colorbar(im, label="Correlation Power")
plt.savefig('../_images/detection_gps_2d_map.png', bbox_inches='tight', dpi=300)

# code-phase slice at best Doppler
best_di = int(np.argmin(np.abs(d_bins - r['doppler_hz'])))
plt.figure(1, figsize=(8, 4))
plt.plot(chips_axis, cmap[best_di], lw=1, color='steelblue')
plt.xlabel("Code Phase (chips)")
plt.ylabel("Correlation Power")
plt.title(f"SV {sv}  —  Code-Phase Slice  (Doppler = {r['doppler_hz']:+.0f} Hz)")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.savefig('../_images/detection_gps_code_phase_slice.svg', bbox_inches='tight')

plt.show()

