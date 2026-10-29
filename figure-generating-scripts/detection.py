# %%
import os
import sys
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, lfilter, upfirdn
from scipy.special import erfc

# %%
# Signal Detection and Correlator Basics
# Long Zadoff-Chu sequence
N = 839  # Length of Zadoff-Chu sequence
u = 25  # Root of ZC sequence
t = np.arange(N)
zadoff_chu = np.exp(-1j * np.pi * u * t * (t + 1) / N)

# Create AWGN and stick the ZC sequence in a random spot
signal_length = 10 * N
offset = np.random.randint(N, signal_length - N)
print(f"True offset: {offset}")
snr_db = -15
noise_power = 1 / (2 * (10**(snr_db / 10)))
signal = np.sqrt(noise_power/2) * (np.random.randn(signal_length) + 1j * np.random.randn(signal_length))
signal[offset:offset+N] += zadoff_chu

plt.figure(0)
plt.plot(np.abs(signal))
plt.xlabel('Sample Index')
plt.ylabel('Signal Magnitude [Linear]')
plt.axvline(float(offset), 0, 1, color='r', linestyle=':')
plt.text(float(offset), -0.5, 'True Offset', color='r', verticalalignment='bottom')
plt.grid()
plt.tight_layout()
#if os.path.isdir('../_images'):
#    plt.savefig('../_images/detection_basic_1.svg', bbox_inches='tight')
# Correlator
correlation = np.abs(np.correlate(signal, zadoff_chu, mode='valid') / N)**2 

plt.figure(1)
plt.plot(correlation)
plt.xlabel('Sample Index')
plt.ylabel('Correlation Magnitude [Linear]')
plt.axvline(float(offset), 0, 0.2, color='r', linestyle=':')
plt.text(float(offset), -0.05, 'True Offset', color='r', verticalalignment='bottom')
plt.grid()
plt.tight_layout()
#if os.path.isdir('../_images'):
#    plt.savefig('../_images/detection_basic_2.svg', bbox_inches='tight')
plt.show()

# %%
# Preamble detection performance, ROC curves and Pd vs SNR
def get_srrc_pulse(sps, roll_off, span):
    """Generates Square Root Raised Cosine (SRRC) pulse coefficients."""
    t = np.arange(-span * sps, span * sps + 1) / sps
    with np.errstate(divide='ignore', invalid='ignore'):
        pulse = (np.sin(np.pi * t * (1 - roll_off)) + 
                 4 * roll_off * t * np.cos(np.pi * t * (1 + roll_off))) / \
                (np.pi * t * (1 - (4 * roll_off * t)**2))
        pulse[t == 0] = 1 - roll_off + (4 * roll_off / np.pi)
        pulse[np.abs(np.abs(4 * roll_off * t) - 1) < 1e-10] = \
            (roll_off / np.sqrt(2)) * (((1 + 2/np.pi) * np.sin(np.pi / (4 * roll_off))) + \
            ((1 - 2/np.pi) * np.cos(np.pi / (4 * roll_off))))
    return pulse / np.sqrt(np.sum(pulse**2))

# Simulation Parameters
N_preamble = 16       # Preamble symbol length
N_data = 1000         # Data symbol length
sps = 4               # Samples per symbol
roll_off = 0.35       # SRRC roll-off factor
snr_db_list = [-20, -15, -10, -5, 0]  # SNR values for ROC
mc_trials = 10000      # Monte Carlo trials per SNR
if sys.platform == 'emscripten': # running in the web-based marimo notebook, which is much slower, so use fewer trials (noisier curves)
    mc_trials = 100

# 1. Generate QPSK Signal
qpsk_map = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
preamble = qpsk_map[np.random.randint(0, 4, N_preamble)]
data = qpsk_map[np.random.randint(0, 4, N_data)]
symbols = np.concatenate([preamble, data])

# 2. Pulse Shaping
pulse = get_srrc_pulse(sps, roll_off, span=6)
upsampled_symbols = np.zeros(len(symbols) * sps, dtype=complex)
upsampled_symbols[::sps] = symbols
tx_signal = np.convolve(upsampled_symbols, pulse, mode='same')

# 3. Monte Carlo Simulation for ROC Curves
plt.figure(figsize=(12, 5))
thresholds = np.linspace(0, 2, 200)
if sys.platform == 'emscripten': # fewer points along each ROC curve in the web-based marimo notebook, to save time
    thresholds = np.linspace(0, 2, 20)

for _snr_db in snr_db_list:
    print(f"Simulating for SNR={_snr_db}dB...")
    pd_curve = []
    pfa_curve = []
    
    # Noise power calculation
    snr_linear = 10**(_snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_linear))
    
    for thresh in thresholds:
        detections = 0
        false_alarms = 0
        
        for _ in range(mc_trials):
            # Trial under H1 (Signal + Noise)
            noise = noise_std * (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal)))
            rx_h1 = tx_signal + noise
            # Test statistic: Correlation with preamble
            corr = np.abs(np.sum(np.conj(tx_signal[:N_preamble*sps]) * rx_h1[:N_preamble*sps])) / (N_preamble*sps)
            if corr > thresh: detections += 1
            
            # Trial under H0 (Noise only)
            rx_h0 = noise_std * (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal)))
            corr_h0 = np.abs(np.sum(np.conj(tx_signal[:N_preamble*sps]) * rx_h0[:N_preamble*sps])) / (N_preamble*sps)
            if corr_h0 > thresh: false_alarms += 1
            
        pd_curve.append(detections / mc_trials)
        pfa_curve.append(false_alarms / mc_trials)

        print(f"thres={thresh} | SNR={_snr_db}")
    
    plt.subplot(1, 2, 1)
    plt.plot(pfa_curve, pd_curve, label=f'SNR={_snr_db}dB')

plt.subplot(1, 2, 1)
plt.title("ROC Curves")
plt.xlabel("Pfa")
plt.ylabel("Pd")
plt.legend()
plt.grid(True)

# 4. Pd vs SNR Curve (Fixed Pfa)
target_pfa = 0.01
snr_range = np.arange(-10, 15, 2)
pd_vs_snr = []

for _snr_db in snr_range:
    snr_linear = 10**(_snr_db / 10)
    noise_std = np.sqrt(1 / (2 * snr_linear))
    # Determine threshold for target Pfa under H0
    noise_trials = [np.abs(np.sum(np.conj(tx_signal[:N_preamble*sps]) * (noise_std*(np.random.randn(N_preamble*sps)+1j*np.random.randn(N_preamble*sps))))) / (N_preamble*sps) for _ in range(1000)]
    thresh = np.percentile(noise_trials, 100 * (1 - target_pfa))
    
    # Calculate Pd
    detections = 0
    for _ in range(mc_trials):
        rx = tx_signal + noise_std * (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal)))
        if np.abs(np.sum(np.conj(tx_signal[:N_preamble*sps]) * rx[:N_preamble*sps])) / (N_preamble*sps) > thresh:
            detections += 1
    pd_vs_snr.append(detections / mc_trials)

plt.subplot(1, 2, 2)
plt.plot(snr_range, pd_vs_snr, 'o-')
plt.title(f"Pd vs SNR (Pfa={target_pfa})")
plt.xlabel("SNR (dB)")
plt.ylabel("Pd")
plt.grid(True)
plt.tight_layout()
#if os.path.isdir('../_images'):
#    plt.savefig('../_images/detection_pd_vs_snr.svg', bbox_inches='tight')
plt.show()

# %%
# Example: Detecting GPS Signals Below the Noise Floor
# If the recording isn't in the current directory it gets downloaded from GitHub, which is also how the web-based marimo notebook gets it
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
if os.path.exists(filename):
    iq = np.fromfile(filename, dtype=np.complex64, count=n_needed)
else:
    iq = np.frombuffer(urllib.request.urlopen('https://raw.githubusercontent.com/777arc/PySDR/refs/heads/master/figure-generating-scripts/' + filename).read(), dtype=np.complex64)[:n_needed]
# For the full version from IQEngine use the following instead
#iq = np.fromfile(filename, dtype=np.int16, count=n_needed * 2)
#iq = (iq[0::2] + 1j * iq[1::2]).astype(np.complex64)

# %%
# Create a spectrogram to show how the signals are under the noise floor
fft_size = 512
num_rows = len(iq) // fft_size # // is an integer division which rounds down
spectrogram = np.zeros((num_rows, fft_size))
for _i in range(num_rows):
    spectrogram[_i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(iq[_i*fft_size:(_i+1)*fft_size])))**2)
plt.figure(2)
plt.imshow(spectrogram, aspect='auto', extent = [sample_rate/-2/1e6, sample_rate/2/1e6, len(iq)/sample_rate, 0])
plt.xlabel("Frequency [MHz]")
plt.ylabel("Time [s]")
#if os.path.isdir('../_images'):
#    plt.savefig('../_images/detection_gps_spectrogram.svg', bbox_inches='tight')
plt.show()

# %%
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
        _t = np.arange(n_total) / sample_rate # time vector
        mixed = iq[:n_total] * np.exp(-2j*np.pi*float(f_d)*_t) # freq shift

        # Non-coherent integration: accumulate squared correlation magnitude
        for k in range(num_integrations):
            blk = mixed[k * samples_per_code:(k + 1) * samples_per_code]
            sig_fft = np.fft.fft(blk)
            _corr = np.fft.ifft(sig_fft * template_signals[sv]) # cross-correlation in freq domain
            corr_map[di] += np.abs(_corr)**2

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
#if os.path.isdir('../_images'):
#    plt.savefig('../_images/detection_gps_2d_map.png', bbox_inches='tight', dpi=300)
# code-phase slice at best Doppler
best_di = int(np.argmin(np.abs(d_bins - r['doppler_hz'])))
plt.figure(1, figsize=(8, 4))
plt.plot(chips_axis, cmap[best_di], lw=1, color='steelblue')
plt.xlabel("Code Phase (chips)")
plt.ylabel("Correlation Power")
plt.title(f"SV {sv}  —  Code-Phase Slice  (Doppler = {r['doppler_hz']:+.0f} Hz)")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
#if os.path.isdir('../_images'):
#    plt.savefig('../_images/detection_gps_code_phase_slice.svg', bbox_inches='tight')
plt.show()

# %%
# CFAR Detectors
def generate_qpsk_packets(num_packets, sps, preamble):
    """Generates repeating QPSK packets with gaps and varying noise."""
    qpsk_map = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
    data_len = 200
    gap_len = 100
    full_signal = []
    
    # Pre-calculate preamble upsampled for correlation
    upsampled_preamble = np.repeat(preamble, sps)
    
    for _ in range(num_packets):
        data = qpsk_map[np.random.randint(0, 4, data_len)]
        packet = np.concatenate([preamble, data])
        full_signal.extend(np.repeat(packet, sps))
        full_signal.extend(np.zeros(gap_len * sps))
    
    return np.array(full_signal), upsampled_preamble

# 1. Setup Parameters
_sps = 4
preamble_syms = np.array([1+1j, 1+1j, -1-1j, -1-1j, 1-1j, -1+1j]) / np.sqrt(2)
_tx_signal, ref_preamble = generate_qpsk_packets(5, _sps, preamble_syms)

# 2. Channel: Time-Varying Noise Floor
_t = np.arange(len(_tx_signal))
noise_env = 0.05 + 0.3 * np.sin(2 * np.pi * 0.0003 * _t)**2
_noise = (np.random.randn(len(_tx_signal)) + 1j*np.random.randn(len(_tx_signal))) * noise_env
rx_signal = _tx_signal + _noise

# 3. Preamble Correlation
# Correlation spike occurs when the reference matches the received segment
corr_out = correlate(rx_signal, ref_preamble, mode='same')
corr_power = np.abs(corr_out)**2

# 4. CFAR Detection on Correlator Output
def ca_cfar_adaptive(data, num_train, num_guard, pfa):
    num_cells = len(data)
    thresholds = np.zeros(num_cells)
    alpha = num_train * (pfa**(-1/num_train) - 1)  # Scaling factor
    
    half_window = (num_train + num_guard) // 2
    guard_half = num_guard // 2
    
    for i in range(half_window, num_cells - half_window):
        # Extract training cells (excluding guard cells and CUT)
        lagging_win = data[i - half_window : i - guard_half]
        leading_win = data[i + guard_half + 1 : i + half_window + 1]
        noise_floor_est = np.mean(np.concatenate([lagging_win, leading_win]))
        
        thresholds[i] = alpha * noise_floor_est
        
    return thresholds

# Detect on correlator power
cfar_thresholds = ca_cfar_adaptive(corr_power, num_train=60, num_guard=20, pfa=1e-5)
_detections = np.where(corr_power > cfar_thresholds)[0]
# Filter detections to only include those where threshold is non-zero (avoid edges)
_detections = _detections[cfar_thresholds[_detections] > 0]

# 5. Visualization
plt.figure(figsize=(14, 8))

# Subplot 1: Received Signal and Raw Power
plt.subplot(2, 1, 1)
plt.plot(np.abs(rx_signal)**2, color='gray', alpha=0.4, label='Rx Signal Power ($|r(t)|^2$)')
plt.title("Time-Domain Received Signal")
plt.ylabel("Power")
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Correlator Output vs Adaptive Threshold
plt.subplot(2, 1, 2)
plt.plot(corr_power, label='Correlator Output $|r(t) * p^*(-t)|^2$', color='blue')
plt.plot(cfar_thresholds, label='CFAR Adaptive Threshold', color='red', linestyle='--', linewidth=1.5)

# Overlay detections
if len(_detections) > 0:
    plt.scatter(_detections, corr_power[_detections], color='lime', edgecolors='black', 
                label='Detections (Preamble Found)', zorder=5)

plt.title("Preamble Correlator Output with Adaptive CFAR Threshold")
plt.xlabel("Sample Index")
plt.ylabel("Correlation Power")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
#if os.path.isdir('../_images'):
#    plt.savefig('../_images/detection_cfar.svg', bbox_inches='tight')
plt.show()

# %%
# Frequency offset, correlation degradation and detection probability
# ==============================
# Helper Functions
# ==============================

def srrc_pulse(alpha, sps, span):
    """Generate SRRC pulse."""
    t = np.arange(-span/2, span/2, 1/sps)
    p = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti == 0:
            p[i] = 1.0 - alpha + 4*alpha/np.pi
        elif abs(ti) == 1/(4*alpha):
            p[i] = (alpha/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*alpha))+(1-2/np.pi)*np.cos(np.pi/(4*alpha)))
        else:
            p[i] = (np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))) / (np.pi*ti*(1-(4*alpha*ti)**2))
    return p/np.sqrt(np.sum(p**2))

def qpsk_mod(bits):
    """Map bits to QPSK symbols."""
    bits = bits.reshape(-1, 2)
    mapping = { (0,0): 1+1j, (0,1): -1+1j, (1,1): -1-1j, (1,0): 1-1j }
    symbols = np.array([mapping[tuple(b)] for b in bits]) / np.sqrt(2)
    return symbols

def apply_freq_offset(x, Fs, f_offset):
    """Apply frequency offset."""
    n = np.arange(len(x))
    return x * np.exp(1j*2*np.pi*f_offset*n/Fs)

def awgn(x, snr_dB):
    """Add AWGN."""
    snr_lin = 10**(snr_dB/10)
    P = np.mean(np.abs(x)**2)
    N0 = P/snr_lin
    noise = np.sqrt(N0/2) * (np.random.randn(len(x)) + 1j*np.random.randn(len(x)))
    return x + noise

def preamble_detector(rx, preamble, Fs, sps, freqs, seg_len, coherent=True):
    """Perform frequency search and segmented correlation."""
    corr_vals = []
    preamble_up = upfirdn([1], preamble, sps)
    for f in freqs:
        rxf = rx * np.exp(-1j*2*np.pi*f*np.arange(len(rx))/Fs)
        seg_corrs = []
        for i in range(0, len(preamble_up), seg_len*sps):
            segment = preamble_up[i:i+seg_len*sps]
            if len(segment) > len(rxf): break
            c = np.abs(np.vdot(rxf[:len(segment)], segment))
            if coherent:
                seg_corrs.append(np.vdot(rxf[:len(segment)], segment))
            else:
                seg_corrs.append(np.abs(np.vdot(rxf[:len(segment)], segment)))
        if coherent:
            total_corr = np.abs(np.sum(seg_corrs))
        else:
            total_corr = np.sum(seg_corrs)
        corr_vals.append(total_corr)
    return np.array(corr_vals)


def determine_freq_spacing(f_max, deg_dB, seg_len, symbol_rate):
    """Decide frequency step spacing based on allowed correlation degradation."""
    # degradation = sinc(f_offset * seg_duration)
    # want degradation (in dB) < max tolerated
    deg_lin = 10**(-deg_dB/20)
    seg_dur = seg_len / symbol_rate
    f_step = (1 / (2*seg_dur)) * np.sqrt(1 - deg_lin)
    freqs = np.arange(-f_max, f_max+f_step, f_step)
    return freqs



# Parameters
_N_preamble = 64
_N_data = 512
_sps = 8
rolloff = 0.35
span = 8
Fs = 1e6
symbol_rate = Fs / _sps

f_max = 5e3
deg_dB = 1
seg_symb_min = 8
EbN0_dBs = [0, 5, 10]
num_trials = 200
if sys.platform == 'emscripten': # running in the web-based marimo notebook, which is much slower, so use fewer trials (noisier curves)
    num_trials = 20

# Generate preamble + data
bits = np.random.randint(0, 2, (_N_preamble+_N_data)*2)
_symbols = qpsk_mod(bits)
_preamble = _symbols[:_N_preamble]
_data = _symbols[_N_preamble:]

# SRRC shaping
_pulse = srrc_pulse(rolloff, _sps, span)
tx = upfirdn(_pulse, np.concatenate([_preamble, _data]), _sps)
tx = tx / np.sqrt(np.mean(np.abs(tx)**2))

# Determine frequency sweep
freqs = determine_freq_spacing(f_max, deg_dB, seg_symb_min, symbol_rate)

# ======================
# Sweep: frequency offset
# ======================
freq_offsets = np.linspace(-f_max, f_max, 15)
_results = {snr: [] for snr in EbN0_dBs}

for snr in EbN0_dBs:
    for fo in freq_offsets:
        corr_peaks = []
        for _ in range(num_trials):
            _rx = apply_freq_offset(tx, Fs, fo)
            _rx = awgn(_rx, snr)
            _corr = preamble_detector(_rx, _preamble, Fs, _sps, freqs, seg_symb_min, coherent=False)
            corr_peaks.append(np.max(_corr))
        _results[snr].append(np.mean(corr_peaks))

plt.figure()
for snr in EbN0_dBs:
    plt.plot(freq_offsets/1e3, 20*np.log10(_results[snr]/np.max(_results[snr])), label=f"SNR={snr}dB")
plt.xlabel("Frequency offset (kHz)")
plt.ylabel("Normalized correlation peak (dB)")
plt.legend()
plt.title("Correlation degradation vs frequency offset")
plt.grid(True)
#if os.path.isdir('../_images'):
#    plt.savefig('../_images/detection_freq_offset.svg', bbox_inches='tight')
plt.show()

# ======================
# Sweep: SNR vs detection probability
# ======================
_snr_range = np.linspace(-5, 20, 10)
test_offsets = [0, 2e3, 5e3]
threshold = 0.3  # detection threshold
det_prob = {fo: [] for fo in test_offsets}

for fo in test_offsets:
    for snr in _snr_range:
        _detections = 0
        for _ in range(num_trials):
            _rx = apply_freq_offset(tx, Fs, fo)
            _rx = awgn(_rx, snr)
            _corr = preamble_detector(_rx, _preamble, Fs, _sps, freqs, seg_symb_min, coherent=False)
            if np.max(_corr)/np.max(_corr) > threshold:
                _detections += 1
        det_prob[fo].append(_detections/num_trials)
    plt.plot(_snr_range, det_prob[fo], label=f"Offset={fo/1e3:.1f} kHz")

plt.xlabel("SNR (dB)")
plt.ylabel("Probability of detection")
plt.title("Detection probability vs SNR for various frequency offsets")
plt.legend()
plt.grid(True)
#if os.path.isdir('../_images'):
#    plt.savefig('../_images/detection_freq_offset2.svg', bbox_inches='tight')
plt.show()

# %%
# Detecting Direct Sequence Spread Spectrum (DSSS) Signals
# Barker 11 sequence: +1, -1, +1, +1, -1, +1, +1, +1, -1, -1, -1
barker11 = np.array([1, -1, 1, 1, -1, 1, 1, 1, -1, -1, -1])
samples_per_chip = 100

# Upsample the sequence to simulate continuous-ish time
sig = np.repeat(barker11, samples_per_chip)

offsets = np.linspace(-1.5, 1.5, 500) # Fractional chip offsets
peaks = []

for _offset in offsets:
    # Shift the signal by a fractional number of chips (converted to samples)
    shift_samples = int(_offset * samples_per_chip)
    if shift_samples > 0:
        shifted_sig = np.pad(sig, (shift_samples, 0))[:len(sig)]
    elif shift_samples < 0:
        shifted_sig = np.pad(sig, (0, abs(shift_samples)))[abs(shift_samples):]
    else:
        shifted_sig = sig
        
    # Compute normalized correlation at zero lag for this specific offset
    _correlation = np.vdot(sig, shifted_sig) / np.vdot(sig, sig)
    peaks.append(np.abs(_correlation))

plt.figure(figsize=(10, 5))
plt.plot(offsets, peaks, label='Normalized Correlation', color='blue', linewidth=2)
plt.axvline(0, color='red', linestyle='--', alpha=0.5, label='Perfect Alignment')
plt.title('DSSS Correlation Peak vs. Fractional Chip Timing Offset')
plt.xlabel('Offset (Fraction of a Chip)')
plt.ylabel('Normalized Correlation Peak Magnitude')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
#if os.path.isdir('../_images'):
#    plt.savefig('../_images/detection_dsss.svg', bbox_inches='tight')
plt.show()

# %%
# Real-Time Packet Detection in Continuous IQ Streams, generates the visualization for that section

def ca_cfar_1d(signal, num_train, num_guard, pfa):
    """Cell-Averaging CFAR detector."""
    n = len(signal)
    threshold = np.zeros(n)
    alpha = num_train * (pfa**(-1/num_train) - 1)
    
    for i in range(n):
        train_start_left = max(0, i - num_guard - num_train)
        train_end_left = max(0, i - num_guard)
        train_start_right = min(n, i + num_guard + 1)
        train_end_right = min(n, i + num_guard + num_train + 1)
        
        train_cells = np.concatenate([
            signal[train_start_left:train_end_left],
            signal[train_start_right:train_end_right]
        ])
        
        if len(train_cells) > 0:
            threshold[i] = alpha * np.mean(train_cells)
    
    return threshold

def detect_packets(buffer, preamble, cfar_guard, cfar_train, pfa):
    """Detect packets in IQ buffer."""
    corr = correlate(buffer, preamble, mode='same')
    corr_power = np.abs(corr)**2
    threshold = ca_cfar_1d(corr_power, cfar_train, cfar_guard, pfa)
    
    detections_raw = np.where(corr_power > threshold)[0]
    half_preamble = len(preamble) // 2
    detections_raw = detections_raw - half_preamble
    detections_raw = detections_raw[
        (detections_raw > half_preamble) & 
        (detections_raw < len(buffer) - half_preamble)
    ]
    
    detections = []
    if len(detections_raw) > 0:
        detections.append(detections_raw[0])
        for det in detections_raw[1:]:
            if det - detections[-1] > len(preamble):
                detections.append(det)
    
    return detections, corr_power, threshold

def generate_packet_stream(preamble, packet_length, num_packets, sample_rate, snr_db):
    """Generate simulated IQ stream with packets."""
    signal_power = 1.0
    noise_power = signal_power / (10**(snr_db/10))
    noise_std = np.sqrt(noise_power / 2)
    
    qpsk_map = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
    avg_gap = int(sample_rate / 1)  # 1 packet per second
    
    signal = []
    true_starts = []
    
    for i in range(num_packets):
        if i == 0:
            gap_length = np.random.randint(avg_gap//2, avg_gap)
        else:
            gap_length = np.random.randint(int(avg_gap*0.8), int(avg_gap*1.2))
        
        noise = noise_std * (np.random.randn(gap_length) + 1j*np.random.randn(gap_length))
        signal.extend(noise)
        
        true_starts.append(len(signal))
        
        data_length = packet_length - len(preamble)
        data = qpsk_map[np.random.randint(0, 4, data_length)]
        packet = np.concatenate([preamble, data])
        packet_noisy = packet + noise_std * (np.random.randn(len(packet)) + 
                                            1j*np.random.randn(len(packet)))
        signal.extend(packet_noisy)
    
    gap_length = np.random.randint(avg_gap//2, avg_gap)
    noise = noise_std * (np.random.randn(gap_length) + 1j*np.random.randn(gap_length))
    signal.extend(noise)
    
    return np.array(signal), true_starts

# Generate test signal
np.random.seed(42)
N_zc, _u = 63, 5
_t = np.arange(N_zc)
_preamble = np.exp(-1j * np.pi * _u * _t * (_t + 1) / N_zc)

_sample_rate = 1e6
if sys.platform == 'emscripten': # running in the web-based marimo notebook, which is much slower, so simulate fewer samples (still 1 packet per second)
    _sample_rate = 1e5
packet_length = 500
_snr_db = -5

_signal, true_starts = generate_packet_stream(
    _preamble, packet_length, num_packets=5, 
    sample_rate=_sample_rate, snr_db=_snr_db
)

# Process one buffer for visualization
buffer_start = max(0, true_starts[0] - 5000)
buffer_end = min(len(_signal), true_starts[2] + 10000)
viz_buffer = _signal[buffer_start:buffer_end]

detections_viz, corr_viz, thresh_viz = detect_packets(
    viz_buffer, _preamble, cfar_guard=10, cfar_train=50, pfa=1e-5
)

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
time_axis = (np.arange(len(viz_buffer)) + buffer_start) / _sample_rate * 1000

# Subplot 1: Received signal power
axes[0].plot(time_axis, np.abs(viz_buffer)**2, 'gray', alpha=0.6, linewidth=0.5)
axes[0].set_ylabel('Power', fontsize=11)
axes[0].set_title('Received IQ Signal Power', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
for _i, ts in enumerate(true_starts):
    if buffer_start <= ts <= buffer_end:
        t_ms = ts / _sample_rate * 1000
        axes[0].axvline(t_ms, color='green', linestyle='--', alpha=0.7, linewidth=1.5,
                      label='True Packet' if _i == 0 else '')
axes[0].legend(fontsize=10)

# Subplot 2: Correlation output
axes[1].plot(time_axis, corr_viz, 'blue', linewidth=1, label='Correlation')
axes[1].plot(time_axis, thresh_viz, 'red', linestyle='--', linewidth=1.5, label='CFAR Threshold')
axes[1].set_ylabel('Correlation Power', fontsize=11)
axes[1].set_title('Preamble Correlation with Adaptive CFAR Threshold', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend(fontsize=10)

# Subplot 3: Detections
detection_mask = np.zeros(len(viz_buffer))
for det in detections_viz:
    detection_mask[det] = corr_viz[det]

axes[2].plot(time_axis, corr_viz, 'blue', alpha=0.4, linewidth=0.8)
axes[2].scatter(time_axis[detection_mask > 0], detection_mask[detection_mask > 0],
               color='lime', edgecolors='black', s=100, zorder=5, label='Detected Packets')
axes[2].set_xlabel('Time (ms)', fontsize=11)
axes[2].set_ylabel('Correlation Power', fontsize=11)
axes[2].set_title('Detected Packet Locations', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].legend(fontsize=10)

plt.tight_layout()
#if os.path.isdir('../_images'):
#    plt.savefig('../_images/detection_realtime.png', bbox_inches='tight', dpi=150)
#    print("Figure saved to ../_images/detection_realtime.png")
plt.show()
