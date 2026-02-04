import numpy as np
from scipy.signal import upfirdn
from scipy.special import erfc
import matplotlib.pyplot as plt

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
N_preamble = 64
N_data = 512
sps = 8
rolloff = 0.35
span = 8
Fs = 1e6
symbol_rate = Fs / sps

f_max = 5e3
deg_dB = 1
seg_symb_min = 8
EbN0_dBs = [0, 5, 10]
num_trials = 200

# Generate preamble + data
bits = np.random.randint(0, 2, (N_preamble+N_data)*2)
symbols = qpsk_mod(bits)
preamble = symbols[:N_preamble]
data = symbols[N_preamble:]

# SRRC shaping
pulse = srrc_pulse(rolloff, sps, span)
tx = upfirdn(pulse, np.concatenate([preamble, data]), sps)
tx = tx / np.sqrt(np.mean(np.abs(tx)**2))

# Determine frequency sweep
freqs = determine_freq_spacing(f_max, deg_dB, seg_symb_min, symbol_rate)

# ======================
# Sweep: frequency offset
# ======================
freq_offsets = np.linspace(-f_max, f_max, 15)
results = {snr: [] for snr in EbN0_dBs}

for snr in EbN0_dBs:
    for fo in freq_offsets:
        corr_peaks = []
        for _ in range(num_trials):
            rx = apply_freq_offset(tx, Fs, fo)
            rx = awgn(rx, snr)
            corr = preamble_detector(rx, preamble, Fs, sps, freqs, seg_symb_min, coherent=False)
            corr_peaks.append(np.max(corr))
        results[snr].append(np.mean(corr_peaks))

plt.figure()
for snr in EbN0_dBs:
    plt.plot(freq_offsets/1e3, 20*np.log10(results[snr]/np.max(results[snr])), label=f"SNR={snr}dB")
plt.xlabel("Frequency offset (kHz)")
plt.ylabel("Normalized correlation peak (dB)")
plt.legend()
plt.title("Correlation degradation vs frequency offset")
plt.grid(True)
plt.savefig('../_images/detection_freq_offset.svg', bbox_inches='tight')
plt.show()

# ======================
# Sweep: SNR vs detection probability
# ======================
snr_range = np.linspace(-5, 20, 10)
test_offsets = [0, 2e3, 5e3]
threshold = 0.3  # detection threshold
det_prob = {fo: [] for fo in test_offsets}

for fo in test_offsets:
    for snr in snr_range:
        detections = 0
        for _ in range(num_trials):
            rx = apply_freq_offset(tx, Fs, fo)
            rx = awgn(rx, snr)
            corr = preamble_detector(rx, preamble, Fs, sps, freqs, seg_symb_min, coherent=False)
            if np.max(corr)/np.max(corr) > threshold:
                detections += 1
        det_prob[fo].append(detections/num_trials)
    plt.plot(snr_range, det_prob[fo], label=f"Offset={fo/1e3:.1f} kHz")

plt.xlabel("SNR (dB)")
plt.ylabel("Probability of detection")
plt.title("Detection probability vs SNR for various frequency offsets")
plt.legend()
plt.grid(True)
plt.savefig('../_images/detection_freq_offset2.svg', bbox_inches='tight')
plt.show()
