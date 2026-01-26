import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter

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
mc_trials = 1000      # Monte Carlo trials per SNR

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
thresholds = np.linspace(0, 2, 50)

for snr_db in snr_db_list:
    pd_curve = []
    pfa_curve = []
    
    # Noise power calculation
    snr_linear = 10**(snr_db / 10)
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

        print(f"thres={thresh} | SNR={snr_db}")
    
    plt.subplot(1, 2, 1)
    plt.plot(pfa_curve, pd_curve, label=f'SNR={snr_db}dB')

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

for snr_db in snr_range:
    snr_linear = 10**(snr_db / 10)
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
plt.savefig('../_images/detection_pd_vs_snr.svg', bbox_inches='tight')
plt.show()
