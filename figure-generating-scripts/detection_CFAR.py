import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

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
sps = 4
preamble_syms = np.array([1+1j, 1+1j, -1-1j, -1-1j, 1-1j, -1+1j]) / np.sqrt(2)
tx_signal, ref_preamble = generate_qpsk_packets(5, sps, preamble_syms)

# 2. Channel: Time-Varying Noise Floor
t = np.arange(len(tx_signal))
noise_env = 0.05 + 0.3 * np.sin(2 * np.pi * 0.0003 * t)**2
noise = (np.random.randn(len(tx_signal)) + 1j*np.random.randn(len(tx_signal))) * noise_env
rx_signal = tx_signal + noise

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
detections = np.where(corr_power > cfar_thresholds)[0]
# Filter detections to only include those where threshold is non-zero (avoid edges)
detections = detections[cfar_thresholds[detections] > 0]

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
if len(detections) > 0:
    plt.scatter(detections, corr_power[detections], color='lime', edgecolors='black', 
                label='Detections (Preamble Found)', zorder=5)

plt.title("Preamble Correlator Output with Adaptive CFAR Threshold")
plt.xlabel("Sample Index")
plt.ylabel("Correlation Power")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../_images/detection_cfar.svg', bbox_inches='tight')
plt.show()
