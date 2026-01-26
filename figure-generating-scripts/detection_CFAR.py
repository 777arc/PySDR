import numpy as np
import matplotlib.pyplot as plt

def generate_qpsk_packets(num_packets, samples_per_sym, preamble):
    """Generates repeating QPSK packets with a fixed preamble."""
    qpsk_map = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
    packet_data_len = 100
    full_signal = []
    
    for _ in range(num_packets):
        data = qpsk_map[np.random.randint(0, 4, packet_data_len)]
        packet = np.concatenate([preamble, data])
        # Upsample (simple rectangular pulse for visualization)
        upsampled = np.repeat(packet, samples_per_sym)
        full_signal.extend(upsampled)
        # Add gap between packets
        full_signal.extend(np.zeros(50 * samples_per_sym))
    
    return np.array(full_signal)

# 1. Setup Parameters
sps = 4
preamble_symbols = np.array([1+1j, 1+1j, -1-1j, -1-1j]) / np.sqrt(2) # Fixed preamble
signal = generate_qpsk_packets(5, sps, preamble_symbols)

# 2. Add Time-Varying Noise Floor
t = np.arange(len(signal))
noise_envelope = 0.1 + 0.4 * np.sin(2 * np.pi * 0.0005 * t)**2 # Noise floor varies
noise = (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal))) * noise_envelope
rx_signal = signal + noise

# 3. CFAR Detection (Cell Averaging)
def ca_cfar(data, num_train, num_guard, pfa):
    """Performs CA-CFAR detection on input data."""
    # Process power of signal (Square-law detector)
    x = np.abs(data)**2
    num_cells = len(x)
    thresholds = np.zeros(num_cells)
    detections = []
    
    # Alpha factor for CA-CFAR
    alpha = num_train * (pfa**(-1/num_train) - 1)
    
    half_train = num_train // 2
    half_guard = num_guard // 2
    offset = half_train + half_guard
    
    for i in range(offset, num_cells - offset):
        # Sliding window for noise estimation
        training_cells = np.concatenate([
            x[i-offset : i-half_guard], 
            x[i+half_guard+1 : i+offset+1]
        ])
        noise_floor_est = np.mean(training_cells)
        thresholds[i] = alpha * noise_floor_est
        
        if x[i] > thresholds[i]:
            detections.append(i)
            
    return thresholds, np.array(detections)

# Run Detector
th, det_indices = ca_cfar(rx_signal, num_train=40, num_guard=10, pfa=1e-4)

# 4. Plotting Results
plt.figure(figsize=(14, 6))
plt.plot(np.abs(rx_signal)**2, label='Received Signal Power', alpha=0.6, color='gray')
plt.plot(th, label='CFAR Adaptive Threshold', color='red', linewidth=1.5)
if len(det_indices) > 0:
    plt.scatter(det_indices, np.abs(rx_signal[det_indices])**2, color='lime', label='Detections', zorder=5, s=20)

plt.title("CFAR Detection with Time-Varying Noise Floor")
plt.xlabel("Samples")
plt.ylabel("Power")
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('../_images/detection_cfar.svg', bbox_inches='tight')
plt.show()
