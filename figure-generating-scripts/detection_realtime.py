"""
Generate visualization for real-time packet detection section.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

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
N_zc, u = 63, 5
t = np.arange(N_zc)
preamble = np.exp(-1j * np.pi * u * t * (t + 1) / N_zc)

sample_rate = 1e6
packet_length = 500
snr_db = -5

signal, true_starts = generate_packet_stream(
    preamble, packet_length, num_packets=5, 
    sample_rate=sample_rate, snr_db=snr_db
)

# Process one buffer for visualization
buffer_start = max(0, true_starts[0] - 5000)
buffer_end = min(len(signal), true_starts[2] + 10000)
viz_buffer = signal[buffer_start:buffer_end]

detections_viz, corr_viz, thresh_viz = detect_packets(
    viz_buffer, preamble, cfar_guard=10, cfar_train=50, pfa=1e-5
)

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
time_axis = (np.arange(len(viz_buffer)) + buffer_start) / sample_rate * 1000

# Subplot 1: Received signal power
axes[0].plot(time_axis, np.abs(viz_buffer)**2, 'gray', alpha=0.6, linewidth=0.5)
axes[0].set_ylabel('Power', fontsize=11)
axes[0].set_title('Received IQ Signal Power', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
for i, ts in enumerate(true_starts):
    if buffer_start <= ts <= buffer_end:
        t_ms = ts / sample_rate * 1000
        axes[0].axvline(t_ms, color='green', linestyle='--', alpha=0.7, linewidth=1.5,
                      label='True Packet' if i == 0 else '')
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
plt.savefig('../_images/detection_realtime.svg', bbox_inches='tight', dpi=150)
print("Figure saved to ../_images/detection_realtime.svg")
plt.show()