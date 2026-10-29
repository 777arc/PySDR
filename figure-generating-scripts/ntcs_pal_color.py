import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp_signal

ntsc_filepath = '/mnt/c/Users/marclichtman/Downloads/never_the_same_color.sigmf-data'  # cf32, 8M sample rate
num_samples = 10000000
sig = np.fromfile(ntsc_filepath, dtype=np.complex64, count=num_samples)
fs = 8e6

print("=== NTSC Color Decoder ===")
print(f"Sample rate: {fs/1e6} MHz")

# NTSC parameters
color_carrier_freq = 3.579545e6  # color subcarrier relative to luma carrier
samples_per_line = 508  # samples per line after resampling
total_lines = 525
frame_rate = 29.97
line_rate = frame_rate * total_lines  # ~15734.25 Hz

# Find luma carrier (peak in lower half of spectrum)
power_spectrum = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(sig))) ** 2)
freq_axis = np.linspace(-fs / 2, fs / 2, len(power_spectrum))
luma_carrier_freq = freq_axis[np.argmax(power_spectrum[0:len(power_spectrum) // 2])]
print(f"Luma carrier at: {luma_carrier_freq / 1e6:.3f} MHz")

# Center on luma carrier
sig = sig * np.exp(-2j * np.pi * luma_carrier_freq * np.arange(len(sig)) / fs)

# Find line sync positions using falling edges of |sig|, filtered for line spacing
sync_threshold = 0.65
neg_edges = np.where(np.diff((np.abs(sig) > sync_threshold).astype(int)) == -1)[0]
sync_positions = [neg_edges[0]]
for k in range(1, len(neg_edges)):
    if neg_edges[k] - sync_positions[-1] > 400:
        sync_positions.append(neg_edges[k])
sync_positions = np.array(sync_positions)
# Skip first edge if it has abnormal gap (partial line at start)
if len(sync_positions) > 1 and (sync_positions[1] - sync_positions[0]) > 600:
    sync_positions = sync_positions[1:]
print(f"Lines found: {len(sync_positions)}")
active_video_offset = 70  # samples from sync edge to start of active video

# Extract chroma: shift by color subcarrier, lowpass
chroma_sig = sig * np.exp(-2j * np.pi * color_carrier_freq * np.arange(len(sig)) / fs)
chroma_sig = np.convolve(chroma_sig, sp_signal.firwin(301, 1e6, fs=fs), 'same')

# Extract luma: lowpass and take magnitude (AM demod)
luma_sig = np.convolve(sig, sp_signal.firwin(301, 3e6, fs=fs), 'same')
luma_sig = np.abs(luma_sig)

# Extract color burst info from each line
burst_delay = 6
burst_length = 22
color_freq_offsets = []
for j in sync_positions:
    burst_segment = chroma_sig[j + burst_delay:j + burst_delay + burst_length]
    if len(burst_segment) < burst_length:
        color_freq_offsets.append(0.0)
        continue
    burst_spectrum = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(burst_segment, 1024))) ** 2)
    burst_freq_axis = np.linspace(-fs / 2, fs / 2, 1024)
    color_freq_offsets.append(burst_freq_axis[np.argmax(burst_spectrum)])

# Find burst phases and filter out bad bursts
burst_angles = []
good_sync_positions = []
good_freq_offsets = []
for k in range(len(sync_positions)):
    burst_chunk = chroma_sig[sync_positions[k] + 12:sync_positions[k] + burst_length]
    if len(burst_chunk) == 0:
        continue
    burst_chunk = burst_chunk * np.exp(-2j * np.pi * color_freq_offsets[k] * np.arange(len(burst_chunk)) / fs)
    if np.max(np.abs(burst_chunk)) > 0.02:
        if np.var(burst_chunk) < 1e-4:
            burst_angles.append(np.mean(np.angle(burst_chunk)))
            good_sync_positions.append(sync_positions[k])
            good_freq_offsets.append(color_freq_offsets[k])
sync_positions = good_sync_positions
color_freq_offsets = good_freq_offsets

# Determine line parity from burst phase (robust to filtered-out lines)
line_parities = []
for k in range(len(burst_angles)):
    angle_deg = (burst_angles[k] * 180 / np.pi) % 360
    dist_225 = min(abs(angle_deg - 225), 360 - abs(angle_deg - 225))
    dist_135 = min(abs(angle_deg - 135), 360 - abs(angle_deg - 135))
    line_parities.append(0 if dist_225 < dist_135 else 1)

# Compute per-line phase correction based on detected parity
phase_corrections = []
for k in range(len(burst_angles)):
    if line_parities[k] == 0:
        adjusted = burst_angles[k] - 225 / 180 * np.pi
    else:
        adjusted = burst_angles[k] - 135 / 180 * np.pi
    phase_corrections.append(adjusted % (2 * np.pi))

# Decode each line into RGB frame
active_start = 32
active_end = 6
rgb_frame = np.zeros((total_lines // 2, samples_per_line, 3))
line_idx = 0

for ln in range(len(sync_positions)):
    pos = sync_positions[ln]
    if pos + samples_per_line + active_end >= len(luma_sig):
        break

    ref_level = luma_sig[pos + 20]
    if ref_level < 0.01:
        ref_level = 1.0
    luma = np.array(luma_sig[pos + active_start:pos + samples_per_line + active_end])
    luma = luma / ref_level

    chroma_segment = chroma_sig[pos + active_start:pos + samples_per_line + active_end]
    chroma_segment = chroma_segment * np.exp(-2j * np.pi * color_freq_offsets[ln] * np.arange(len(chroma_segment)) / fs)
    chroma_segment = chroma_segment * np.exp(1j * phase_corrections[ln])
    i_signal = chroma_segment.real
    q_signal = chroma_segment.imag
    if line_parities[ln] == 0:
        q_signal *= -1

    i_signal *= 4.5
    q_signal *= 5.5

    blue = luma + 2.029 * i_signal
    red = luma + 1.14 * q_signal
    green = luma - 0.396 * i_signal - 0.581 * q_signal

    if line_idx < total_lines // 2:
        count = min(len(luma), samples_per_line)
        rgb_frame[line_idx, 0:count, 0] = 1 - red[:count]
        rgb_frame[line_idx, 0:count, 1] = 1 - green[:count]
        rgb_frame[line_idx, 0:count, 2] = 1 - blue[:count]

    line_idx += 1
    if line_idx == total_lines - 18:
        break  # one frame decoded

rgb_frame = np.clip(rgb_frame, 0, 1)
print(f"Decoded {line_idx} lines")
print(f"RGB max: R={rgb_frame[:,:,0].max():.2f} G={rgb_frame[:,:,1].max():.2f} B={rgb_frame[:,:,2].max():.2f}")

plt.figure(figsize=(10, 7))
plt.imshow(rgb_frame, aspect=0.6)
plt.title('Decoded NTSC Color Frame')
plt.axis('off')
plt.tight_layout()
#plt.savefig('/home/marc/PySDR/figure-generating-scripts/ntsc_color_frame.png', dpi=150)
print("Saved to ntsc_color_frame.png")
plt.show()
