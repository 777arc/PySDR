import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import combinations
from scipy.signal import firwin, lfilter

sample_rate = 50e6
c = 3e8  # speed of light [m/s]
snr_db = 10  # SNR of the received signal at each receiver [dB]
tx_len_samples = 1000 # samples to transmit
rx_positions = np.array([
    [65,  229],   # Rx0
    [676, 123],  # Rx1
    [153, 543],  # Rx2
])
num_rx = rx_positions.shape[0]
tx_position = np.array([153, 355])
pairs = list(combinations(range(num_rx), 2)) # For 3 receivers it's (Rx0,Rx1), (Rx0,Rx2), (Rx1,Rx2) -> 3 pairs

# For the tx signal itself it's arbitrary, although bandwidth matters, we'll transmit band-limited noise
bandwidth = 20e6
taps = firwin(numtaps=129, cutoff=bandwidth / 2, fs=sample_rate)
tx_signal = lfilter(taps, 1.0, np.random.randn(tx_len_samples) + 1j * np.random.randn(tx_len_samples))

# Simulate what each receiver records
true_distances = np.linalg.norm(rx_positions - tx_position, axis=1)
true_delays = true_distances / c
unknown_tx_time = 1.234e-5   # seconds. arbitray, unknown to receivers and we wont use it in any TDOA calcs

# Calc the actual TDOAs to act as ground truth
for k, (a, b) in enumerate(pairs):
    true_rd = true_distances[b] - true_distances[a]

# Figure out how many samples we have to simulate
total_delay_samples = (unknown_tx_time + true_delays.max()) * sample_rate
buffer_len = tx_len_samples + int(np.ceil(total_delay_samples)) + 10

# Taken from Synchronization chapter
def frac_delay_filter(delay): # delay is in samples, but it can (and will be) not an integer
    N = 21 # number of taps, keep this odd
    n = np.arange(-(N-1)//2, N//2+1) # -10,-9,...,0,...,9,10
    h = np.sinc(n - delay) # calc filter taps
    h *= np.hamming(N) # window the filter to make sure it decays to 0 on both sides
    h /= np.sum(h) # normalize to get unity gain, we don't want to change the amplitude/power
    return h

# Simulate the delayed signal being received by each sensor
rx_signals = np.zeros((num_rx, buffer_len), dtype=complex)
for i in range(num_rx):
    tau = unknown_tx_time + true_delays[i] # absolute delay at this Rx, in seconds
    tau_samples = tau * sample_rate
    tau_integer_samps = int(np.round(tau_samples))
    tau_frac_samps = tau_samples - tau_integer_samps
    rx = np.zeros(buffer_len, dtype=complex)
    rx[tau_integer_samps:tau_integer_samps+tx_len_samples] = tx_signal
    frac_delay_i = frac_delay_filter(tau_frac_samps)
    rx = np.convolve(rx, frac_delay_i, "same")

    # Each receiver adds its own thermal noise, scaled to hit the SNR set at the top
    signal_power = np.mean(np.abs(tx_signal)**2)
    noise_power = signal_power / 10**(snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (np.random.randn(buffer_len) + 1j * np.random.randn(buffer_len))
    rx_signals[i] = rx + noise

# Estimate the TDOAs using a normal cross-correlation
range_diff = np.zeros(len(pairs)) # meters
for k, (a, b) in enumerate(pairs):
    xcorr = np.correlate(rx_signals[b], rx_signals[a], mode='full') 
    peak_lag = np.argmax(np.abs(xcorr)) - (buffer_len - 1) # 'full' puts zero lag at index buffer_len-1
    range_diff[k] = (peak_lag / sample_rate) * c # meters

# FIGURE 1: the integer-only result.
# Precompute the distance from each receiver to every grid point, this will get used in the contour plot
grid_x = np.linspace(-200, 800, 400)
grid_y = np.linspace(-200, 800, 400)
GX, GY = np.meshgrid(grid_x, grid_y)
rx_dist = []
for i in range(num_rx):
    rx_dist.append(np.sqrt((GX - rx_positions[i, 0])**2 + (GY - rx_positions[i, 1])**2))
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# Left: the hyperbolas, which all cross at the transmitter.
hyperbola_handles = []
pair_colors = ['tab:blue', 'tab:orange', 'tab:green']
for k, (a, b) in enumerate(pairs):
    # the next line is what calculates the hyperbola, note levels=[0] means we're making a contour map but only one level, specifically the level where the difference is zero
    ax1.contour(GX, GY, (rx_dist[b] - rx_dist[a]) - range_diff[k], levels=[0], colors=pair_colors[k], linestyles='--')
    hyperbola_handles.append(Line2D([0], [0], color=pair_colors[k], linestyle='--', label=f'Rx{a}-Rx{b}'))
ax1.scatter(rx_positions[:, 0], rx_positions[:, 1], c='tab:blue', marker='^', s=120, edgecolors='k', label='Receivers', zorder=5)
for i in range(num_rx):
    ax1.annotate(f'Rx{i}', rx_positions[i], textcoords='offset points', xytext=(8, 8), fontweight='bold', zorder=6)
ax1.scatter(*tx_position, c='red', marker='*', s=300, edgecolors='k', label='True Tx', zorder=5)
ax1.set_xlim(grid_x[0], grid_x[-1]); ax1.set_ylim(grid_y[0], grid_y[-1])
ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
ax1.set_title('TDOA hyperbolas')
ax1.legend(handles=ax1.get_legend_handles_labels()[0] + hyperbola_handles, loc='upper right')
ax1.set_aspect('equal')
# Cross-correlation of one pair, integer only
a, b = pairs[1] # the (Rx0, Rx2) pair
xcorr = np.abs(np.correlate(rx_signals[b], rx_signals[a], mode='full'))
lags = np.arange(xcorr.size) - (buffer_len - 1)
peak = lags[np.argmax(xcorr)]
ax2.plot(lags, xcorr, 'o-', markersize=5, label='correlation samples')
ax2.axvline(peak, color='red', linestyle='--', label=f'integer peak = {peak} samples')
ax2.set_xlim(peak - 6, peak + 6)
ax2.set_xlabel('lag [samples]'); ax2.set_ylabel('|cross-correlation|')
ax2.set_title(f'Cross-correlation of Rx{b} vs Rx{a}')
ax2.legend()
ax2.grid()
fig1.savefig('../_images/tdoa_python_integer.svg', bbox_inches='tight')
fig1.tight_layout()

# Subsample TDOA calc using a freq domain cross-correlation that was padded as a way to interpolate
U = 16 # correlation upsampling factor
half = (buffer_len + 1) // 2 # number of DC + positive-frequency bins
range_diff = np.zeros(len(pairs)) # meters
for k, (a, b) in enumerate(pairs):
    # Cross-correlation in the frequency domain
    X = np.conj(np.fft.fft(rx_signals[a])) * np.fft.fft(rx_signals[b])

    # Insert zeros in the high-frequency MIDDLE: DC + positive freqs at the front, negative freqs at the back, so it stays a valid FFT layout.
    X_padded = np.zeros(U * buffer_len, dtype=complex) 
    X_padded[:half] = X[:half]
    X_padded[U * buffer_len - (buffer_len - half):] = X[half:]

    # Now IFFT to finish the crosscorrelation
    xcorr = np.abs(np.fft.ifft(X_padded)) * U

    # Peak index -> signed lag; indices past the midpoint are negative lags
    peak_idx = np.argmax(xcorr)
    if peak_idx > U * buffer_len // 2:
        peak_idx -= U * buffer_len
    peak_lag = peak_idx / U # sub-sample lag, +ve => Rx_b farther
    range_diff[k] = (peak_lag / sample_rate) * c # meters

print("METHOD 2 (sub-sample, zero-padded FFT)")
print(" Pair  |  true range diff [m] | measured range diff [m]")
for k, (a, b) in enumerate(pairs):
    true_rd = true_distances[b] - true_distances[a]
    print(f"Rx{b}-Rx{a} |     {true_rd:9.1f}        |    {range_diff[k]:9.1f}")

# 8. FIGURE 2: the sub-sample result, same layout as Figure 1.
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: the hyperbolas from the refined range differences.
hyperbola_handles = []
for k, (a, b) in enumerate(pairs):
    ax1.contour(GX, GY, (rx_dist[b] - rx_dist[a]) - range_diff[k], levels=[0],
                colors=pair_colors[k], linewidths=1.5, linestyles='--')
    hyperbola_handles.append(Line2D([0], [0], color=pair_colors[k],
                                    linestyle='--', label=f'Rx{a}-Rx{b}'))
ax1.scatter(rx_positions[:, 0], rx_positions[:, 1], c='tab:blue', marker='^',
            s=120, edgecolors='k', label='Receivers', zorder=5)
for i in range(num_rx):
    ax1.annotate(f'Rx{i}', rx_positions[i], textcoords='offset points',
                 xytext=(8, 8), fontweight='bold', zorder=6)
ax1.scatter(*tx_position, c='red', marker='*', s=300, edgecolors='k',
            label='True Tx', zorder=5)
ax1.set_xlim(grid_x[0], grid_x[-1]); ax1.set_ylim(grid_y[0], grid_y[-1])
ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
ax1.set_title('TDOA hyperbolas')
ax1.legend(handles=ax1.get_legend_handles_labels()[0] + hyperbola_handles, loc='upper right')
ax1.set_aspect('equal')

# Right: coarse (1 sample/lag) correlation as dots vs the U-times upsampled
# correlation as a smooth curve, so the sub-sample shift is visible.
a, b = pairs[1]                        # the (Rx0, Rx2) pair
X = np.conj(np.fft.fft(rx_signals[a])) * np.fft.fft(rx_signals[b])
cc_coarse = np.abs(np.fft.ifft(X))
X_padded = np.zeros(U * buffer_len, dtype=complex)
X_padded[:half] = X[:half]
X_padded[U * buffer_len - (buffer_len - half):] = X[half:]
cc_fine = np.abs(np.fft.ifft(X_padded)) * U
lags_coarse = np.where(np.arange(buffer_len) <= buffer_len // 2, np.arange(buffer_len), np.arange(buffer_len) - buffer_len)
lags_fine = np.arange(U * buffer_len) / U
lags_fine = np.where(lags_fine <= buffer_len / 2, lags_fine, lags_fine - buffer_len)
peak = lags_coarse[np.argmax(cc_coarse)]
subsample_peak = lags_fine[np.argmax(cc_fine)]
# The lag axes wrap from + back to - partway through, so sort before plotting,
# otherwise the connecting line jumps across the figure.
order_c = np.argsort(lags_coarse)
order_f = np.argsort(lags_fine)
ax2.plot(lags_coarse[order_c], cc_coarse[order_c], 'o', markersize=6, label='coarse (1 sample/lag)')
ax2.plot(lags_fine[order_f], cc_fine[order_f], '.-', label=f'{U}x interpolation')
ax2.axvline(subsample_peak, color='red', linestyle='--',
            label=f'sub-sample peak = {subsample_peak:.3f}')
ax2.set_xlim(peak - 6, peak + 6)
ax2.set_xlabel('lag [samples]'); ax2.set_ylabel('|cross-correlation|')
ax2.set_title(f'Cross-correlation of Rx{b} vs Rx{a}')
ax2.legend()
ax2.grid()
fig2.tight_layout()
fig2.savefig('../_images/tdoa_python_subsample.svg', bbox_inches='tight')

plt.show()
