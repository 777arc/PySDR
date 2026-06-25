import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import combinations
from scipy.signal import firwin, lfilter

np.random.seed(0)

sample_rate = 50e6
c = 3e8  # speed of light [m/s]
rx_positions = np.array([
    [0.0,   0.0],    # Rx0
    [600.0, 100.0],    # Rx1
    [150.0, 500.0],  # Rx2
])
num_rx = rx_positions.shape[0]
tx_position = np.array([150.0, 350.0])
pairs = list(combinations(range(num_rx), 2)) # For 3 receivers it's (Rx0,Rx1), (Rx0,Rx2), (Rx1,Rx2) -> 3 pairs

# For the tx signal itself it's arbitrary, although bandwidth matters, we'll transmit band-limited noise
N = 10000 # samples to transmit
bandwidth = 20e6
taps = firwin(numtaps=129, cutoff=bandwidth / 2, fs=sample_rate)
tx_signal = lfilter(taps, 1.0, np.random.randn(N) + 1j * np.random.randn(N))

# Simulate what each receiver records
true_distances = np.linalg.norm(rx_positions - tx_position, axis=1)
true_delays = true_distances / c
unknown_tx_time = 1.234e-5   # seconds. arbitray, unknown to receivers and we wont use it in any TDOA calcs

# Figure out how many samples we have to simulate
total_delay_samples = (unknown_tx_time + true_delays.max()) * sample_rate
buffer_len = N + int(np.ceil(total_delay_samples)) + 10

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
    rx[tau_integer_samps:tau_integer_samps+N] = tx_signal
    frac_delay_i = frac_delay_filter(tau_frac_samps)
    rx = np.convolve(rx, frac_delay_i, "same")

    # Each receiver adds its own thermal noise
    noise_power = 0.05
    noise = np.sqrt(noise_power / 2) * (np.random.randn(buffer_len) + 1j * np.random.randn(buffer_len))
    rx_signals[i] = rx + noise

# =============================================================================
# METHOD 1: integer-only TDOA via a plain time-domain cross-correlation
# =============================================================================
# 4. Estimate the time differences.  np.correlate just slides Rx_b past Rx_a and
# reports how well they overlap at every integer shift -- about as simple as DSP
# gets.  Two consequences: it resolves the lag only to the nearest WHOLE sample
# (6 m of range at 50 MHz), and being a direct O(N^2) correlation it is slow,
# which is exactly why N is modest and why Method 2 later switches to the FFT.
range_diff_int = np.zeros(len(pairs))
for k, (a, b) in enumerate(pairs):
    xcorr = np.correlate(rx_signals[b], rx_signals[a], mode='full')
    # 'full' puts zero lag at index buffer_len-1; subtract it to get the lag.
    peak_lag = np.argmax(np.abs(xcorr)) - (buffer_len - 1)  # +ve => Rx_b farther
    range_diff_int[k] = (peak_lag / sample_rate) * c

print("METHOD 1 (integer-only, time domain)")
print(" Pair  |  true range diff [m] | measured range diff [m]")
for k, (a, b) in enumerate(pairs):
    true_rd = true_distances[b] - true_distances[a]
    print(f"Rx{b}-Rx{a} |     {true_rd:9.1f}        |    {range_diff_int[k]:9.1f}")

# 5. Solve for the transmitter with a grid search.  Each measurement says: "for
# the true Tx location, distance to Rx_b minus distance to Rx_a should equal
# range_diff[k]."  Rather than solving the hyperbola equations algebraically, we
# brute-force it: score every candidate cell by how well it matches all the
# measured range differences, and take the best.  This naturally shows the cost
# surface and is dead simple to read.
# Precompute the distance from each receiver to every grid point, since the solver and the hyperbola overlays both need it.
grid_x = np.linspace(-200, 800, 400)
grid_y = np.linspace(-200, 800, 400)
GX, GY = np.meshgrid(grid_x, grid_y)
rx_dist = []
for i in range(num_rx):
    rx_dist.append(np.sqrt((GX - rx_positions[i, 0])**2 + (GY - rx_positions[i, 1])**2))
error_surface_int = np.zeros_like(GX)
for k, (a, b) in enumerate(pairs):
    predicted = rx_dist[b] - rx_dist[a]
    error_surface_int += (predicted - range_diff_int[k])**2
best = np.unravel_index(np.argmin(error_surface_int), error_surface_int.shape)
est_int = np.array([GX[best], GY[best]])
print(f"True Tx: ({tx_position[0]:.0f}, {tx_position[1]:.0f})   "
      f"Estimate: ({est_int[0]:.0f}, {est_int[1]:.0f})   "
      f"Error: {np.linalg.norm(est_int - tx_position):.1f} m\n")

# 6. FIGURE 1: the integer-only result.
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle('Method 1: integer-only TDOA (time-domain cross-correlation)')

# Left: cost surface + hyperbolas.  The hyperbolas all cross at the transmitter,
# and the cost surface is darkest there.
im = ax1.pcolormesh(GX, GY, np.log10(error_surface_int + 1), shading='auto', cmap='viridis')
fig1.colorbar(im, ax=ax1, label='log10 of total squared range-difference error')
hyperbola_handles = []
pair_colors = ['white', 'orange', 'magenta']
for k, (a, b) in enumerate(pairs):
    ax1.contour(GX, GY, (rx_dist[b] - rx_dist[a]) - range_diff_int[k], levels=[0],
                colors=pair_colors[k], linewidths=1.5, linestyles='--')
    hyperbola_handles.append(Line2D([0], [0], color=pair_colors[k],
                                    linestyle='--', label=f'Rx{a}-Rx{b}'))
ax1.scatter(rx_positions[:, 0], rx_positions[:, 1], c='cyan', marker='^',
            s=120, edgecolors='k', label='Receivers', zorder=5)
for i in range(num_rx):
    ax1.annotate(f'Rx{i}', rx_positions[i], textcoords='offset points',
                 xytext=(8, 8), color='white', fontweight='bold', zorder=6)
ax1.scatter(*tx_position, c='red', marker='*', s=300, edgecolors='k',
            label='True Tx', zorder=5)
ax1.scatter(*est_int, c='lime', marker='x', s=150, linewidths=3,
            label='Estimated Tx', zorder=5)
ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
ax1.set_title('TDOA cost surface and hyperbolas')
ax1.legend(handles=ax1.get_legend_handles_labels()[0] + hyperbola_handles, loc='upper right')
ax1.set_aspect('equal')

# Right: the raw time-domain cross-correlation of one pair, with the integer
# (whole-sample) peak we picked.  The peak sits on a sample; we can't do better.
a, b = pairs[1]                        # the (Rx0, Rx2) pair
xcorr = np.abs(np.correlate(rx_signals[b], rx_signals[a], mode='full'))
lags = np.arange(xcorr.size) - (buffer_len - 1)
peak = lags[np.argmax(xcorr)]
ax2.plot(lags, xcorr, 'o-', markersize=5, label='correlation samples')
ax2.axvline(peak, color='red', linestyle='--', label=f'integer peak = {peak} samples')
ax2.set_xlim(peak - 6, peak + 6)
ax2.set_xlabel('lag [samples]'); ax2.set_ylabel('|cross-correlation|')
ax2.set_title(f'Cross-correlation of Rx{b} vs Rx{a}')
ax2.legend()
fig1.tight_layout()

# =============================================================================
# METHOD 2: sub-sample TDOA via a zero-padded frequency-domain correlation
# =============================================================================
# 7. Same idea, but in the frequency domain.  The cross-correlation is the IFFT
# of the cross-power spectrum conj(A)*B, which for large signals is far cheaper
# than the direct correlation above.  And we get sub-sample resolution almost for
# free: ZERO-PAD the spectrum before the inverse FFT.  Padding a spectrum is
# exact sinc interpolation in the time domain, so the IFFT lands on a grid U
# times finer, and we just take its argmax.
U = 16                                 # correlation upsampling factor
L = buffer_len
half = (L + 1) // 2                    # number of DC + positive-frequency bins
range_diff = np.zeros(len(pairs))
for k, (a, b) in enumerate(pairs):
    X = np.conj(np.fft.fft(rx_signals[a])) * np.fft.fft(rx_signals[b])
    # Insert zeros in the high-frequency MIDDLE: DC + positive freqs at the
    # front, negative freqs at the back, so it stays a valid FFT layout.
    X_padded = np.zeros(U * L, dtype=complex)
    X_padded[:half] = X[:half]
    X_padded[U * L - (L - half):] = X[half:]
    cc = np.abs(np.fft.ifft(X_padded)) * U
    # Peak index -> signed lag; indices past the midpoint are negative lags.
    peak_idx = np.argmax(cc)
    if peak_idx > U * L // 2:
        peak_idx -= U * L
    peak_lag = peak_idx / U            # sub-sample lag, +ve => Rx_b farther
    range_diff[k] = (peak_lag / sample_rate) * c

print("METHOD 2 (sub-sample, zero-padded FFT)")
print(" Pair  |  true range diff [m] | measured range diff [m]")
for k, (a, b) in enumerate(pairs):
    true_rd = true_distances[b] - true_distances[a]
    print(f"Rx{b}-Rx{a} |     {true_rd:9.1f}        |    {range_diff[k]:9.1f}")

# 8. Solve again, identical grid search but with the refined range differences.
error_surface = np.zeros_like(GX)
for k, (a, b) in enumerate(pairs):
    predicted = rx_dist[b] - rx_dist[a]
    error_surface += (predicted - range_diff[k])**2
best = np.unravel_index(np.argmin(error_surface), error_surface.shape)
est_position = np.array([GX[best], GY[best]])
print(f"True Tx: ({tx_position[0]:.0f}, {tx_position[1]:.0f})   "
      f"Estimate: ({est_position[0]:.0f}, {est_position[1]:.0f})   "
      f"Error: {np.linalg.norm(est_position - tx_position):.1f} m")

# 9. FIGURE 2: the sub-sample result, same layout as Figure 1.
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle('Method 2: sub-sample TDOA (zero-padded FFT cross-correlation)')

im = ax1.pcolormesh(GX, GY, np.log10(error_surface + 1), shading='auto', cmap='viridis')
fig2.colorbar(im, ax=ax1, label='log10 of total squared range-difference error')
hyperbola_handles = []
for k, (a, b) in enumerate(pairs):
    ax1.contour(GX, GY, (rx_dist[b] - rx_dist[a]) - range_diff[k], levels=[0],
                colors=pair_colors[k], linewidths=1.5, linestyles='--')
    hyperbola_handles.append(Line2D([0], [0], color=pair_colors[k],
                                    linestyle='--', label=f'Rx{a}-Rx{b}'))
ax1.scatter(rx_positions[:, 0], rx_positions[:, 1], c='cyan', marker='^',
            s=120, edgecolors='k', label='Receivers', zorder=5)
for i in range(num_rx):
    ax1.annotate(f'Rx{i}', rx_positions[i], textcoords='offset points',
                 xytext=(8, 8), color='white', fontweight='bold', zorder=6)
ax1.scatter(*tx_position, c='red', marker='*', s=300, edgecolors='k',
            label='True Tx', zorder=5)
ax1.scatter(*est_position, c='lime', marker='x', s=150, linewidths=3,
            label='Estimated Tx', zorder=5)
ax1.set_xlabel('x [m]'); ax1.set_ylabel('y [m]')
ax1.set_title('TDOA cost surface and hyperbolas')
ax1.legend(handles=ax1.get_legend_handles_labels()[0] + hyperbola_handles, loc='upper right')
ax1.set_aspect('equal')

# Right: coarse (1 sample/lag) correlation as dots vs the U-times upsampled
# correlation as a smooth curve, so the sub-sample shift is visible.
a, b = pairs[1]                        # the (Rx0, Rx2) pair
X = np.conj(np.fft.fft(rx_signals[a])) * np.fft.fft(rx_signals[b])
cc_coarse = np.abs(np.fft.ifft(X))
X_padded = np.zeros(U * L, dtype=complex)
X_padded[:half] = X[:half]
X_padded[U * L - (L - half):] = X[half:]
cc_fine = np.abs(np.fft.ifft(X_padded)) * U
lags_coarse = np.where(np.arange(L) <= L // 2, np.arange(L), np.arange(L) - L)
lags_fine = np.arange(U * L) / U
lags_fine = np.where(lags_fine <= L / 2, lags_fine, lags_fine - L)
peak = lags_coarse[np.argmax(cc_coarse)]
subsample_peak = lags_fine[np.argmax(cc_fine)]
# The lag axes wrap from + back to - partway through, so sort before plotting,
# otherwise the connecting line jumps across the figure.
order_c = np.argsort(lags_coarse)
order_f = np.argsort(lags_fine)
ax2.plot(lags_coarse[order_c], cc_coarse[order_c], 'o', markersize=6, label='coarse (1 sample/lag)')
ax2.plot(lags_fine[order_f], cc_fine[order_f], '-', label=f'{U}x zero-padded')
ax2.axvline(subsample_peak, color='red', linestyle='--',
            label=f'sub-sample peak = {subsample_peak:.3f}')
ax2.set_xlim(peak - 6, peak + 6)
ax2.set_xlabel('lag [samples]'); ax2.set_ylabel('|cross-correlation|')
ax2.set_title(f'Cross-correlation of Rx{b} vs Rx{a}')
ax2.legend()
fig2.tight_layout()

plt.show()
