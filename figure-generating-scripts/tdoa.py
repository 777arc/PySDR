# %%
import os
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

# %%
rng = np.random.default_rng(0) # seeded so every run (including the web notebook) gives the same results

# For the tx signal itself it's arbitrary, although bandwidth matters, we'll transmit band-limited noise
bandwidth = 20e6
taps = firwin(numtaps=129, cutoff=bandwidth / 2, fs=sample_rate)
tx_signal = lfilter(taps, 1.0, rng.standard_normal(tx_len_samples) + 1j * rng.standard_normal(tx_len_samples))

# Simulate what each receiver records
true_distances = np.linalg.norm(rx_positions - tx_position, axis=1)
true_delays = true_distances / c
unknown_tx_time = 1.234e-5   # seconds. arbitray, unknown to receivers and we wont use it in any TDOA calcs

# Calc the actual TDOAs to act as ground truth
for _k, (_a, _b) in enumerate(pairs):
    _true_rd = true_distances[_b] - true_distances[_a]

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
for _i in range(num_rx):
    tau = unknown_tx_time + true_delays[_i] # absolute delay at this Rx, in seconds
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
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(buffer_len) + 1j * rng.standard_normal(buffer_len))
    rx_signals[_i] = rx + noise

# %%
# Estimate the TDOAs using a normal cross-correlation
range_diff = np.zeros(len(pairs)) # meters
for _k, (_a, _b) in enumerate(pairs):
    _xcorr = np.correlate(rx_signals[_b], rx_signals[_a], mode='full') 
    _peak_lag = np.argmax(np.abs(_xcorr)) - (buffer_len - 1) # 'full' puts zero lag at index buffer_len-1
    range_diff[_k] = (_peak_lag / sample_rate) * c # meters

# %%
# FIGURE 1: the integer-only result.
# Precompute the distance from each receiver to every grid point, this will get used in the contour plot
grid_x = np.linspace(-200, 800, 400)
grid_y = np.linspace(-200, 800, 400)
GX, GY = np.meshgrid(grid_x, grid_y)
rx_dist = []
for _i in range(num_rx):
    rx_dist.append(np.sqrt((GX - rx_positions[_i, 0])**2 + (GY - rx_positions[_i, 1])**2))
pair_colors = ['tab:blue', 'tab:orange', 'tab:green']

# Draws the hyperbolas, which all cross at the transmitter, along with the receivers and true Tx position. Used by all 3 figures
def plot_hyperbolas(ax, range_diffs, rx_color='tab:blue', label_color=None):
    hyperbola_handles = []
    for k, (a, b) in enumerate(pairs):
        # the next line is what calculates the hyperbola, note levels=[0] means we're making a contour map but only one level, specifically the level where the difference is zero
        ax.contour(GX, GY, (rx_dist[b] - rx_dist[a]) - range_diffs[k], levels=[0], colors=pair_colors[k], linewidths=1.5, linestyles='--')
        hyperbola_handles.append(Line2D([0], [0], color=pair_colors[k], linestyle='--', label=f'Rx{a}-Rx{b}'))
    ax.scatter(rx_positions[:, 0], rx_positions[:, 1], c=rx_color, marker='^', s=120, edgecolors='k', label='Receivers', zorder=5)
    for i in range(num_rx):
        ax.annotate(f'Rx{i}', rx_positions[i], textcoords='offset points', xytext=(8, 8), color=label_color, fontweight='bold', zorder=6)
    ax.scatter(*tx_position, c='red', marker='*', s=300, edgecolors='k', label='True Tx', zorder=5)
    ax.set_xlim(grid_x[0], grid_x[-1]); ax.set_ylim(grid_y[0], grid_y[-1])
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.legend(handles=ax.get_legend_handles_labels()[0] + hyperbola_handles, loc='upper right')
    ax.set_aspect('equal')

fig1, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 6))
plot_hyperbolas(_ax1, range_diff)
_ax1.set_title('TDOA hyperbolas')
# Cross-correlation of one pair, integer only
_a, _b = pairs[1] # the (Rx0, Rx2) pair
_xcorr = np.abs(np.correlate(rx_signals[_b], rx_signals[_a], mode='full'))
_lags = np.arange(_xcorr.size) - (buffer_len - 1)
_peak = _lags[np.argmax(_xcorr)]
_ax2.plot(_lags, _xcorr, 'o-', markersize=5, label='correlation samples')
_ax2.axvline(_peak, color='red', linestyle='--', label=f'integer peak = {_peak} samples')
_ax2.set_xlim(_peak - 6, _peak + 6)
_ax2.set_xlabel('lag [samples]'); _ax2.set_ylabel('|cross-correlation|')
_ax2.set_title(f'Cross-correlation of Rx{_b} vs Rx{_a}')
_ax2.legend()
_ax2.grid()
#if os.path.isdir('../_images'):
#    fig1.savefig('../_images/tdoa_python_integer.svg', bbox_inches='tight')
fig1.tight_layout()
fig1

# %%
# Subsample TDOA calc using a freq domain cross-correlation that was padded as a way to interpolate
U = 16 # correlation upsampling factor
half = (buffer_len + 1) // 2 # number of DC + positive-frequency bins
range_diff_sub = np.zeros(len(pairs)) # meters
for _k, (_a, _b) in enumerate(pairs):
    # Cross-correlation in the frequency domain
    _X = np.conj(np.fft.fft(rx_signals[_a])) * np.fft.fft(rx_signals[_b])

    # Insert zeros in the high-frequency MIDDLE: DC + positive freqs at the front, negative freqs at the back, so it stays a valid FFT layout.
    _X_padded = np.zeros(U * buffer_len, dtype=complex) 
    _X_padded[:half] = _X[:half]
    _X_padded[U * buffer_len - (buffer_len - half):] = _X[half:]

    # Now IFFT to finish the crosscorrelation
    _xcorr = np.abs(np.fft.ifft(_X_padded)) * U

    # Peak index -> signed lag; indices past the midpoint are negative lags
    _peak_idx = np.argmax(_xcorr)
    if _peak_idx > U * buffer_len // 2:
        _peak_idx -= U * buffer_len
    _peak_lag = _peak_idx / U # sub-sample lag, +ve => Rx_b farther
    range_diff_sub[_k] = (_peak_lag / sample_rate) * c # meters

print("METHOD 2 (sub-sample, zero-padded FFT)")
print(" Pair  |  true range diff [m] | measured range diff [m]")
for _k, (_a, _b) in enumerate(pairs):
    _true_rd = true_distances[_b] - true_distances[_a]
    print(f"Rx{_b}-Rx{_a} |     {_true_rd:9.1f}        |    {range_diff_sub[_k]:9.1f}")

# %%
# 8. FIGURE 2: the sub-sample result, same layout as Figure 1.
fig2, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: the hyperbolas from the refined range differences.
plot_hyperbolas(_ax1, range_diff_sub)
_ax1.set_title('TDOA hyperbolas')

# Right: coarse (1 sample/lag) correlation as dots vs the U-times upsampled
# correlation as a smooth curve, so the sub-sample shift is visible.
_a, _b = pairs[1]                        # the (Rx0, Rx2) pair
_X = np.conj(np.fft.fft(rx_signals[_a])) * np.fft.fft(rx_signals[_b])
cc_coarse = np.abs(np.fft.ifft(_X))
_X_padded = np.zeros(U * buffer_len, dtype=complex)
_X_padded[:half] = _X[:half]
_X_padded[U * buffer_len - (buffer_len - half):] = _X[half:]
cc_fine = np.abs(np.fft.ifft(_X_padded)) * U
lags_coarse = np.where(np.arange(buffer_len) <= buffer_len // 2, np.arange(buffer_len), np.arange(buffer_len) - buffer_len)
lags_fine = np.arange(U * buffer_len) / U
lags_fine = np.where(lags_fine <= buffer_len / 2, lags_fine, lags_fine - buffer_len)
_peak = lags_coarse[np.argmax(cc_coarse)]
subsample_peak = lags_fine[np.argmax(cc_fine)]
# The lag axes wrap from + back to - partway through, so sort before plotting,
# otherwise the connecting line jumps across the figure.
order_c = np.argsort(lags_coarse)
order_f = np.argsort(lags_fine)
_ax2.plot(lags_coarse[order_c], cc_coarse[order_c], 'o', markersize=6, label='coarse (1 sample/lag)')
_ax2.plot(lags_fine[order_f], cc_fine[order_f], '.-', label=f'{U}x interpolation')
_ax2.axvline(subsample_peak, color='red', linestyle='--',
            label=f'sub-sample peak = {subsample_peak:.3f}')
_ax2.set_xlim(_peak - 6, _peak + 6)
_ax2.set_xlabel('lag [samples]'); _ax2.set_ylabel('|cross-correlation|')
_ax2.set_title(f'Cross-correlation of Rx{_b} vs Rx{_a}')
_ax2.legend()
_ax2.grid()
fig2.tight_layout()
#if os.path.isdir('../_images'):
#    fig2.savefig('../_images/tdoa_python_subsample.svg', bbox_inches='tight')
fig2

# %%
# Heatmap portion: evaluate the TDOA cost on the grid and display it under the hyperbola map
cost = np.zeros_like(GX)
for _k, (_a, _b) in enumerate(pairs):
    cost += ((rx_dist[_b] - rx_dist[_a]) - range_diff_sub[_k])**2 # squared mismatch for this pair, summed over pairs

# The best estimate is simply the grid cell with the lowest cost
iy, ix = np.unravel_index(np.argmin(cost), cost.shape)
emitter_grid = np.array([grid_x[ix], grid_y[iy]])
print("Grid estimate:", emitter_grid) # ~[153, 355]

fig3, _ax1 = plt.subplots(1, 1, figsize=(7, 6))
# Invert the cost into a likelihood-style surface so higher (brighter) = more likely emitter location
likelihood = -np.log10(cost + 1e-9)
# Cap the colormap a bit below the peak so the bright region around the emitter spreads out and is easier to see
vmax = likelihood.min() + 0.5 * (likelihood.max() - likelihood.min()) # 0.5 was adjusted manually to look good
im = _ax1.imshow(likelihood, origin='lower', cmap='viridis', vmax=vmax,
                extent=[grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]])
fig3.colorbar(im, ax=_ax1, label='likelihood (higher = more likely)')
# Overlay the hyperbolas, receivers, true Tx, and the grid-search estimate
#_ax1.scatter(*emitter_grid, c='white', marker='x', s=120, linewidths=2, label='Grid estimate', zorder=6)
plot_hyperbolas(_ax1, range_diff_sub, rx_color='tab:cyan', label_color='w')
fig3.tight_layout()
#if os.path.isdir('../_images'):
#    fig3.savefig('../_images/tdoa_python_heatmap.svg', bbox_inches='tight')

# %%
# Kept in its own cell, otherwise marimo would display fig3 twice
plt.show()
