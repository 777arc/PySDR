import numpy as np
import matplotlib.pyplot as plt

# Parameters
num_symbols = 20000
sps = 32         # use 32 for time domain parts, 8 for PSD
span = 6         # filter span in symbols (each side)
mode = 'OQPSK'   # 'QPSK' or 'OQPSK'

# Generate QPSK symbols
bits = np.random.randint(0, 4, num_symbols)
symbols = np.exp(1j * (np.pi/4 + bits * np.pi/2)).astype(complex)  # points at 45°, 135°, 225°, 315°

if False:
    # RC filter
    beta = 0.35      # roll-off factor
    t = np.arange(-span * sps, span * sps + 1) / sps  # in symbol periods
    h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2 + 1e-20)
else:
    # Half-sine pulse shape
    t = np.arange(sps)
    h = np.sin(np.pi * t / sps)
    output_file = '../_images/msk_magnitude.svg'

if mode == 'QPSK':
    upsampled = np.zeros(num_symbols * sps, dtype=complex)
    upsampled[::sps] = symbols
    signal = np.convolve(upsampled, h, mode='same')
    output_file = '../_images/qpsk_magnitude.svg'

elif mode == 'OQPSK':
    # Delay Q impulses by half a symbol before filtering so the pulse shaping filter handles the ramp-up naturally (no post-filter roll/zero-fill artifact)
    # half = sps // 2
    # I_up = np.zeros(num_symbols * sps)
    # Q_up = np.zeros(num_symbols * sps)
    # I_up[::sps] = np.real(symbols)
    # Q_up[half::sps] = np.imag(symbols)
    # I_filt = np.convolve(I_up, h, mode='same')
    # Q_filt = np.convolve(Q_up, h, mode='same')
    # signal = I_filt + 1j * Q_filt
    #output_file = '../_images/oqpsk_magnitude.svg'
    #mode = "MSK" # TEMPORARY

    bits = np.random.randint(0, 2, num_symbols)
    symbols = 2 * bits - 1 # map {0,1} → {-1, +1}

    # Build the instantaneous frequency deviation
    mod_index = 0.5
    t = np.arange(num_symbols * sps / 2) / (sps / 2)
    freq_dev = np.zeros(num_symbols * sps // 2)
    for k, a in enumerate(symbols):
        freq_dev[k * sps // 2 : (k + 1) * sps // 2] = a * mod_index / 2.0

    phase = 2.0 * np.pi * np.cumsum(freq_dev) / (sps / 2) # accumulate phase
    signal = np.exp(1j * phase)

    mode = "CPFSK"
    output_file = '../_images/cpfsk_magnitude.svg'

#signal *= np.sqrt(2)

# Plot
N = 10
fig, axes = plt.subplots(2, 1, figsize=(7, 4), tight_layout=True)
axes[0].plot(signal.real[:N*sps], label='I')
axes[0].plot(signal.imag[:N*sps], label='Q', alpha=0.7)
axes[0].axhline(1, color='gray', linestyle='--', linewidth=1)
axes[0].axhline(-1, color='gray', linestyle='--', linewidth=1)
# NOTE THE -1 IS ONLY FOR CPFSK TO ALIGN THINGS
for x in range(-1, N * sps, sps):
    axes[0].axvline(x, color='gray', linestyle='--', linewidth=1)
if mode == 'OQPSK' or mode == 'MSK' or mode == 'CPFSK':
    for x in range(sps // 2 - 1, N * sps, sps):
        axes[0].axvline(x, color='blue', linestyle='--', linewidth=1, alpha=0.5)
axes[0].set_title(mode)
axes[0].legend()
axes[1].plot(np.abs(signal[:N*sps]))
axes[1].set_ylabel('Magnitude')
axes[1].set_xlabel('Sample Index (Time)')
axes[1].set_ylim(bottom=0, top=1.2)
axes[1].grid(True)
plt.savefig(output_file, bbox_inches='tight')
plt.show()

# Plot the PSD
# psd = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(signal)))**2)
# psd -= np.max(psd)  # Normalize to 0 dB max
# f = np.linspace(-0.5, 0.5, len(psd))
# plt.figure(figsize=(7, 4), tight_layout=True)
# plt.plot(f, psd)
# plt.ylim(bottom=-80, top=5)
# #plt.title(f'{mode}')
# plt.title('QPSK or OQPSK with RC Pulse Shaping')
# plt.xlabel('Normalized Frequency (cycles/sample)')
# plt.ylabel('PSD [dB]')
# plt.grid(True)
# plt.savefig('../_images/qpsk_psd.svg', bbox_inches='tight')
# plt.show()
