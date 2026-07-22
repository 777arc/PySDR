import matplotlib.pyplot as plt
import numpy as np

np.random.seed(5)

# BPSK + RRC pulse shaping, same style of setup as the chapter's Python exercise
num_symbols = 400
sps = 32
beta = 0.35
span = 8  # filter span in symbols (each side)

# Random BPSK symbols
symbols = np.random.randint(0, 2, num_symbols) * 2 - 1

# Upsample (impulses spaced by sps)
x = np.zeros(num_symbols * sps)
x[::sps] = symbols

# Raised-cosine filter (Tx + Rx combined for this illustration)
t = np.arange(-span * sps, span * sps + 1) / sps
h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2 + 1e-20)
h /= np.max(np.convolve(np.ones(1), h))  # keep peaks near +/-1

y = np.convolve(x, h, mode='same')

# Build the eye diagram: chop the signal into 2-symbol-wide slices and overlay
span_samps = 2 * sps
n_traces = 250
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
start = span * sps  # skip filter transient
for i in range(n_traces):
    s = start + i * sps
    seg = y[s:s + span_samps]
    if len(seg) < span_samps:
        break
    ax.plot(np.arange(span_samps) / sps - 1, seg, color='#1f77b4', alpha=0.15, linewidth=1, rasterized=True)

# Mark the ideal sampling instant (center of the eye)
ax.axvline(0, color='k', linestyle='--', linewidth=1)
ax.text(0.02, 1.35, 'ideal sample time', fontsize=11)

ax.set_xlabel('Time (symbol periods)', fontsize=12)
ax.set_ylabel('Amplitude', fontsize=12)
ax.set_xlim(-1, 1)
ax.set_ylim(-1.6, 1.6)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
fig.savefig('../_images/eye_diagram.svg', bbox_inches='tight', dpi=100)
