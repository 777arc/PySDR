import matplotlib.pyplot as plt
import numpy as np

# Shows the fs = 4*f_IF trick, using the sampling chapter's example point 0.7 - 0.4j
I = 0.7
Q = -0.4
f_IF = 1.0  # arbitrary, everything is relative
fs = 4 * f_IF
num_samples = 12

t = np.linspace(0, (num_samples - 1) / fs, 1000)
x_t = I * np.cos(2 * np.pi * f_IF * t) + Q * np.sin(2 * np.pi * f_IF * t)

n = np.arange(num_samples)
x = I * np.cos(np.pi / 2 * n) + Q * np.sin(np.pi / 2 * n) # sampled IF signal: I, Q, -I, -Q, ...
lo_cos = np.round(np.cos(np.pi / 2 * n)) # 1, 0, -1, 0, ...
lo_sin = np.round(np.sin(np.pi / 2 * n)) # 0, 1, 0, -1, ...

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.subplots_adjust(hspace=0.45)

# Top: real IF signal and its samples
ax1.plot(n[0] + t * fs, x_t, 'g')
ax1.stem(n, x, linefmt='b:', markerfmt='bo', basefmt='k-')
labels = ['I', 'Q', '-I', '-Q']
for i in range(num_samples):
    offset = 0.15 if x[i] >= 0 else -0.35
    ax1.text(n[i], x[i] + offset, labels[i % 4], color='red', fontsize=14, ha='center')
ax1.set_title(r'Real IF Signal Sampled at $f_s = 4 f_{IF}$')
ax1.set_ylabel('Amplitude')
ax1.set_ylim(-1.3, 1.3)

# Middle: the two digital LOs
ax2.stem(n - 0.08, lo_cos, linefmt='b-', markerfmt='bo', basefmt='k-', label=r'$\cos(\pi n/2)$ = 1, 0, -1, 0, ...')
ax2.stem(n + 0.08, lo_sin, linefmt='r-', markerfmt='rs', basefmt='k-', label=r'$\sin(\pi n/2)$ = 0, 1, 0, -1, ...')
ax2.set_title('Digital LO (no multiplications needed, just sign flips and zeros)')
ax2.set_ylabel('Amplitude')
ax2.set_ylim(-1.4, 2.0)
ax2.legend(loc='upper center', ncol=2, fontsize=10)

# Bottom: mixer outputs
ax3.stem(n - 0.08, x * lo_cos, linefmt='b-', markerfmt='bo', basefmt='k-', label='I branch (samples x cos)')
ax3.stem(n + 0.08, x * lo_sin, linefmt='r-', markerfmt='rs', basefmt='k-', label='Q branch (samples x sin)')
ax3.axhline(I, color='b', linestyle='--', linewidth=0.8)
ax3.axhline(Q, color='r', linestyle='--', linewidth=0.8)
ax3.text(num_samples - 0.4, I + 0.07, 'I = 0.7', color='b', fontsize=12, ha='right')
ax3.text(num_samples - 0.4, Q - 0.25, 'Q = -0.4', color='r', fontsize=12, ha='right')
ax3.set_title('After Mixing: I and Q Are Recovered on Alternating Samples')
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Amplitude')
ax3.set_ylim(-0.9, 1.5)
ax3.set_xlim(-0.5, num_samples - 0.3)
ax3.set_xticks(n)
ax3.legend(loc='upper center', ncol=2, fontsize=10)

plt.show()

fig.savefig('../_images/fs_4_fif.svg', bbox_inches='tight')
