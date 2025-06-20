import matplotlib.pyplot as plt
import numpy as np

sample_rate = 1e6
symbol_len_samples = 16
num_symbols = 1000
freqs = [-0.3e6, -0.1e6, 0.1e6, 0.3e6]

# Gaussian pulse shaping FIR filter
symbol_period = symbol_len_samples / sample_rate
alpha = 0.5 # bandwidth-time product, length of filter will be 1/alpha symbols
sigma = np.sqrt(np.log(2)) / (2*np.pi*alpha)
t = np.arange(int(symbol_period * sample_rate / alpha)) / sample_rate - symbol_period / (2 * alpha)
h = 1 / (np.sqrt(2*np.pi) * sigma * symbol_period) * np.exp(-t**2 / (2 * sigma**2 * symbol_period**2))
h /= sample_rate # normalize
if False:
    print(len(h))
    print(np.sum(h)) # sum of taps should be 1 if normalized
    plt.plot(h, '.-')
    plt.show()
    exit()

x = np.zeros((4, num_symbols*symbol_len_samples), dtype=complex) # generate each freq separately at first

for i in range(num_symbols):
    two_bits = np.random.randint(4)
    fi = freqs[two_bits]
    x[two_bits, i*symbol_len_samples:(i+1)*symbol_len_samples] = np.ones(symbol_len_samples, dtype=complex)

# Apply the pulse shaping filter, then frequency shift to each of the four signals, then combine
for i in range(4):
    x[i, :] = np.convolve(x[i, :], h, mode='same')
    x[i, :] -= np.mean(x[i, :]) # remove DC offset, not sure why I needed to add this
    t = np.arange(0, num_symbols*symbol_len_samples) / sample_rate
    x[i, :] = np.exp(2j * np.pi * freqs[i] * t) * x[i, :]
x = np.sum(x, axis=0)

# add some noise
n = np.random.randn(len(x)) + 1j * np.random.randn(len(x))
r = x + 0.1 * n

# PSD
R = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(r))) ** 2)
R -= np.max(R) # normalize

fig, ax = plt.subplots(figsize=(6, 3))
f = np.linspace(sample_rate/-2, sample_rate/2, len(R))
ax.plot(f/1e6, R)
ax.set_xlabel("Frequency [MHz]")
ax.set_ylabel("Power Spectral Density [dB]")
ax.axis((-0.5, 0.5, -30, 5))
plt.show()
fig.savefig('../_images/fsk.svg', bbox_inches='tight')
