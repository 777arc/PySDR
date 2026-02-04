import numpy as np
import matplotlib.pyplot as plt

# Long Zadoff-Chu sequence
N = 839  # Length of Zadoff-Chu sequence
u = 25  # Root of ZC sequence
t = np.arange(N)
zadoff_chu = np.exp(-1j * np.pi * u * t * (t + 1) / N)

# Create AWGN and stick the ZC sequence in a random spot
signal_length = 10 * N
offset = np.random.randint(N, signal_length - N)
print(f"True offset: {offset}")
snr_db = -15
noise_power = 1 / (2 * (10**(snr_db / 10)))
signal = np.sqrt(noise_power/2) * (np.random.randn(signal_length) + 1j * np.random.randn(signal_length))
signal[offset:offset+N] += zadoff_chu

plt.figure(0)
plt.plot(np.abs(signal))
plt.xlabel('Sample Index')
plt.ylabel('Signal Magnitude [Linear]')
plt.axvline(float(offset), 0, 1, color='r', linestyle=':')
plt.text(float(offset), -0.5, 'True Offset', color='r', verticalalignment='bottom')
plt.grid()
plt.tight_layout()
plt.savefig('../_images/detection_basic_1.svg', bbox_inches='tight')

# Correlator
correlation = np.abs(np.correlate(signal, zadoff_chu, mode='valid') / N)**2 

plt.figure(1)
plt.plot(correlation)
plt.xlabel('Sample Index')
plt.ylabel('Correlation Magnitude [Linear]')
plt.axvline(float(offset), 0, 0.2, color='r', linestyle=':')
plt.text(float(offset), -0.05, 'True Offset', color='r', verticalalignment='bottom')
plt.grid()
plt.tight_layout()
plt.savefig('../_images/detection_basic_2.svg', bbox_inches='tight')
plt.show()
