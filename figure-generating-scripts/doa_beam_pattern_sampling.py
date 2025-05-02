import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.animation import FuncAnimation

d = 0.5
Nr = 16

# Using beam pattern sampling to come up with the beamforming weights
# Van trees page 124
B_theta = np.ones(Nr) # desired beam pattern
phi = (np.arange(Nr) - (Nr - 1)/2) * 2 * np.pi / Nr # eq 3.92
B = np.conj(B_theta) * np.exp(-1j * phi * (Nr - 1)/2) # eq 3.94
b = np.fft.ifft(B)
n = np.arange(Nr)
w = b * np.exp(-1j * n * np.pi * (Nr - 1)/Nr) # eq 3.101
w = w / np.sum(w) # normalize weights

# Plot beam pattern
N_fft = 1024
w = np.conj(w) # or else our answer will be negative/inverted
w = w.squeeze()
w_padded = np.concatenate((w, np.zeros(N_fft - Nr))) # zero pad to N_fft elements to get more resolution in the FFT
w_fft_dB = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(w_padded)))**2) # magnitude of fft in dB
w_fft_dB -= np.max(w_fft_dB) # normalize to 0 dB at peak
theta_bins = np.arcsin(np.linspace(-1, 1, N_fft)) # Map the FFT bins to angles in radians
fig, ax = plt.subplots()
ax.plot(theta_bins * 180/np.pi, w_fft_dB) # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_xlabel("Theta [Degrees]")
ax.set_ylabel("Beam Pattern [dB]")
ax.set_ylim((-30, 0))
ax.grid()
plt.show()
