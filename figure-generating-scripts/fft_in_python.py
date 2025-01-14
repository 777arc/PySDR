import matplotlib.pyplot as plt
import numpy as np

t = np.arange(100)
s = np.sin(0.15*2*np.pi*t)
fig, (ax1) = plt.subplots(1, 1, figsize=(5, 3))
plt.subplots_adjust(wspace=0.4)
ax1.plot(s, '.-')
ax1.set_xlabel("Sample Index")
ax1.set_ylabel("Signal Amplitude")
ax1.grid()
fig.savefig('../_images/fft-python1.svg', bbox_inches='tight')


S = np.fft.fft(s)
S_mag = np.abs(S)
S_phase = np.angle(S)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))
plt.subplots_adjust(wspace=0.4)
ax1.plot(t,S_mag,'.-')
ax1.set_ylabel("FFT Magnitude")
ax1.grid()
ax2.plot(t,S_phase,'.-')
ax2.set_xlabel("FFT Index")
ax2.set_ylabel("FFT Phase [radians]")
ax2.grid()
fig.savefig('../_images/fft-python2.svg', bbox_inches='tight')


Fs = 1 # Hz
N = 100 # number of points to simulate, and our FFT size
t = np.arange(N) # because our sample rate is 1 Hz
s = np.sin(0.15*2*np.pi*t)
S = np.fft.fftshift(np.fft.fft(s))
S_mag = np.abs(S)
S_phase = np.angle(S)
f = np.arange(Fs/-2, Fs/2, Fs/N)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5))
plt.subplots_adjust(wspace=0.4)
ax1.plot(f, S_mag,'.-')
ax1.set_ylabel("FFT Magnitude")
ax1.grid()
ax1.text(-0.02, 2, '0.15 Hz', color='r')
ax1.vlines(x=0.15, ymin=-5, ymax=5, colors='r')
ax1.set_ylim(-2, 55)
ax1.set_xlim(-0.5, 0.5)
ax1.set_xticks(np.arange(-0.5, 0.6, 0.25))
ax2.plot(f, S_phase,'.-')
ax2.set_xlabel("Frequency [Hz]")
ax2.set_ylabel("FFT Phase [radians]")
ax2.set_xlim(-0.5, 0.5)
ax2.set_xticks(np.arange(-0.5, 0.6, 0.25))
ax2.grid()
fig.savefig('../_images/fft-python5.svg', bbox_inches='tight')


plt.show()