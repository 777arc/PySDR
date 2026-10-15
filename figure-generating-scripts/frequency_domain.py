# %%
import os
import numpy as np
import matplotlib.pyplot as plt

# %%
# FFT in Python, start with a sine wave at 0.15 Hz, using a sample rate of 1 Hz
t = np.arange(100)
s = np.sin(0.15*2*np.pi*t)
fig1, _ax1 = plt.subplots(1, 1, figsize=(5, 3))
_ax1.plot(s, '.-')
_ax1.set_xlabel("Sample Index")
_ax1.set_ylabel("Signal Amplitude")
_ax1.grid()
#if os.path.isdir('../_images'):
#    fig1.savefig('../_images/fft-python1.svg', bbox_inches='tight')
fig1

# %%
# Magnitude and phase of the FFT, before doing the FFT shift
S = np.fft.fft(s)
S_mag = np.abs(S)
S_phase = np.angle(S)
fig2, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(5, 5))
_ax1.plot(t, S_mag, '.-')
_ax1.set_ylabel("FFT Magnitude")
_ax1.grid()
_ax2.plot(t, S_phase, '.-')
_ax2.set_xlabel("FFT Index")
_ax2.set_ylabel("FFT Phase [radians]")
_ax2.grid()
#if os.path.isdir('../_images'):
#    fig2.savefig('../_images/fft-python2.svg', bbox_inches='tight')
fig2

# %%
# Final version, with the FFT shift and a proper frequency axis
Fs = 1 # Hz
N = 100 # number of points to simulate, and our FFT size
_t = np.arange(N) # because our sample rate is 1 Hz
_s = np.sin(0.15*2*np.pi*_t)
_S = np.fft.fftshift(np.fft.fft(_s))
_S_mag = np.abs(_S)
_S_phase = np.angle(_S)
f = np.arange(Fs/-2, Fs/2, Fs/N)
fig3, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(5, 5))
_ax1.plot(f, _S_mag, '.-')
_ax1.set_ylabel("FFT Magnitude")
_ax1.grid()
_ax1.text(-0.02, 2, '0.15 Hz', color='r')
_ax1.vlines(x=0.15, ymin=-5, ymax=5, colors='r')
_ax1.set_ylim(-2, 55)
_ax1.set_xlim(-0.5, 0.5)
_ax1.set_xticks(np.arange(-0.5, 0.6, 0.25))
_ax2.plot(f, _S_phase, '.-')
_ax2.set_xlabel("Frequency [Hz]")
_ax2.set_ylabel("FFT Phase [radians]")
_ax2.set_xlim(-0.5, 0.5)
_ax2.set_xticks(np.arange(-0.5, 0.6, 0.25))
_ax2.grid()
#if os.path.isdir('../_images'):
#    fig3.savefig('../_images/fft-python5.svg', bbox_inches='tight')
fig3

# %%
# Window functions in the time and frequency domain
fig4, [_ax, _ax2] = plt.subplots(1, 2, figsize=(10, 4))
_n = 1000
_rect = np.ones(_n)
_rect[0] = 0
_rect[-1] = 0
_ax.plot(_rect, '--')
_ax.plot(np.hamming(_n))
_ax.plot(np.hanning(_n))
_ax.plot(np.bartlett(_n))
_ax.plot(np.blackman(_n))
_ax.plot(np.kaiser(_n, 4))
_ax.set_title("Time Domain")
_ax.legend(['Rectangular', 'Hamming', 'Hanning', 'Bartlett', 'Blackman', 'Kaiser'], fontsize='large', bbox_to_anchor=(0.5, 0.35), framealpha=1)
_ax.spines['left'].set_position('zero') # set the x-spine
_ax.spines['right'].set_color('none') # turn off the right spine/ticks
_ax.yaxis.tick_left()
_ax.spines['bottom'].set_position('zero') # set the y-spine
_ax.spines['top'].set_color('none') # turn off the top spine/ticks
_ax.xaxis.tick_bottom()
_ax.set_ylabel("Amplitude / Value")
_ax.text(_n-10, -0.07, "N")
_ax.text(-10, -0.07, "0")
_ax.set_xticklabels([]) # turn off tick numbering/labels
_N = 10000
_f = np.linspace(0, 100/_N, 100)
_ax2.plot(_f, 10*np.log10(np.abs(np.fft.fft(_rect, _N)[0:100]))-30, '--')
_ax2.plot(_f, 10*np.log10(np.abs(np.fft.fft(np.hamming(_n), _N)[0:100]))-27-0.3)
_ax2.plot(_f, 10*np.log10(np.abs(np.fft.fft(np.hanning(_n), _N)[0:100]))-27)
_ax2.plot(_f, 10*np.log10(np.abs(np.fft.fft(np.bartlett(_n), _N)[0:100]))-27)
_ax2.plot(_f, 10*np.log10(np.abs(np.fft.fft(np.blackman(_n), _N)[0:100]))-27+0.75)
_ax2.plot(_f, 10*np.log10(np.abs(np.fft.fft(np.kaiser(_n, 4), _N)[0:100]))-27-0.8)
_ax2.axis([0, np.max(_f), -50, 3])
_ax2.set_xlabel("Normalized Frequency")
_ax2.set_ylabel("dB")
_ax2.set_title("Frequency Domain")
#if os.path.isdir('../_images'):
#    fig4.savefig('../_images/windows.svg', bbox_inches='tight')
fig4

# %%
# Same sine wave as before, but with a Hamming window applied right before the FFT
_s = s * np.hamming(N)
_S = np.fft.fftshift(np.fft.fft(_s))
_S_mag = np.abs(_S)
fig5, _ax1 = plt.subplots(1, 1, figsize=(5, 3))
_ax1.plot(f, _S_mag, '.-')
_ax1.set_xlabel("Frequency [Hz]")
_ax1.set_ylabel("FFT Magnitude")
_ax1.grid()
fig5

# %%
# Spectrogram, the example signal is a tone plus noise
sample_rate = 1e6
_t = np.arange(1024*1000)/sample_rate # time vector
_f = 50e3 # freq of tone
x = np.sin(2*np.pi*_f*_t) + 0.2*np.random.randn(len(_t))

# Here is what it looks like in the time domain (first 200 samples)
fig6, _ax1 = plt.subplots()
_ax1.plot(x[0:200])
_ax1.set_xlabel("Time")
#if os.path.isdir('../_images'):
#    fig6.savefig('../_images/spectrogram_time.svg', bbox_inches='tight')
fig6

# %%
fft_size = 1024
num_rows = len(x) // fft_size # // is an integer division which rounds down
spectrogram = np.zeros((num_rows, fft_size))
for i in range(num_rows):
    spectrogram[i,:] = 10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(x[i*fft_size:(i+1)*fft_size])))**2)

# Time starts at the top and goes down, eg sample x[0] will be part of the top row displayed
fig7, _ax1 = plt.subplots()
_ax1.imshow(spectrogram, aspect='auto', extent = (sample_rate/-2/1e6, sample_rate/2/1e6, len(x)/sample_rate, 0))
_ax1.set_xlabel("Frequency [MHz]")
_ax1.set_ylabel("Time [s]")
#if os.path.isdir('../_images'):
#    fig7.savefig('../_images/spectrogram.svg', bbox_inches='tight')
fig7

# %%
# FFT Implementation, a simple recursive version of the Cooley-Tukey FFT
def fft(x):
    N = len(x)
    if N == 1:
        return x
    twiddle_factors = np.exp(-2j * np.pi * np.arange(N//2) / N)
    x_even = fft(x[::2]) # yay recursion!
    x_odd = fft(x[1::2])
    return np.concatenate([x_even + twiddle_factors * x_odd,
                           x_even - twiddle_factors * x_odd])

# Simulate a tone + noise
_sample_rate = 1e6
f_offset = 0.2e6 # 200 kHz offset from carrier
_N = 1024
_t = np.arange(_N)/_sample_rate
_s = np.exp(2j*np.pi*f_offset*_t)
_n = (np.random.randn(_N) + 1j*np.random.randn(_N))/np.sqrt(2) # unity complex noise
r = _s + _n # 0 dB SNR

# Perform fft, fftshift, convert to dB
X = fft(r)
X_shifted = np.roll(X, _N//2) # equivalent to np.fft.fftshift
X_mag = 10*np.log10(np.abs(X_shifted)**2)

# Plot results
_f = np.linspace(_sample_rate/-2, _sample_rate/2, _N)/1e6 # plt in MHz
fig8, _ax1 = plt.subplots()
_ax1.plot(_f, X_mag)
_ax1.plot(_f[np.argmax(X_mag)], np.max(X_mag), 'rx') # show max
_ax1.grid()
_ax1.set_xlabel('Frequency [MHz]')
_ax1.set_ylabel('Magnitude [dB]')
#if os.path.isdir('../_images'):
#    fig8.savefig('../_images/fft_in_python.svg', bbox_inches='tight')
fig8

# %%
# Kept in its own cell, otherwise marimo would display the figures twice
plt.show()
